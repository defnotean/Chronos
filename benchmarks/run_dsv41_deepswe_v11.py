"""
Official 1:1 DeepSWE v1.1 (datacurve-ai/deep-swe, 113 Harbor tasks) Benchmark Runner
for Resident Chronos DeepSeek-V4.1-Flash (`GLOBAL_DSV41_MODEL`).

Executes `mini-swe-agent` trajectories inside isolated `udocker` agent containers
with zero test/solution visibility, collects `model.patch` via each task's
`[[verifier.collect]]` command, and grades in a separate pristine verifier
container via `/tests/test.sh` -> `/logs/verifier/reward.json`.
"""
from __future__ import annotations

import argparse
import builtins
from datetime import datetime, timezone
import gc
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_REPO_DIR = "/content/deep-swe"
DEFAULT_UDOCKER_DIR = "/content/udocker_store"
DEFAULT_TASK_ORDER_JSON = "/content/dsv41_deepswe_v11_task_order.json"
DEFAULT_PROGRESS_JSONL = "/content/v8_deepswe_v11_progress.jsonl"
DEFAULT_REPORT_JSON = "/content/dsv41_v8_deepswe_v11_report.json"


def wilson_ci_95(passed: int, n: int) -> tuple[float, float]:
    """Return 95% Wilson score confidence interval percentages `(lo, hi)`."""
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = float(passed) / float(n)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return (round(100.0 * max(0.0, center - half), 2), round(100.0 * min(1.0, center + half), 2))

BASH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a non-interactive bash command in /app inside the Linux container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]

SYSTEM_PROMPT_DSML = """You are an expert autonomous software engineering agent working in /app inside a Linux container.

Use the `bash` tool to inspect the codebase, reproduce the issue, implement clean fixes, run tests to verify, commit all changes with git, and submit by running `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as your final command."""

INSTANCE_TEMPLATE_DSML = """Please solve this issue: {task}

You can execute bash commands and edit files via the `bash` tool to implement the necessary changes.

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script or run existing tests to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running tests
5. Test edge cases to ensure your fix is robust
6. Commit all your changes (`git status`, `git add`, `git commit`), and then submit and finish your work by issuing ONLY the command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

## Important Rules

1. Use the `bash` tool for every action (never simulate or write command output yourself)
2. Do NOT run interactive commands (`vi`, `nano`, `less`, `more`, interactive shells)
3. Directory and environment variable changes are not persistent across `bash` calls (each runs from `/app`)
4. Keep inspection commands concise (avoid dumping huge files) and watch the `<budget>` indicator so you leave enough turns and context to implement, test, commit, and submit your changes.
5. Avoid inspecting external/standard-library source trees outside `/app` unless strictly necessary; once you understand the relevant files in `/app`, move promptly to implementing edits, running tests, and committing your fix."""

SYSTEM_PROMPT = SYSTEM_PROMPT_DSML
INSTANCE_TEMPLATE = INSTANCE_TEMPLATE_DSML

CONTAINER_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
    "GIT_TERMINAL_PROMPT": "0",
}

_BASH_BLOCK_RE = re.compile(
    r"```(?:bash|sh|mswea_bash_command)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_DSML_CMD_RE = re.compile(
    r'<｜DSML｜\s*parameter\s+name="command"(?:\s+string="(?:true|false)")?\s*>(.*?)</｜DSML｜\s*parameter>',
    re.DOTALL,
)
_DSML_CMD_MISSING_GT_RE = re.compile(
    r'<｜DSML｜\s*parameter\s+name="command"[^<]*?string="(?:true"\s*>\s*)?(.*?)</｜DSML｜\s*parameter>',
    re.DOTALL,
)
_DSML_BLOCK_STRIP_RE = re.compile(r'<｜DSML｜\s*(?:function_)?calls>[\s\S]*$', re.DOTALL)
_UDOCKER_BANNER_RE = re.compile(
    r"\s*\*{20,}[\s\S]*?STARTING [0-9a-f-]+[\s\S]*?\*{20,}\s*(?:executing:\s*\S+\s*)?",
    re.IGNORECASE,
)


def _ensure_paths() -> None:
    for p in (
        "/content",
        "/content/chronos_1t",
        "/content/scripts",
        "/content/benchmarks",
        "/content/dsv41_standalone/inference",
        "/content/dsv41_standalone/encoding",
    ):
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)


def load_harbor_task(task_dir: str | Path) -> dict[str, Any]:
    """Load `task.toml`, `instruction.md`, and `tests/config.json` metadata."""
    tdir = Path(task_dir)
    toml_data = tomllib.loads((tdir / "task.toml").read_text(encoding="utf-8"))
    meta = toml_data.get("metadata", {})
    env_cfg = toml_data.get("environment", {})
    ver_cfg = toml_data.get("verifier", {})
    collect_cfg = ver_cfg.get("collect", [])
    test_cfg_path = tdir / "tests" / "config.json"
    test_cfg = json.loads(test_cfg_path.read_text(encoding="utf-8")) if test_cfg_path.exists() else {}

    task_id = str(meta.get("task_id") or tdir.name)
    docker_image = str(env_cfg.get("docker_image") or "")
    if not docker_image:
        raise ValueError(f"Missing [environment].docker_image in {tdir / 'task.toml'}")

    return {
        "task_id": task_id,
        "task_dir": str(tdir),
        "language": str(meta.get("language") or "unknown").lower(),
        "category": str(meta.get("category") or ""),
        "repository_url": str(meta.get("repository_url") or ""),
        "base_commit_hash": str(meta.get("base_commit_hash") or test_cfg.get("base_commit") or ""),
        "docker_image": docker_image,
        "verifier_timeout_sec": float(ver_cfg.get("timeout_sec") or 1800.0),
        "collect_commands": [str(c.get("command")) for c in collect_cfg if c.get("command")],
        "instruction": (tdir / "instruction.md").read_text(encoding="utf-8"),
        "f2p_total": len(test_cfg.get("f2p_node_ids") or []),
        "p2p_total": len(test_cfg.get("p2p_node_ids") or []),
    }


def build_or_load_task_order(
    repo_dir: str | Path = DEFAULT_REPO_DIR,
    out_json: str | Path = DEFAULT_TASK_ORDER_JSON,
) -> list[dict[str, Any]]:
    """Build or load the frozen deterministic 113-task schedule.

    Indices 0..24 are language-stratified across the task languages for the 25-task milestone;
    indices 25..112 are the remaining tasks in sorted task_id order.
    """
    out_p = Path(out_json)
    if out_p.exists():
        return json.loads(out_p.read_text(encoding="utf-8"))

    tasks_root = Path(repo_dir) / "tasks"
    all_tasks = [
        load_harbor_task(p)
        for p in sorted(tasks_root.iterdir())
        if p.is_dir() and (p / "task.toml").exists()
    ]
    by_lang: dict[str, list[dict[str, Any]]] = {}
    for t in all_tasks:
        by_lang.setdefault(t["language"], []).append(t)

    # Round-robin proportional allocation across languages for first 25 tasks
    total = len(all_tasks)
    milestone_target = min(25, total)
    milestone: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    lang_order = sorted(by_lang.keys(), key=lambda l: (-len(by_lang[l]), l))

    # Give every language proportional quota (at least 1 if present)
    quotas: dict[str, int] = {}
    for lang in lang_order:
        quotas[lang] = max(1, round(milestone_target * len(by_lang[lang]) / max(1, total)))
    while sum(quotas.values()) > milestone_target:
        largest = max(lang_order, key=lambda l: quotas[l])
        quotas[largest] -= 1
    while sum(quotas.values()) < milestone_target:
        for lang in lang_order:
            if quotas[lang] < len(by_lang[lang]):
                quotas[lang] += 1
                if sum(quotas.values()) == milestone_target:
                    break

    for lang in lang_order:
        group = by_lang[lang]
        q = quotas[lang]
        if q <= 0:
            continue
        # Evenly spread indices inside this language group
        picks = [group[(i * len(group)) // q] for i in range(q)]
        for pk in picks:
            if pk["task_id"] not in chosen_ids:
                chosen_ids.add(pk["task_id"])
                milestone.append(pk)

    milestone.sort(key=lambda t: (t["language"], t["task_id"]))
    remaining = sorted([t for t in all_tasks if t["task_id"] not in chosen_ids], key=lambda t: t["task_id"])
    ordered = milestone + remaining

    schedule = [
        {
            "order_idx": idx,
            "milestone25": idx < milestone_target,
            "task_id": t["task_id"],
            "language": t["language"],
            "docker_image": t["docker_image"],
            "base_commit_hash": t["base_commit_hash"],
            "f2p_total": t["f2p_total"],
            "p2p_total": t["p2p_total"],
        }
        for idx, t in enumerate(ordered)
    ]
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    return schedule


_NET_BLOCK_C_SRC = r"""
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>

static int _is_blocked_sa(const struct sockaddr *sa) {
    if (!sa) return 0;
    if (sa->sa_family == AF_INET) {
        const struct sockaddr_in *in = (const struct sockaddr_in *)sa;
        uint32_t ip = ntohl(in->sin_addr.s_addr);
        return ((ip >> 24) != 127);
    }
    if (sa->sa_family == AF_INET6) {
        const struct sockaddr_in6 *in6 = (const struct sockaddr_in6 *)sa;
        static const unsigned char lo[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};
        static const unsigned char v4p[12] = {0,0,0,0,0,0,0,0,0,0,0xff,0xff};
        if (memcmp(in6->sin6_addr.s6_addr, lo, 16) == 0) return 0;
        if (memcmp(in6->sin6_addr.s6_addr, v4p, 12) == 0 && in6->sin6_addr.s6_addr[12] == 127) return 0;
        return 1;
    }
    return 0;
}

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    static int (*real_fn)(int, const struct sockaddr *, socklen_t) = NULL;
    if (!real_fn) real_fn = (int (*)(int, const struct sockaddr *, socklen_t))dlsym(RTLD_NEXT, "connect");
    if (_is_blocked_sa(addr)) { errno = EACCES; return -1; }
    return real_fn(sockfd, addr, addrlen);
}

ssize_t sendto(int sockfd, const void *buf, size_t len, int flags, const struct sockaddr *dest_addr, socklen_t addrlen) {
    static ssize_t (*real_fn)(int, const void *, size_t, int, const struct sockaddr *, socklen_t) = NULL;
    if (!real_fn) real_fn = (ssize_t (*)(int, const void *, size_t, int, const struct sockaddr *, socklen_t))dlsym(RTLD_NEXT, "sendto");
    if (_is_blocked_sa(dest_addr)) { errno = EACCES; return -1; }
    return real_fn(sockfd, buf, len, flags, dest_addr, addrlen);
}

ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags) {
    static ssize_t (*real_fn)(int, const struct msghdr *, int) = NULL;
    if (!real_fn) real_fn = (ssize_t (*)(int, const struct msghdr *, int))dlsym(RTLD_NEXT, "sendmsg");
    if (msg && msg->msg_name && _is_blocked_sa((const struct sockaddr *)msg->msg_name)) { errno = EACCES; return -1; }
    return real_fn(sockfd, msg, flags);
}
"""


class UDockerManager:
    """Wrapper around `udocker --allow-root` storing layers in `udocker_dir`."""

    def __init__(self, udocker_dir: str = DEFAULT_UDOCKER_DIR, execmode: str = ""):
        self.udocker_dir = udocker_dir
        self.execmode = execmode
        Path(self.udocker_dir).mkdir(parents=True, exist_ok=True)
        self.base_env = os.environ.copy()
        self.base_env["UDOCKER_DIR"] = self.udocker_dir
        self.base_env["UDOCKER_TARBALL"] = self.base_env.get("UDOCKER_TARBALL", "")
        self.base_env["DEBIAN_FRONTEND"] = "noninteractive"
        self._offline_mode: str | None = None
        self._ld_preload_dir: str | None = None

    def _detect_offline_mode(self) -> str:
        """Detect best available outbound network isolation mode (`unshare_net`, `unshare_r_net`, or `ld_preload`)."""
        if self._offline_mode is not None:
            return self._offline_mode
        for mode, cmd in (
            ("unshare_net", ["unshare", "--net", "--", "bash", "-c", "ip link set lo up"]),
            ("unshare_r_net", ["unshare", "-r", "--net", "--", "bash", "-c", "ip link set lo up"]),
        ):
            try:
                cp = subprocess.run(cmd, capture_output=True, text=True, timeout=15.0)
                if cp.returncode == 0:
                    self._offline_mode = mode
                    return mode
            except Exception:
                pass
        net_dir = Path("/tmp/chronos_net_guard")
        net_dir.mkdir(parents=True, exist_ok=True)
        so_path = net_dir / "chronos_block_net.so"
        if not so_path.exists():
            c_path = net_dir / "chronos_block_net.c"
            c_path.write_text(_NET_BLOCK_C_SRC, encoding="utf-8")
            cp_cc = subprocess.run(
                ["gcc", "-shared", "-fPIC", "-O2", str(c_path), "-o", str(so_path), "-ldl"],
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            if cp_cc.returncode != 0:
                raise RuntimeError(f"Offline isolation unavailable: unshare failed and gcc failed: {cp_cc.stderr}")
            so_path.chmod(0o755)
        self._ld_preload_dir = str(net_dir)
        self._offline_mode = "ld_preload"
        return self._offline_mode

    def _run_udocker(
        self,
        args: list[str],
        timeout: float = 600.0,
        prefix_cmd: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [*(prefix_cmd or []), "udocker", "--quiet", "--allow-root", *args]
        return subprocess.run(
            cmd,
            env=self.base_env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )

    def ensure_installed(self) -> None:
        cp = self._run_udocker(["install"], timeout=180.0)
        if cp.returncode != 0:
            raise RuntimeError(f"udocker install failed ({cp.returncode}): {cp.stderr or cp.stdout}")

    def list_images(self) -> list[str]:
        cp = self._run_udocker(["images", "-l"], timeout=60.0)
        if cp.returncode != 0:
            return []
        imgs: list[str] = []
        for ln in cp.stdout.splitlines():
            s = ln.strip()
            if not s or s.startswith("REPOSITORY") or s.startswith("---"):
                continue
            parts = s.split()
            if len(parts) >= 2 and parts[0] != "REPOSITORY":
                # Output columns vary by udocker version: repo:tag or repo tag
                imgs.append(f"{parts[0]}:{parts[1]}" if ":" not in parts[0] and parts[1] != "." else parts[0])
        return imgs

    def ensure_image(self, image: str, retries: int = 2) -> None:
        cp_chk = self._run_udocker(["inspect", image], timeout=60.0)
        if cp_chk.returncode == 0:
            return
        last_err = ""
        for attempt in range(retries + 1):
            cp = self._run_udocker(["pull", image], timeout=900.0)
            if cp.returncode == 0:
                return
            last_err = (cp.stderr or cp.stdout or "")[-1000:]
            if attempt < retries:
                time.sleep(3.0 * (attempt + 1))
        raise RuntimeError(f"udocker pull {image} failed: {last_err}")

    def create_container(self, cname: str, image: str) -> None:
        self.rm_container(cname)
        cp = self._run_udocker(["create", f"--name={cname}", image], timeout=600.0)
        if cp.returncode != 0:
            raise RuntimeError(f"udocker create {cname} from {image} failed: {cp.stderr or cp.stdout}")
        if self.execmode:
            cp_s = self._run_udocker(["setup", f"--execmode={self.execmode}", cname], timeout=120.0)
            if cp_s.returncode != 0:
                raise RuntimeError(f"udocker setup --execmode={self.execmode} {cname} failed: {cp_s.stderr or cp_s.stdout}")

    def rm_container(self, cname: str) -> None:
        self._run_udocker(["rm", cname], timeout=120.0)

    def prune_images_if_low_disk(self, min_free_gib: float = 80.0, keep_images: set[str] | None = None) -> list[str]:
        free_gib = shutil.disk_usage("/content").free / (1024**3)
        if free_gib >= min_free_gib:
            return []
        keep = keep_images or set()
        removed: list[str] = []
        for img in self.list_images():
            if img in keep:
                continue
            cp = self._run_udocker(["rmi", img], timeout=120.0)
            if cp.returncode == 0:
                removed.append(img)
                if shutil.disk_usage("/content").free / (1024**3) >= min_free_gib + 15.0:
                    break
        return removed

    def run_shell_in_container(
        self,
        cname: str,
        shell_cmd: str,
        *,
        mounts: list[tuple[str, str]] | None = None,
        env_vars: dict[str, str] | None = None,
        workdir: str = "/app",
        timeout: float = 120.0,
        no_network: bool = True,
    ) -> dict[str, Any]:
        """Execute `shell_cmd` inside `cname` using a mounted script to avoid quoting errors."""
        with tempfile.TemporaryDirectory(prefix="dsv41_cmd_") as host_tmp:
            script_path = Path(host_tmp) / "cmd.sh"
            script_path.write_text(shell_cmd + "\n", encoding="utf-8")
            script_path.chmod(0o755)

            run_args = ["run", f"--workdir={workdir}", f"--volume={host_tmp}:/_chronos_cmd"]
            offline_mode = self._detect_offline_mode() if no_network else "none"
            if no_network:
                resolv_path = Path(host_tmp) / "resolv.conf"
                resolv_path.write_text("nameserver 127.0.0.1\n", encoding="utf-8")
                resolv_path.chmod(0o444)
                run_args.append(f"--volume={resolv_path}:/etc/resolv.conf")
                if offline_mode == "ld_preload" and self._ld_preload_dir:
                    run_args.append(f"--volume={self._ld_preload_dir}:/_chronos_net")
            for h_dir, c_dir in mounts or []:
                run_args.append(f"--volume={h_dir}:{c_dir}")
            merged_env = dict(CONTAINER_ENV)
            if no_network:
                merged_env.update(
                    {
                        "http_proxy": "http://127.0.0.1:9",
                        "https_proxy": "http://127.0.0.1:9",
                        "HTTP_PROXY": "http://127.0.0.1:9",
                        "HTTPS_PROXY": "http://127.0.0.1:9",
                        "all_proxy": "http://127.0.0.1:9",
                        "ALL_PROXY": "http://127.0.0.1:9",
                        "no_proxy": "localhost,127.0.0.1",
                        "NO_PROXY": "localhost,127.0.0.1",
                    }
                )
                if offline_mode == "ld_preload":
                    merged_env["LD_PRELOAD"] = "/_chronos_net/chronos_block_net.so"
            if env_vars:
                merged_env.update(env_vars)
            for ek, ev in merged_env.items():
                run_args.append(f"--env={ek}={ev}")
            run_args.extend([cname, "bash", "/_chronos_cmd/cmd.sh"])
            if not no_network:
                net_prefix = None
            elif offline_mode == "unshare_net":
                net_prefix = ["unshare", "--net", "--", "bash", "-c", 'ip link set lo up 2>/dev/null || true; exec "$@"', "_"]
            elif offline_mode == "unshare_r_net":
                net_prefix = ["unshare", "-r", "--net", "--", "bash", "-c", 'ip link set lo up 2>/dev/null || true; exec "$@"', "_"]
            else:
                net_prefix = None

            t0 = time.perf_counter()
            try:
                cp = self._run_udocker(run_args, timeout=timeout, prefix_cmd=net_prefix)
                raw_out = _UDOCKER_BANNER_RE.sub("", (cp.stdout or "") + (cp.stderr or "")).lstrip("\r\n")
                return {
                    "returncode": int(cp.returncode),
                    "output": raw_out,
                    "timed_out": False,
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                }
            except subprocess.TimeoutExpired as exc:
                out_s = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
                err_s = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
                raw_out = _UDOCKER_BANNER_RE.sub("", f"{out_s}{err_s}").lstrip("\r\n")
                return {
                    "returncode": -1,
                    "output": f"{raw_out}\n[TIMEOUT after {timeout}s]",
                    "timed_out": True,
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                }


def format_observation(
    res: dict[str, Any],
    *,
    step_i: int | None = None,
    step_limit: int | None = None,
    prompt_tokens: int | None = None,
    max_context: int = 124000,
) -> str:
    """Format tool output cleanly (6,500-char head+tail truncation) with two-stage step/context budget."""
    rc = res["returncode"]
    raw_out = res.get("output") or ""
    if len(raw_out) <= 6500:
        body = f"Exit code: {rc}\nOutput:\n{raw_out}"
    else:
        elided = len(raw_out) - 6500
        body = (
            f"Exit code: {rc}\n"
            f"Output:\n{raw_out[:3250]}\n...[elided {elided} chars]...\n{raw_out[-3250:]}"
        )
    if step_i is not None and step_limit is not None:
        ptok_val = prompt_tokens if prompt_tokens is not None else 0
        alert = ""
        if (step_limit - step_i) <= 10 or ptok_val >= int(0.85 * max_context):
            alert = " [FINAL BUDGET ALERT: <15% context left — finish edits, run tests, git add -A && git commit, and run echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT now or uncommitted work will be lost.]"
        elif (step_limit - step_i) <= 30 or ptok_val >= int(0.55 * max_context):
            alert = " [Budget alert: prioritize implementing your edits, running tests, git add/commit, and submitting now.]"
        body += f"\n<budget>Step {step_i}/{step_limit} | Context ~{ptok_val}/{max_context} tokens{alert}</budget>"
    return body


def verify_patch_harbor_v11(
    mgr: UDockerManager,
    task: dict[str, Any],
    model_patch_bytes: bytes,
) -> dict[str, Any]:
    """Run Harbor v1.1 `environment_mode = 'separate'` verification in a fresh container."""
    tid = task["task_id"]
    safe_tid = re.sub(r"[^A-Za-z0-9_.-]+", "_", tid)[:40]
    cname = f"ver_{safe_tid}_{int(time.time())}"
    mgr.ensure_image(task["docker_image"])

    with tempfile.TemporaryDirectory(prefix=f"harbor_ver_{safe_tid}_") as work_dir:
        wd = Path(work_dir)
        tests_copy = wd / "tests"
        artifacts_dir = wd / "artifacts"
        verifier_dir = wd / "verifier"
        shutil.copytree(Path(task["task_dir"]) / "tests", tests_copy)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        verifier_dir.mkdir(parents=True, exist_ok=True)
        if model_patch_bytes:
            (artifacts_dir / "model.patch").write_bytes(model_patch_bytes)

        mgr.create_container(cname, task["docker_image"])
        try:
            ver_res = mgr.run_shell_in_container(
                cname,
                "bash /tests/test.sh",
                mounts=[
                    (str(tests_copy), "/tests"),
                    (str(artifacts_dir), "/logs/artifacts"),
                    (str(verifier_dir), "/logs/verifier"),
                ],
                workdir="/app",
                timeout=max(300.0, float(task.get("verifier_timeout_sec") or 1800.0)),
            )
        finally:
            mgr.rm_container(cname)

        reward_json_p = verifier_dir / "reward.json"
        reward_txt_p = verifier_dir / "reward.txt"
        reward_data: dict[str, Any] = {}
        if reward_json_p.exists():
            reward_data = json.loads(reward_json_p.read_text(encoding="utf-8"))
        elif reward_txt_p.exists():
            try:
                reward_data = {"reward": float(reward_txt_p.read_text(encoding="utf-8").strip())}
            except Exception:
                reward_data = {"reward": -1}
        else:
            reward_data = {"reward": -1}

        return {
            "reward": reward_data.get("reward", -1),
            "f2p": reward_data.get("f2p"),
            "p2p": reward_data.get("p2p"),
            "f2p_passed": reward_data.get("f2p_passed"),
            "f2p_total": reward_data.get("f2p_total", task["f2p_total"]),
            "p2p_passed": reward_data.get("p2p_passed"),
            "p2p_total": reward_data.get("p2p_total", task["p2p_total"]),
            "partial": reward_data.get("partial"),
            "apply_failed": reward_data.get("apply_failed", False),
            "verifier_returncode": ver_res["returncode"],
            "verifier_elapsed_s": ver_res["elapsed_s"],
            "verifier_stdout_tail": (ver_res["output"] or "")[-2500:],
        }


def _collect_patch_from_agent(
    mgr: UDockerManager,
    cname: str,
    task: dict[str, Any],
    artifacts_dir: Path,
) -> bytes:
    cmds = list(task.get("collect_commands") or [])
    if not cmds:
        base = task.get("base_commit_hash") or "HEAD"
        cmds = [
            f"cd /app && mkdir -p /logs/artifacts && git config --global --add safe.directory /app && "
            f"git diff --binary {base} HEAD > /logs/artifacts/model.patch"
        ]
    for cmd in cmds:
        mgr.run_shell_in_container(
            cname,
            cmd,
            mounts=[(str(artifacts_dir), "/logs/artifacts")],
            workdir="/app",
            timeout=300.0,
        )
    mp = artifacts_dir / "model.patch"
    return mp.read_bytes() if mp.exists() else b""


def _parse_dsml_response(
    raw_text: str,
    answer_text: str,
    reasoning_text: str,
    enc_mod: Any,
    step_i: int,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Parse native DSML tool calls and reasoning cleanly with regex fallbacks."""
    clean_raw = raw_text
    if clean_raw.endswith("<｜end▁of▁sentence｜>"):
        clean_raw = clean_raw[: -len("<｜end▁of▁sentence｜>")]

    parsed_reasoning = ""
    parsed_content = ""
    raw_tcs: list[dict[str, Any]] = []
    try:
        parsed = enc_mod.parse_message_from_completion_text(
            clean_raw + "<｜end▁of▁sentence｜>",
            thinking_mode="thinking",
        )
        parsed_reasoning = (parsed.get("reasoning_content") or "").strip()
        parsed_content = (parsed.get("content") or "").strip()
        raw_tcs = list(parsed.get("tool_calls") or [])
    except Exception:
        parsed_reasoning = ""
        parsed_content = (answer_text or "").strip()
        raw_tcs = []

    reasoning_out = parsed_reasoning or (reasoning_text or "").strip()
    used_fallback = False
    if not raw_tcs:
        for cmd_m in _DSML_CMD_RE.findall(raw_text):
            raw_tcs.append(
                {
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": cmd_m})},
                }
            )
            used_fallback = True
    if not raw_tcs:
        for cmd_m in _DSML_CMD_MISSING_GT_RE.findall(raw_text):
            cmd_clean = cmd_m[:-2] if cmd_m.endswith('">') else cmd_m
            raw_tcs.append(
                {
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": cmd_clean})},
                }
            )
            used_fallback = True
    if not raw_tcs and answer_text:
        for blk in _BASH_BLOCK_RE.findall(answer_text):
            raw_tcs.append(
                {
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": blk.strip()})},
                }
            )
            used_fallback = True
    if used_fallback and parsed_content:
        parsed_content = _DSML_BLOCK_STRIP_RE.sub("", parsed_content).strip()

    tool_calls: list[dict[str, Any]] = []
    for j, tc in enumerate(raw_tcs):
        fn = dict(tc.get("function") or {})
        tool_calls.append(
            {
                "id": f"call_{step_i}_{j}",
                "type": tc.get("type", "function"),
                "function": {
                    "name": str(fn.get("name") or "bash"),
                    "arguments": str(fn.get("arguments") or "{}"),
                },
            }
        )
    return reasoning_out, parsed_content, tool_calls


def _extract_command_from_args(args_str: str) -> str:
    try:
        parsed_args = json.loads(args_str)
        if isinstance(parsed_args, dict):
            direct = parsed_args.get("command") or parsed_args.get("cmd")
            if isinstance(direct, str) and direct.strip():
                return direct
            for k, v in parsed_args.items():
                if isinstance(v, str) and v.strip() and ("command" in str(k).lower() or len(parsed_args) == 1):
                    return v
    except Exception:
        pass
    m = re.search(r'"command"\s*:\s*"(.*)"\s*\}?$', args_str, re.DOTALL)
    if m:
        return m.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
    return args_str.strip()


def run_single_deepswe_task(
    mgr: UDockerManager,
    task: dict[str, Any],
    model: Any,
    tokenizer: Any,
    dsv41_model_mod: Any,
    runner_mod: Any,
    spec_eng: Any,
    *,
    step_limit: int = 120,
    cmd_timeout: float = 180.0,
    reasoning_effort: str | int = "max",
    think_ceiling: int = 2048,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_context: int = 60000,
    use_speculative: bool = False,
) -> dict[str, Any]:
    """Run one DeepSWE v1.1 Harbor task end-to-end with native DSML tool calling + interleaved thinking."""
    _ensure_paths()
    import encoding as enc_mod

    tid = task["task_id"]
    safe_tid = re.sub(r"[^A-Za-z0-9_.-]+", "_", tid)[:40]
    cname = f"agent_{safe_tid}_{int(time.time())}"
    mgr.ensure_image(task["docker_image"])
    offline_mode = mgr._detect_offline_mode()

    t_task0 = time.perf_counter()
    trajectory: list[dict[str, Any]] = []
    submitted = False
    stop_reason = "step_limit"
    total_gen_tokens = 0
    total_think_tokens = 0
    total_prefilled_tokens = 0
    total_reused_tokens = 0
    think_forced_count = 0
    total_drafted = 0
    total_accepted = 0
    tot_decode_s = 0.0
    tot_decode_steps = 0
    pf_hits = 0
    pf_t1 = 0
    pf_t2 = 0
    pf_t3 = 0
    dec_hits = 0
    dec_t1 = 0
    dec_t2 = 0
    dec_t3 = 0
    dec_nz_rows = 0

    with tempfile.TemporaryDirectory(prefix=f"harbor_agent_{safe_tid}_") as work_dir:
        artifacts_dir = Path(work_dir) / "agent_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        mgr.create_container(cname, task["docker_image"])
        sess = None
        no_tool_streak = 0

        try:
            sess = spec_eng.PrefixKVCacheSession(model, dsv41_model_mod, runner_mod)
            sess.reset()
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT_DSML, "tools": BASH_TOOLS},
                {"role": "user", "content": INSTANCE_TEMPLATE_DSML.format(task=task["instruction"])},
            ]

            for step_i in range(1, step_limit + 1):
                res = spec_eng.generate_with_speculative_and_prefix(
                    model,
                    tokenizer,
                    dsv41_model_mod,
                    runner_mod,
                    messages,
                    max_new_tokens=4096,
                    think_ceiling=think_ceiling,
                    ablate_guards=True,
                    thinking_mode="thinking",
                    reasoning_effort=reasoning_effort,
                    tools=None,
                    use_speculative=use_speculative,
                    prefix_session=sess,
                    stop_on_bash_block=False,
                    temperature=temperature,
                    top_p=top_p,
                    log_routing=True,
                )
                if not res.get("logits_finite", True):
                    raise RuntimeError(f"Non-finite logits in {tid} step {step_i}!")
                step_gtok = int(res.get("generated_tokens", 0))
                if step_gtok > 1 and int(res.get("decode_nonzero_rows", 0)) <= 0:
                    raise RuntimeError(f"decode_nonzero_rows == 0 in {tid} step {step_i}!")

                total_gen_tokens += step_gtok
                total_think_tokens += int(res.get("think_tokens", 0))
                total_prefilled_tokens += int(res.get("newly_prefilled_tokens", 0))
                total_reused_tokens += int(res.get("cached_prefix_tokens", 0))
                total_drafted += int(res.get("drafted_tokens", 0))
                total_accepted += int(res.get("accepted_tokens", 0))
                tot_decode_s += float(res.get("decode_s", 0.0))
                tot_decode_steps += max(step_gtok - 1, 1) if step_gtok > 0 else 0
                pf_hits += int(res.get("prefill_route_hits", 0))
                pf_t1 += int(res.get("prefill_t1_hits", 0))
                pf_t2 += int(res.get("prefill_t2_hits", 0))
                pf_t3 += int(res.get("prefill_t3_hits", 0))
                dec_hits += int(res.get("decode_route_hits", 0))
                dec_t1 += int(res.get("decode_t1_hits", 0))
                dec_t2 += int(res.get("decode_t2_hits", 0))
                dec_t3 += int(res.get("decode_t3_hits", 0))
                dec_nz_rows += int(res.get("decode_nonzero_rows", 0))
                if res.get("think_forced"):
                    think_forced_count += 1

                raw_text = tokenizer.decode(res.get("gen_ids") or [], skip_special_tokens=False)
                if not raw_text:
                    raw_text = res.get("text") or ""
                reasoning_content, content, tool_calls = _parse_dsml_response(
                    raw_text,
                    res.get("answer_text") or "",
                    res.get("reasoning_text") or "",
                    enc_mod,
                    step_i,
                )

                stored_content = content
                if not tool_calls and len(stored_content) > 600:
                    stored_content = stored_content[:600] + "\n...[truncated malformed response]"

                asst_entry: dict[str, Any] = {
                    "role": "assistant",
                    "reasoning_content": reasoning_content,
                    "content": stored_content,
                }
                if tool_calls:
                    asst_entry["tool_calls"] = tool_calls
                messages.append(asst_entry)

                cmd_list: list[str] = []
                last_rc = -2
                last_elapsed = 0.0
                last_obs = ""
                prompt_plus_gen = int(res.get("prompt_tokens", 0)) + int(res.get("generated_tokens", 0))

                if tool_calls:
                    no_tool_streak = 0
                    for tc in tool_calls:
                        tool_name = str(tc.get("function", {}).get("name") or "bash")
                        raw_args = tc.get("function", {}).get("arguments", "")
                        cmd_str = _extract_command_from_args(raw_args).strip()
                        cmd_list.append(cmd_str)
                        if tool_name != "bash" and not cmd_str:
                            exec_res = {
                                "returncode": -2,
                                "output": f"Unknown tool '{tool_name}'. Only the 'bash' tool with parameter 'command' is available.",
                                "timed_out": False,
                                "elapsed_s": 0.0,
                            }
                        elif cmd_str:
                            exec_res = mgr.run_shell_in_container(
                                cname,
                                cmd_str,
                                mounts=[(str(artifacts_dir), "/logs/artifacts")],
                                workdir="/app",
                                timeout=cmd_timeout,
                            )
                        else:
                            exec_res = {
                                "returncode": -2,
                                "output": f"Invalid bash tool call: missing 'command' parameter (raw args: {str(raw_args)[:200]}).",
                                "timed_out": False,
                                "elapsed_s": 0.0,
                            }
                        obs = format_observation(
                            exec_res,
                            step_i=step_i,
                            step_limit=step_limit,
                            prompt_tokens=prompt_plus_gen,
                            max_context=max_context,
                        )
                        last_rc = int(exec_res["returncode"])
                        last_elapsed = float(exec_res["elapsed_s"])
                        last_obs = obs
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "name": tc["function"].get("name", "bash"),
                                "content": obs,
                            }
                        )
                        out_lines = [ln.strip() for ln in (exec_res.get("output") or "").splitlines()]
                        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in out_lines:
                            submitted = True
                            stop_reason = "submitted"
                            break
                else:
                    no_tool_streak += 1
                    cur_patch = _collect_patch_from_agent(mgr, cname, task, artifacts_dir)
                    if len(cur_patch) > 0:
                        submitted = True
                        stop_reason = "final_response_with_patch"
                        last_rc = 0
                        last_obs = f"Detected non-empty git diff ({len(cur_patch)} bytes) on final response."
                    else:
                        nudge = (
                            "No bash tool call or committed git diff detected. Use the bash tool to inspect/edit files, "
                            "run tests, git add/commit your changes, and run `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."
                        )
                        last_obs = nudge
                        messages.append({"role": "user", "content": nudge})

                cmd_joined = " ; ".join(cmd_list)
                trajectory.append(
                    {
                        "step": step_i,
                        "prompt_tokens": int(res.get("prompt_tokens", 0)),
                        "cached_prefix_tokens": int(res.get("cached_prefix_tokens", 0)),
                        "newly_prefilled_tokens": int(res.get("newly_prefilled_tokens", 0)),
                        "generated_tokens": int(res.get("generated_tokens", 0)),
                        "think_tokens": int(res.get("think_tokens", 0)),
                        "think_forced": bool(res.get("think_forced", False)),
                        "decode_tok_s": float(res.get("decode_tok_s", 0.0)),
                        "drafted_tokens": int(res.get("drafted_tokens", 0)),
                        "accepted_tokens": int(res.get("accepted_tokens", 0)),
                        "decode_nonzero_rows": int(res.get("decode_nonzero_rows", 0)),
                        "prefill_4bit_frac": float(res.get("prefill_4bit_frac", 0.0)),
                        "decode_4bit_frac": float(res.get("decode_4bit_frac", 0.0)),
                        "n_tool_calls": len(tool_calls),
                        "reasoning_preview": reasoning_content[:600],
                        "assistant_preview": (content or str(tool_calls))[:800],
                        "cmd": cmd_joined[:1200],
                        "returncode": last_rc,
                        "cmd_elapsed_s": last_elapsed,
                        "obs_preview": last_obs[:600],
                    }
                )
                print(
                    f"  [{tid} step {step_i:02d}/{step_limit}] "
                    f"ptok={res.get('prompt_tokens')}(reuse={res.get('cached_prefix_tokens')}) "
                    f"gtok={res.get('generated_tokens')}(th={res.get('think_tokens')}) "
                    f"@ {res.get('decode_tok_s')} tok/s | 4b(pf={100*float(res.get('prefill_4bit_frac', 0.0)):.1f}%,dec={100*float(res.get('decode_4bit_frac', 0.0)):.1f}%) "
                    f"| tc={len(tool_calls)} rc={last_rc} | "
                    f"{cmd_joined.splitlines()[0][:70] if cmd_joined else 'NO_TOOL'}",
                    flush=True,
                )

                if submitted:
                    break
                if (not tool_calls) and no_tool_streak >= 3:
                    stop_reason = "no_tool_calls"
                    break
                if prompt_plus_gen >= max_context:
                    stop_reason = "context_limit"
                    break

            patch_bytes = _collect_patch_from_agent(mgr, cname, task, artifacts_dir)
        finally:
            if sess is not None:
                try:
                    sess.reset()
                except Exception:
                    pass
                del sess
            gc.collect()
            torch.cuda.empty_cache()
            mgr.rm_container(cname)

    ver_out = verify_patch_harbor_v11(mgr, task, patch_bytes)
    elapsed_s = round(time.perf_counter() - t_task0, 2)
    reward = ver_out["reward"]
    passed = float(reward) == 1.0

    return {
        "model_version": getattr(builtins, "GLOBAL_DSV41_VERSION", "chronos_v8_calibrated_3tier"),
        "task_id": tid,
        "language": task["language"],
        "docker_image": task["docker_image"],
        "offline_isolation_mode": offline_mode,
        "passed": passed,
        "reward": reward,
        "f2p": ver_out["f2p"],
        "p2p": ver_out["p2p"],
        "f2p_passed": ver_out["f2p_passed"],
        "f2p_total": ver_out["f2p_total"],
        "p2p_passed": ver_out["p2p_passed"],
        "p2p_total": ver_out["p2p_total"],
        "partial": ver_out["partial"],
        "apply_failed": ver_out["apply_failed"],
        "submitted": submitted,
        "stop_reason": stop_reason,
        "step_limit": step_limit,
        "temperature": temperature,
        "top_p": top_p,
        "use_speculative": use_speculative,
        "steps_taken": len(trajectory),
        "patch_bytes": len(patch_bytes),
        "patch_preview": patch_bytes.decode("utf-8", "replace")[:2000],
        "total_generated_tokens": total_gen_tokens,
        "total_think_tokens": total_think_tokens,
        "total_prefilled_tokens": total_prefilled_tokens,
        "total_reused_tokens": total_reused_tokens,
        "think_forced_turns": think_forced_count,
        "drafted_tokens": total_drafted,
        "accepted_tokens": total_accepted,
        "spec_accept_rate": round(total_accepted / max(1, total_drafted), 4) if total_drafted > 0 else 0.0,
        "mean_decode_tok_s": round(tot_decode_steps / max(tot_decode_s, 1e-6), 2),
        "decode_steps": tot_decode_steps,
        "decode_s": round(tot_decode_s, 3),
        "prefill_route_hits": pf_hits,
        "prefill_t1_hits": pf_t1,
        "prefill_t2_hits": pf_t2,
        "prefill_t3_hits": pf_t3,
        "decode_route_hits": dec_hits,
        "decode_t1_hits": dec_t1,
        "decode_t2_hits": dec_t2,
        "decode_t3_hits": dec_t3,
        "decode_nonzero_rows": dec_nz_rows,
        "prefill_4bit_pct": round(100.0 * (pf_t1 + pf_t2) / max(1, pf_hits), 2),
        "decode_4bit_pct": round(100.0 * (dec_t1 + dec_t2) / max(1, dec_hits), 2),
        "elapsed_s": elapsed_s,
        "verifier": ver_out,
        "trajectory": trajectory,
    }


def _agg_routing_deepswe(recs: list[dict[str, Any]]) -> dict[str, float]:
    pf_tot = sum(int(r.get("prefill_route_hits", 0)) for r in recs)
    dec_tot = sum(int(r.get("decode_route_hits", 0)) for r in recs)
    pf_t1 = sum(int(r.get("prefill_t1_hits", 0)) for r in recs)
    pf_t2 = sum(int(r.get("prefill_t2_hits", 0)) for r in recs)
    pf_t3 = sum(int(r.get("prefill_t3_hits", 0)) for r in recs)
    dec_t1 = sum(int(r.get("decode_t1_hits", 0)) for r in recs)
    dec_t2 = sum(int(r.get("decode_t2_hits", 0)) for r in recs)
    dec_t3 = sum(int(r.get("decode_t3_hits", 0)) for r in recs)
    all_tot = pf_tot + dec_tot
    out: dict[str, float] = {}
    if pf_tot > 0:
        out["prefill_t1_pct"] = round(100.0 * pf_t1 / pf_tot, 2)
        out["prefill_t2_pct"] = round(100.0 * pf_t2 / pf_tot, 2)
        out["prefill_t3_pct"] = round(100.0 * pf_t3 / pf_tot, 2)
        out["prefill_4bit_pct"] = round(100.0 * (pf_t1 + pf_t2) / pf_tot, 2)
    if dec_tot > 0:
        out["decode_t1_pct"] = round(100.0 * dec_t1 / dec_tot, 2)
        out["decode_t2_pct"] = round(100.0 * dec_t2 / dec_tot, 2)
        out["decode_t3_pct"] = round(100.0 * dec_t3 / dec_tot, 2)
        out["decode_4bit_pct"] = round(100.0 * (dec_t1 + dec_t2) / dec_tot, 2)
    if all_tot > 0:
        out["all_t1_pct"] = round(100.0 * (pf_t1 + dec_t1) / all_tot, 2)
        out["all_t2_pct"] = round(100.0 * (pf_t2 + dec_t2) / all_tot, 2)
        out["all_t3_pct"] = round(100.0 * (pf_t3 + dec_t3) / all_tot, 2)
        out["all_4bit_pct"] = round(100.0 * (pf_t1 + pf_t2 + dec_t1 + dec_t2) / all_tot, 2)
    return out


def summarize_deepswe_report(
    progress_jsonl: str,
    out_json: str,
    schedule: list[dict[str, Any]],
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if os.path.isfile(progress_jsonl):
        seen: set[str] = set()
        with open(progress_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rec = json.loads(s)
                if rec["task_id"] not in seen:
                    seen.add(rec["task_id"])
                    records.append(rec)

    n = len(records)
    n_pass = sum(1 for r in records if r.get("passed"))
    lo, hi = wilson_ci_95(n_pass, n)
    by_lang: dict[str, dict[str, Any]] = {}
    for r in records:
        lang = r.get("language", "unknown")
        b = by_lang.setdefault(lang, {"n": 0, "passed": 0})
        b["n"] += 1
        if r.get("passed"):
            b["passed"] += 1
    for b in by_lang.values():
        b["pass_rate_pct"] = round(100.0 * b["passed"] / max(1, b["n"]), 2)
        b_lo, b_hi = wilson_ci_95(b["passed"], b["n"])
        b["wilson_ci_95_pct"] = [b_lo, b_hi]

    m25_ids = {s["task_id"] for s in schedule if s.get("milestone25")}
    m25_recs = [r for r in records if r["task_id"] in m25_ids]
    m25_pass = sum(1 for r in m25_recs if r.get("passed"))
    m25_lo, m25_hi = wilson_ci_95(m25_pass, len(m25_recs))
    tot_dec_steps = sum(int(r.get("decode_steps", 0)) for r in records)
    tot_dec_s = sum(float(r.get("decode_s", 0.0)) for r in records)
    spec_drafted = sum(int(r.get("drafted_tokens", 0)) for r in records)
    spec_accepted = sum(int(r.get("accepted_tokens", 0)) for r in records)
    iso_modes = sorted({str(r.get("offline_isolation_mode", "unknown")) for r in records})

    report = {
        "benchmark": "datacurve-ai/deep-swe v1.1 (Harbor)",
        "model_version": getattr(builtins, "GLOBAL_DSV41_VERSION", "chronos_v8_calibrated_3tier"),
        "source_repo": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings or {},
        "offline_isolation_modes": iso_modes,
        "total_scheduled": len(schedule),
        "completed": n,
        "passed": n_pass,
        "pass_rate_pct": round(100.0 * n_pass / max(1, n), 2),
        "wilson_ci_95_pct": [lo, hi],
        "milestone25_completed": len(m25_recs),
        "milestone25_passed": m25_pass,
        "milestone25_pass_rate_pct": round(100.0 * m25_pass / max(1, len(m25_recs)), 2),
        "milestone25_wilson_ci_95_pct": [m25_lo, m25_hi],
        "mean_steps": round(sum(r.get("steps_taken", 0) for r in records) / max(1, n), 2),
        "mean_elapsed_s": round(sum(r.get("elapsed_s", 0.0) for r in records) / max(1, n), 2),
        "mean_decode_tok_s": round(tot_dec_steps / max(tot_dec_s, 1e-6), 2) if tot_dec_steps > 0 else 0.0,
        "decode_nonzero_rows": sum(int(r.get("decode_nonzero_rows", 0)) for r in records),
        "spec_drafted": spec_drafted,
        "spec_accepted": spec_accepted,
        "spec_accept_rate": round(spec_accepted / max(1, spec_drafted), 4) if spec_drafted > 0 else 0.0,
        "routing": _agg_routing_deepswe(records),
        "submitted_count": sum(1 for r in records if r.get("submitted")),
        "by_language": by_lang,
    }
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _attach_resident_model(target_max_seq_len: int = 65536) -> tuple[Any, Any, Any, Any, Any]:
    _ensure_paths()
    import shutil as _shutil
    import types as _types

    for _fn in ("dsv41_chronos_moe.py", "dsv41_longctx_spec_engine.py"):
        _src = f"/content/chronos_1t/models/{_fn}"
        _dst = f"/content/{_fn}"
        if os.path.isfile(_src) and (not os.path.isfile(_dst) or os.path.getmtime(_src) > os.path.getmtime(_dst)):
            _shutil.copy2(_src, _dst)

    import chronos_1t.models.dsv41_chronos_moe as moe_mod
    import chronos_1t.models.dsv41_longctx_spec_engine as spec_eng
    importlib.reload(spec_eng)
    import scripts.run_40layer_chimera_dsv41_a100 as runner_mod

    model = getattr(builtins, "GLOBAL_DSV41_MODEL", None)
    tokenizer = getattr(builtins, "GLOBAL_DSV41_TOKENIZER", None)
    if model is None or tokenizer is None:
        raise RuntimeError("GLOBAL_DSV41_MODEL not loaded in current session! Do not reload blindly; inspect session state.")

    ver_tag = getattr(builtins, "GLOBAL_DSV41_VERSION", "")
    if not getattr(model, "_chronos_unleashed_ready", False):
        if ver_tag != "v8_calibrated_3tier":
            target_fwd = moe_mod.ChronosDsv41SparseMoeBlock.forward
            hot_rebound = False
            for lyr in model.layers:
                ffn = getattr(lyr, "ffn", None)
                if ffn is not None and hasattr(ffn, "w1_gu_u8"):
                    if getattr(ffn.forward, "__func__", ffn.forward) is not target_fwd:
                        hot_rebound = True
                    ffn.forward = _types.MethodType(target_fwd, ffn)
            if hot_rebound:
                runner_mod.warmup_40layer_cudagraphs(model)
        up_info = spec_eng.upgrade_resident_model_long_context(
            model, runner_mod.dsv41_model_mod, runner_mod, target_max_seq_len=target_max_seq_len
        )
        model._chronos_v8_attach_info = up_info
        model._chronos_unleashed_ready = True
    elif int(model.layers[0].attn.freqs_cis.size(0)) < target_max_seq_len:
        up_info = spec_eng.upgrade_resident_model_long_context(
            model, runner_mod.dsv41_model_mod, runner_mod, target_max_seq_len=target_max_seq_len
        )
        model._chronos_v8_attach_info = up_info
    else:
        up_info = spec_eng.upgrade_resident_model_long_context(
            model, runner_mod.dsv41_model_mod, runner_mod, target_max_seq_len=target_max_seq_len
        )
        model._chronos_v8_attach_info = up_info

    spec_eng.install_general_chunked_attention(runner_mod.dsv41_model_mod)
    return model, tokenizer, runner_mod.dsv41_model_mod, runner_mod, spec_eng


def run_golden_gate(
    repo_dir: str = DEFAULT_REPO_DIR,
    task_id: str = "abs-module-cache-flags",
    out_json: str = "/content/dsv41_deepswe_v11_golden_gate.json",
    udocker_dir: str = DEFAULT_UDOCKER_DIR,
    execmode: str = "",
) -> dict[str, Any]:
    task = load_harbor_task(Path(repo_dir) / "tasks" / task_id)
    oracle_patch = (Path(repo_dir) / "tasks" / task_id / "solution" / "solution.patch").read_bytes()
    mgr = UDockerManager(udocker_dir=udocker_dir, execmode=execmode)
    mgr.ensure_installed()

    print(f"[Golden Gate] Verifying nop (empty patch) on {task_id} ({task['docker_image']})...", flush=True)
    nop_res = verify_patch_harbor_v11(mgr, task, b"")
    print(f"[Golden Gate] nop -> reward={nop_res['reward']}, f2p={nop_res['f2p_passed']}/{nop_res['f2p_total']}, p2p={nop_res['p2p_passed']}/{nop_res['p2p_total']}", flush=True)

    print(f"[Golden Gate] Verifying oracle (solution.patch) on {task_id}...", flush=True)
    ora_res = verify_patch_harbor_v11(mgr, task, oracle_patch)
    print(f"[Golden Gate] oracle -> reward={ora_res['reward']}, f2p={ora_res['f2p_passed']}/{ora_res['f2p_total']}, p2p={ora_res['p2p_passed']}/{ora_res['p2p_total']}", flush=True)

    passed_gate = float(nop_res["reward"]) == 0.0 and float(ora_res["reward"]) == 1.0
    out = {
        "task_id": task_id,
        "docker_image": task["docker_image"],
        "execmode": execmode or "default",
        "gate_passed": passed_gate,
        "nop": nop_res,
        "oracle": ora_res,
    }
    Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    if not passed_gate:
        raise RuntimeError(f"Golden Harbor Verifier Gate FAILED: nop={nop_res['reward']}, oracle={ora_res['reward']}")
    return out


def run_deepswe_slice(
    *,
    repo_dir: str = DEFAULT_REPO_DIR,
    task_order_json: str = DEFAULT_TASK_ORDER_JSON,
    progress_jsonl: str = DEFAULT_PROGRESS_JSONL,
    out_json: str = DEFAULT_REPORT_JSON,
    udocker_dir: str = DEFAULT_UDOCKER_DIR,
    execmode: str = "",
    start_idx: int = 0,
    limit: int = 2,
    step_limit: int = 120,
    cmd_timeout: float = 180.0,
    reasoning_effort: str | int = "max",
    think_ceiling: int = 2048,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_context: int = 60000,
    target_max_seq_len: int = 65536,
    use_speculative: bool = False,
    min_free_gib: float = 80.0,
    max_wall_s: float = 1100.0,
) -> dict[str, Any]:
    schedule = build_or_load_task_order(repo_dir=repo_dir, out_json=task_order_json)
    done_ids: set[str] = set()
    if os.path.isfile(progress_jsonl):
        with open(progress_jsonl, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if s:
                    done_ids.add(json.loads(s)["task_id"])

    mgr = UDockerManager(udocker_dir=udocker_dir, execmode=execmode)
    mgr.ensure_installed()
    offline_mode = mgr._detect_offline_mode()
    model, tokenizer, dsv41_model_mod, runner_mod, spec_eng = _attach_resident_model(
        target_max_seq_len=target_max_seq_len
    )
    settings_dict = {
        "max_new_tokens_per_turn": 4096,
        "think_ceiling": int(think_ceiling),
        "reasoning_effort": reasoning_effort,
        "ablate_guards": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "step_limit": int(step_limit),
        "max_context": int(max_context),
        "max_seq_len": int(model.layers[0].attn.freqs_cis.size(0)),
        "use_speculative": bool(use_speculative),
        "offline_isolation_mode": offline_mode,
    }
    t_slice0 = time.perf_counter()

    end_idx = min(len(schedule), start_idx + limit) if limit > 0 else len(schedule)
    for item in schedule[start_idx:end_idx]:
        tid = item["task_id"]
        if tid in done_ids:
            continue
        if max_wall_s > 0 and (time.perf_counter() - t_slice0) >= max_wall_s:
            print(
                f"\n[DeepSWE] Reached slice wall budget ({max_wall_s:.0f}s) before {tid}; checkpointing cleanly.",
                flush=True,
            )
            break
        task = load_harbor_task(Path(repo_dir) / "tasks" / tid)
        print(f"\n=== [DeepSWE v1.1 #{item['order_idx'] + 1:03d}/113] {tid} ({task['language']}) ===", flush=True)
        rec = run_single_deepswe_task(
            mgr,
            task,
            model,
            tokenizer,
            dsv41_model_mod,
            runner_mod,
            spec_eng,
            step_limit=step_limit,
            cmd_timeout=cmd_timeout,
            reasoning_effort=reasoning_effort,
            think_ceiling=think_ceiling,
            temperature=temperature,
            top_p=top_p,
            max_context=max_context,
            use_speculative=use_speculative,
        )
        rec["order_idx"] = item["order_idx"]
        rec["milestone25"] = item["milestone25"]
        with open(progress_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done_ids.add(tid)

        rep = summarize_deepswe_report(progress_jsonl, out_json, schedule, settings=settings_dict)
        print(
            f"[DeepSWE #{item['order_idx'] + 1:03d}] {tid}: passed={rec['passed']} reward={rec['reward']} "
            f"(f2p={rec['f2p_passed']}/{rec['f2p_total']}, p2p={rec['p2p_passed']}/{rec['p2p_total']}) "
            f"| steps={rec['steps_taken']} {rec['elapsed_s']}s | {rec['mean_decode_tok_s']} tok/s "
            f"4b(pf={rec['prefill_4bit_pct']}%,dec={rec['decode_4bit_pct']}%) | Overall={rep['passed']}/{rep['completed']} ({rep['pass_rate_pct']}%)",
            flush=True,
        )
        pruned = mgr.prune_images_if_low_disk(min_free_gib=min_free_gib, keep_images={task["docker_image"]})
        if pruned:
            print(f"[Disk Prune] Removed {len(pruned)} cached images to keep >= {min_free_gib} GiB free", flush=True)

    return summarize_deepswe_report(progress_jsonl, out_json, schedule, settings=settings_dict)


def main() -> int:
    ap = argparse.ArgumentParser(description="Official 1:1 DeepSWE v1.1 Harbor Runner")
    ap.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    ap.add_argument("--udocker-dir", default=DEFAULT_UDOCKER_DIR)
    ap.add_argument("--execmode", default="")
    ap.add_argument("--golden-gate", action="store_true")
    ap.add_argument("--task-order-json", default=DEFAULT_TASK_ORDER_JSON)
    ap.add_argument("--progress-jsonl", default=DEFAULT_PROGRESS_JSONL)
    ap.add_argument("--out-json", default=DEFAULT_REPORT_JSON)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--step-limit", type=int, default=120)
    ap.add_argument("--cmd-timeout", type=float, default=180.0)
    ap.add_argument("--reasoning-effort", default="max")
    ap.add_argument("--think-ceiling", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-context", type=int, default=60000)
    ap.add_argument("--target-max-seq-len", type=int, default=65536)
    ap.add_argument("--use-speculative", action="store_true")
    ap.add_argument("--min-free-gib", type=float, default=80.0)
    ap.add_argument("--max-wall-s", type=float, default=1100.0)
    args = ap.parse_args()

    if args.golden_gate:
        run_golden_gate(
            repo_dir=args.repo_dir,
            udocker_dir=args.udocker_dir,
            execmode=args.execmode,
        )
        return 0

    eff = int(args.reasoning_effort) if str(args.reasoning_effort).isdigit() else args.reasoning_effort
    run_deepswe_slice(
        repo_dir=args.repo_dir,
        task_order_json=args.task_order_json,
        progress_jsonl=args.progress_jsonl,
        out_json=args.out_json,
        udocker_dir=args.udocker_dir,
        execmode=args.execmode,
        start_idx=args.start_idx,
        limit=args.limit,
        step_limit=args.step_limit,
        cmd_timeout=args.cmd_timeout,
        reasoning_effort=eff,
        think_ceiling=args.think_ceiling,
        temperature=args.temperature,
        top_p=args.top_p,
        max_context=args.max_context,
        target_max_seq_len=args.target_max_seq_len,
        use_speculative=args.use_speculative,
        min_free_gib=args.min_free_gib,
        max_wall_s=args.max_wall_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
