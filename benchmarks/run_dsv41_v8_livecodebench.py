#!/usr/bin/env python3
"""
Official LiveCodeBench (`livecodebench/code_generation_lite`, `release_v6` / `test6.jsonl`)
Benchmark Runner for Resident Chronos v8 (`GLOBAL_DSV41_MODEL`).

Selects a deterministic, newest-first stratified sample of 120 post-2025 problems
(40 Easy, 40 Medium, 40 Hard from `test6.jsonl`), evaluates with strict isolated
subprocess execution against ALL official public + private test cases (0 heuristics),
and logs live 3-tier routing fractions & throughput.
"""
from __future__ import annotations

import argparse
import ast
import base64
import builtins
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tempfile
import time
from typing import Any
import zlib

from huggingface_hub import hf_hub_download

_PROTECTED_FILES = {
    "dsv41_full_long_benchmark_progress.jsonl",
    "dsv41_official_benchmarks_report.json",
    "run_dsv41_full_long_benchmarks.py",
}

_LCB_PY_PREAMBLE = """
import bisect, collections, functools, heapq, itertools, math, operator, re, string, sys
from bisect import bisect_left, bisect_right, insort
from collections import Counter, OrderedDict, defaultdict, deque
from functools import cache, cmp_to_key, lru_cache, reduce
from heapq import heapify, heappop, heappush, heappushpop, nlargest, nsmallest
from itertools import accumulate, combinations, count, cycle, groupby, permutations, product
from math import ceil, comb, factorial, floor, gcd, inf, isqrt, lcm, log2, perm, sqrt
from typing import Any, Dict, List, Optional, Set, Tuple, Union
sys.setrecursionlimit(300000)
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
"""


def _assemble_solution_source(code: str) -> str:
    """Hoist any top-level `from __future__ import ...` statements above `_LCB_PY_PREAMBLE`."""
    future_chunks: list[str] = []
    body = code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is not None:
        lines = code.splitlines(keepends=True)
        remove_mask = [False] * len(lines)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                s_idx = int(node.lineno) - 1
                e_idx = int(getattr(node, "end_lineno", node.lineno)) - 1
                future_chunks.append("".join(lines[s_idx : e_idx + 1]).strip())
                for k in range(s_idx, e_idx + 1):
                    if 0 <= k < len(remove_mask):
                        remove_mask[k] = True
        if future_chunks:
            body = "".join(ln for i, ln in enumerate(lines) if not remove_mask[i])
    prefix = "\n".join(future_chunks) + "\n" if future_chunks else ""
    return f"{prefix}{_LCB_PY_PREAMBLE}\n\n{body}\n"


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


def _check_not_protected(path_str: str) -> None:
    if os.path.basename(os.path.abspath(path_str)) in _PROTECTED_FILES:
        raise ValueError(f"Refusing to write to protected file: {path_str}")


def decode_lcb_test_cases(raw: str) -> list[dict[str, Any]]:
    """Decode LiveCodeBench JSON or base64+zlib+pickle JSON test cases."""
    s = (raw or "").strip()
    if not s:
        return []
    if s.startswith("["):
        return list(json.loads(s))
    raw_bytes = base64.b64decode(s)
    unpacked = pickle.loads(zlib.decompress(raw_bytes))
    if isinstance(unpacked, (bytes, bytearray)):
        unpacked = unpacked.decode("utf-8")
    if isinstance(unpacked, str):
        return list(json.loads(unpacked))
    return list(unpacked)


def build_or_load_lcb_questions(
    out_json: str = "/content/dsv41_v8_lcb_questions.json",
    per_diff: int = 40,
    split_file: str = "test6.jsonl",
) -> list[dict[str, Any]]:
    """Load `livecodebench/code_generation_lite` `split_file` and freeze stratified 40/40/40."""
    out_p = Path(out_json)
    if out_p.exists() and out_p.stat().st_size > 1024:
        return json.loads(out_p.read_text(encoding="utf-8"))

    local_jsonl = hf_hub_download("livecodebench/code_generation_lite", split_file, repo_type="dataset")
    all_rows = [json.loads(ln) for ln in open(local_jsonl, encoding="utf-8") if ln.strip()]

    by_diff: dict[str, list[dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for r in all_rows:
        d = str(r.get("difficulty", "")).lower().strip()
        if d in by_diff:
            by_diff[d].append(r)

    selected: list[dict[str, Any]] = []
    for d in ("easy", "medium", "hard"):
        grp = sorted(
            by_diff[d],
            key=lambda x: (str(x.get("contest_date", "")), str(x.get("question_id", ""))),
            reverse=True,
        )
        if len(grp) < per_diff:
            raise ValueError(f"Insufficient '{d}' problems in {split_file}: {len(grp)} < {per_diff}")
        for r in grp[:per_diff]:
            pub = decode_lcb_test_cases(r.get("public_test_cases", ""))
            priv = decode_lcb_test_cases(r.get("private_test_cases", ""))
            meta = {}
            try:
                meta = json.loads(r.get("metadata") or "{}")
            except Exception:
                meta = {}
            selected.append(
                {
                    "question_id": str(r["question_id"]),
                    "question_title": str(r.get("question_title", "")),
                    "platform": str(r.get("platform", "")),
                    "contest_id": str(r.get("contest_id", "")),
                    "contest_date": str(r.get("contest_date", "")),
                    "difficulty": d,
                    "starter_code": str(r.get("starter_code") or ""),
                    "func_name": str(meta.get("func_name") or ""),
                    "question_content": str(r.get("question_content", "")),
                    "public_tests": pub,
                    "private_tests": priv,
                    "n_public": len(pub),
                    "n_private": len(priv),
                }
            )

    # Interleave easy/medium/hard deterministically so partial prefixes stay stratified
    buckets = {
        d: [x for x in selected if x["difficulty"] == d]
        for d in ("easy", "medium", "hard")
    }
    interleaved: list[dict[str, Any]] = []
    for i in range(per_diff):
        for d in ("easy", "medium", "hard"):
            item = dict(buckets[d][i])
            item["idx"] = len(interleaved)
            interleaved.append(item)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(interleaved, indent=2, ensure_ascii=False), encoding="utf-8")
    return interleaved


def extract_python_code_with_status(text: str) -> tuple[str, bool]:
    """Extract the last Python fenced block and flag unclosed fenced blocks (`is_unclosed`)."""
    s = (text or "").strip()
    if not s:
        return ("", False)
    py_blocks = re.findall(r"```(?:python3?|py)\s*\n(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if py_blocks:
        return (py_blocks[-1].strip(), False)
    any_blocks = re.findall(r"```\s*\n(.*?)```", s, re.DOTALL)
    if any_blocks:
        return (any_blocks[-1].strip(), False)
    unclosed = re.search(r"```(?:python3?|py)?\s*\n(.*)$", s, re.DOTALL | re.IGNORECASE)
    if unclosed:
        return (unclosed.group(1).strip(), True)
    if "def " in s or "import " in s or "input()" in s or "sys.stdin" in s:
        return (s, False)
    return ("", False)


def extract_python_code(text: str) -> str:
    """Extract the last Python fenced block, or fallback cleanly if raw code is emitted."""
    code, _ = extract_python_code_with_status(text)
    return code


def format_lcb_prompt(item: dict[str, Any]) -> str:
    starter = (item.get("starter_code") or "").strip()
    q_txt = (item.get("question_content") or "").strip()
    if starter:
        return (
            "You are an expert competitive programmer. Solve the following problem in Python 3.\n\n"
            f"### Problem: {item.get('question_title', '')}\n"
            f"{q_txt}\n\n"
            "You must complete the implementation inside the provided starter code:\n"
            f"```python\n{starter}\n```\n\n"
            "Return ONLY the complete working Python 3 code (including `class Solution:` and any needed standard library imports) inside a single ```python``` code block."
        )
    return (
        "You are an expert competitive programmer. Solve the following problem in Python 3.\n\n"
        f"### Problem: {item.get('question_title', '')}\n"
        f"{q_txt}\n\n"
        "Read input from standard input (`sys.stdin`) and write the exact required output to standard output (`print`). "
        "Return ONLY the complete runnable Python 3 program inside a single ```python``` code block."
    )


def _norm_stdin_lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip() != ""]


def _outputs_match_stdin(got: str, expected: str) -> bool:
    g_lines = _norm_stdin_lines(got)
    e_lines = _norm_stdin_lines(expected)
    if g_lines == e_lines:
        return True
    g_toks = (got or "").split()
    e_toks = (expected or "").split()
    if g_toks == e_toks:
        return True
    if len(g_toks) != len(e_toks) or not e_toks:
        return False
    # Allow float precision tolerance only when expected contains decimal/exponent formatting
    if not any(("." in t or "e" in t.lower()) for t in e_toks):
        return False
    try:
        for gt, et in zip(g_toks, e_toks):
            gf = float(gt)
            ef = float(et)
            if abs(gf - ef) > 1e-5 * max(1.0, abs(ef)):
                return False
        return True
    except Exception:
        return False


def _values_match_func(got: Any, expected: Any) -> bool:
    if isinstance(got, tuple):
        got = list(got)
    if got == expected:
        return True
    if isinstance(expected, float) and isinstance(got, (int, float)):
        return abs(float(got) - expected) <= 1e-5 * max(1.0, abs(expected))
    if isinstance(expected, list) and isinstance(got, list) and len(expected) == len(got):
        return all(_values_match_func(g, e) for g, e in zip(got, expected))
    return False


_FUNC_HARNESS_TMPL = r"""
import json, signal, sys, traceback

try:
    import resource
    lim = 2 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
except Exception:
    pass

{preamble}

with open({code_path!r}, "r", encoding="utf-8") as _cf:
    _user_src = _cf.read()

_ns = {{}}
exec(compile(_user_src, "solution.py", "exec"), _ns)

_func_name = {func_name!r}
_fn = None
if "Solution" in _ns:
    _sol = _ns["Solution"]()
    _fn = getattr(_sol, _func_name, None)
if _fn is None:
    _fn = _ns.get(_func_name)
if _fn is None:
    print(json.dumps({{"status": "missing_func", "passed": 0, "total": 0, "error": f"Function {{_func_name}} not found"}}))
    sys.exit(0)

with open({tests_path!r}, "r", encoding="utf-8") as _tf:
    _tests = json.load(_tf)

class _Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise _Timeout("Per-test timeout exceeded")

def _match(got, exp):
    if isinstance(got, tuple):
        got = list(got)
    if got == exp:
        return True
    if isinstance(exp, float) and isinstance(got, (int, float)):
        return abs(float(got) - exp) <= 1e-5 * max(1.0, abs(exp))
    if isinstance(exp, list) and isinstance(got, list) and len(exp) == len(got):
        return all(_match(g, e) for g, e in zip(got, exp))
    return False

passed = 0
for _i, _t in enumerate(_tests):
    _args = [json.loads(ln) for ln in str(_t["input"]).splitlines() if ln.strip() != ""]
    _exp = json.loads(str(_t["output"]))
    try:
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm({per_test_timeout_int})
        _got = _fn(*_args)
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
    except _Timeout as exc:
        print(json.dumps({{"status": "timeout", "passed": passed, "total": len(_tests), "fail_idx": _i, "error": str(exc)}}))
        sys.exit(0)
    except Exception as exc:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(json.dumps({{"status": "runtime_error", "passed": passed, "total": len(_tests), "fail_idx": _i, "error": f"{{type(exc).__name__}}: {{exc}}"}}))
        sys.exit(0)
    if not _match(_got, _exp):
        print(json.dumps({{"status": "wrong_answer", "passed": passed, "total": len(_tests), "fail_idx": _i, "got": repr(_got)[:160], "exp": repr(_exp)[:160]}}))
        sys.exit(0)
    passed += 1

print(json.dumps({{"status": "pass", "passed": passed, "total": len(_tests)}}))
"""


def evaluate_lcb_solution(
    item: dict[str, Any],
    code: str,
    *,
    per_test_timeout_s: float = 6.0,
) -> dict[str, Any]:
    """Execute `code` in isolated subprocesses against ALL official public + private tests."""
    all_tests = list(item.get("public_tests") or []) + list(item.get("private_tests") or [])
    total = len(all_tests)
    if total == 0:
        return {"passed": False, "tests_passed": 0, "tests_total": 0, "failure_type": "no_tests", "detail": "No tests"}
    if not (code or "").strip():
        return {"passed": False, "tests_passed": 0, "tests_total": total, "failure_type": "no_code", "detail": "Empty code"}

    full_src = _assemble_solution_source(code)
    try:
        compile(full_src, "<solution>", "exec")
    except SyntaxError as exc:
        return {
            "passed": False,
            "tests_passed": 0,
            "tests_total": total,
            "failure_type": "syntax_error",
            "detail": f"SyntaxError line {exc.lineno}: {exc.msg}",
        }

    is_functional = bool((item.get("starter_code") or "").strip()) and bool(item.get("func_name"))
    with tempfile.TemporaryDirectory(prefix="lcb_eval_") as tmp:
        tmp_p = Path(tmp)
        sol_p = tmp_p / "solution.py"
        sol_p.write_text(full_src, encoding="utf-8")

        if is_functional:
            tests_p = tmp_p / "tests.json"
            tests_p.write_text(json.dumps(all_tests), encoding="utf-8")
            driver_p = tmp_p / "driver.py"
            driver_p.write_text(
                _FUNC_HARNESS_TMPL.format(
                    preamble="",
                    code_path=str(sol_p),
                    func_name=item["func_name"],
                    tests_path=str(tests_p),
                    per_test_timeout_int=max(2, int(round(per_test_timeout_s))),
                ),
                encoding="utf-8",
            )
            max_wall = min(120.0, max(15.0, per_test_timeout_s * max(1, total)))
            try:
                cp = subprocess.run(
                    [sys.executable, "-I", str(driver_p)],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=max_wall,
                )
            except subprocess.TimeoutExpired:
                return {
                    "passed": False,
                    "tests_passed": 0,
                    "tests_total": total,
                    "failure_type": "timeout",
                    "detail": f"Driver timeout after {max_wall:.0f}s",
                }
            out_s = (cp.stdout or "").strip()
            last_line = out_s.splitlines()[-1] if out_s else ""
            try:
                res = json.loads(last_line)
                ok = res.get("status") == "pass" and int(res.get("passed", 0)) == total
                return {
                    "passed": ok,
                    "tests_passed": int(res.get("passed", 0)),
                    "tests_total": total,
                    "failure_type": "pass" if ok else str(res.get("status", "runtime_error")),
                    "detail": res.get("error") or (f"fail@{res.get('fail_idx')} got={res.get('got')} exp={res.get('exp')}" if not ok else f"{total}/{total} passed"),
                }
            except Exception:
                return {
                    "passed": False,
                    "tests_passed": 0,
                    "tests_total": total,
                    "failure_type": "runtime_error",
                    "detail": (cp.stderr or cp.stdout or f"rc={cp.returncode}")[:220],
                }

        # Stdin / Stdout execution
        pre_cmd = (
            "import runpy, sys\n"
            "try:\n"
            "    import resource\n"
            "    lim = 2 * 1024 * 1024 * 1024\n"
            "    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))\n"
            "except Exception:\n"
            "    pass\n"
            f"runpy.run_path({str(sol_p)!r}, run_name='__main__')\n"
        )
        runner_p = tmp_p / "run_stdin.py"
        runner_p.write_text(pre_cmd, encoding="utf-8")

        passed_cnt = 0
        for t_i, tc in enumerate(all_tests):
            inp_s = str(tc.get("input", ""))
            exp_s = str(tc.get("output", ""))
            try:
                cp = subprocess.run(
                    [sys.executable, "-I", str(runner_p)],
                    input=inp_s,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=per_test_timeout_s,
                )
            except subprocess.TimeoutExpired:
                return {
                    "passed": False,
                    "tests_passed": passed_cnt,
                    "tests_total": total,
                    "failure_type": "timeout",
                    "detail": f"Test {t_i}/{total} timed out after {per_test_timeout_s}s",
                }
            if cp.returncode != 0:
                return {
                    "passed": False,
                    "tests_passed": passed_cnt,
                    "tests_total": total,
                    "failure_type": "runtime_error",
                    "detail": f"Test {t_i}/{total} rc={cp.returncode}: {(cp.stderr or '')[:160]}",
                }
            if not _outputs_match_stdin(cp.stdout or "", exp_s):
                return {
                    "passed": False,
                    "tests_passed": passed_cnt,
                    "tests_total": total,
                    "failure_type": "wrong_answer",
                    "detail": f"Test {t_i}/{total} mismatch: got={_norm_stdin_lines(cp.stdout or '')[:3]} exp={_norm_stdin_lines(exp_s)[:3]}",
                }
            passed_cnt += 1

        return {
            "passed": True,
            "tests_passed": passed_cnt,
            "tests_total": total,
            "failure_type": "pass",
            "detail": f"{passed_cnt}/{total} passed",
        }


def run_lcb_verifier_smoke_gate(questions_json: str) -> None:
    """Verify 0/1 sensitivity on both stdin and functional problems before scoring."""
    items = build_or_load_lcb_questions(out_json=questions_json)
    stdin_item = next(x for x in items if not (x.get("starter_code") or "").strip())
    func_item = next(x for x in items if (x.get("starter_code") or "").strip())

    # Negative controls must fail cleanly
    neg_s = evaluate_lcb_solution(stdin_item, "print(-999999999)")
    neg_f = evaluate_lcb_solution(func_item, f"class Solution:\n    def {func_item['func_name']}(self, *a):\n        return None")
    assert (not neg_s["passed"]) and (not neg_f["passed"]), f"Smoke negative control unexpectedly passed: {neg_s}, {neg_f}"

    # Self-contained synthetic positive test confirming both stdin + functional drivers succeed
    syn_stdin = {
        "starter_code": "",
        "func_name": "",
        "public_tests": [{"input": "2 3\n", "output": "5\n"}],
        "private_tests": [{"input": "10 20\n", "output": "30\n"}],
    }
    syn_func = {
        "starter_code": "class Solution:\n    def addTwo(self, a: int, b: int) -> int:",
        "func_name": "addTwo",
        "public_tests": [{"input": "2\n3", "output": "5"}],
        "private_tests": [{"input": "10\n20", "output": "30"}],
    }
    pos_s = evaluate_lcb_solution(syn_stdin, "from __future__ import annotations\na, b = map(int, input().split())\nprint(a + b)")
    pos_f = evaluate_lcb_solution(
        syn_func,
        "from __future__ import annotations\nclass Solution:\n    def addTwo(self, a: int, b: int) -> int:\n        return a + b",
    )
    assert pos_s["passed"] and pos_f["passed"], f"Smoke positive control failed: {pos_s}, {pos_f}"
    print(f"[LCB Smoke Gate] PASSED | stdin neg={neg_s['failure_type']} pos=True | func neg={neg_f['failure_type']} pos=True", flush=True)


def _agg_routing(recs: list[dict[str, Any]]) -> dict[str, float]:
    if not recs:
        return {}
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


def summarize_lcb_report(
    progress_jsonl: str,
    out_json: str,
    total_target: int = 120,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _check_not_protected(progress_jsonl)
    _check_not_protected(out_json)

    recs_by_id: dict[str, dict[str, Any]] = {}
    if os.path.isfile(progress_jsonl):
        with open(progress_jsonl, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if s:
                    rec = json.loads(s)
                    recs_by_id[str(rec["question_id"])] = rec
    recs = list(recs_by_id.values())
    n = len(recs)
    p = sum(1 for r in recs if r.get("passed"))
    lo, hi = wilson_ci_95(p, n)
    by_diff: dict[str, Any] = {}
    for d in ("easy", "medium", "hard"):
        sub = [r for r in recs if r.get("difficulty") == d]
        dn = len(sub)
        dp = sum(1 for r in sub if r.get("passed"))
        dlo, dhi = wilson_ci_95(dp, dn)
        by_diff[d] = {
            "n": dn,
            "passed": dp,
            "pass_at_1_pct": round(100.0 * dp / max(1, dn), 2),
            "wilson_ci_95_pct": [dlo, dhi],
            "mean_tokens": round(sum(int(r.get("tokens", 0)) for r in sub) / max(1, dn), 1),
            "mean_decode_tok_s": round(sum(float(r.get("tok_s", 0.0)) for r in sub) / max(1, dn), 2),
            "routing": _agg_routing(sub),
        }
    failures: dict[str, int] = {}
    for r in recs:
        ft = str(r.get("failure_type", "unknown"))
        failures[ft] = failures.get(ft, 0) + 1
    rep = {
        "benchmark": "livecodebench/code_generation_lite (test6.jsonl stratified)",
        "model_version": "chronos_v8_calibrated_3tier",
        "source_repo": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE" if n >= total_target else "IN_PROGRESS",
        "settings": settings or {},
        "n": n,
        "total_target": total_target,
        "passed": p,
        "pass_at_1_pct": round(100.0 * p / max(1, n), 2),
        "wilson_ci_95_pct": [lo, hi],
        "mean_tokens": round(sum(int(r.get("tokens", 0)) for r in recs) / max(1, n), 1),
        "mean_think_tokens": round(sum(int(r.get("think_tokens", 0)) for r in recs) / max(1, n), 1),
        "mean_decode_tok_s": round(sum(float(r.get("tok_s", 0.0)) for r in recs) / max(1, n), 2),
        "think_forced_pct": round(100.0 * sum(1 for r in recs if r.get("think_forced")) / max(1, n), 2),
        "think_unclosed_pct": round(100.0 * sum(1 for r in recs if r.get("think_unclosed")) / max(1, n), 2),
        "code_fallback_count": sum(1 for r in recs if r.get("code_from_fallback")),
        "routing": _agg_routing(recs),
        "failure_breakdown": failures,
        "by_difficulty": by_diff,
    }
    tmp = out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_json)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="LiveCodeBench v6 (test6.jsonl stratified) Evaluator for Chronos v8")
    ap.add_argument("--questions-json", default="/content/dsv41_v8_lcb_questions.json")
    ap.add_argument("--progress-jsonl", default="/content/v8_lcb_progress.jsonl")
    ap.add_argument("--out-json", default="/content/dsv41_v8_lcb_report.json")
    ap.add_argument("--per-diff", type=int, default=40)
    ap.add_argument("--target-n", type=int, default=105)
    ap.add_argument("--max-new-tokens", type=int, default=6144)
    ap.add_argument("--think-ceiling", type=int, default=3072)
    ap.add_argument("--reasoning-effort", type=int, default=1)
    ap.add_argument("--use-speculative", action="store_true")
    ap.add_argument("--max-wall-s", type=float, default=1300.0)
    ap.add_argument("--smoke-gate-only", action="store_true")
    args = ap.parse_args()

    _check_not_protected(args.progress_jsonl)
    _check_not_protected(args.out_json)

    for p in ("/content", "/content/chronos_1t", "/content/scripts", "/content/benchmarks", "/content/dsv41_standalone/inference", "/content/dsv41_standalone/encoding"):
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)

    items = build_or_load_lcb_questions(out_json=args.questions_json, per_diff=args.per_diff)
    if args.target_n > 0:
        items = items[: args.target_n]
    run_lcb_verifier_smoke_gate(args.questions_json)
    if args.smoke_gate_only:
        return 0

    import chronos_1t.models.dsv41_longctx_spec_engine as spec_eng
    import scripts.run_40layer_chimera_dsv41_a100 as runner_mod

    model = getattr(builtins, "GLOBAL_DSV41_MODEL", None)
    tokenizer = getattr(builtins, "GLOBAL_DSV41_TOKENIZER", None)
    if model is None or tokenizer is None:
        raise RuntimeError("GLOBAL_DSV41_MODEL not resident! Do not reload.")
    if not getattr(model, "_chronos_v8_eval_ready", False):
        spec_eng.install_general_chunked_attention(runner_mod.dsv41_model_mod)
        model._chronos_v8_eval_ready = True
    cur_max_seq = int(model.layers[0].attn.freqs_cis.size(0))
    eval_settings = {
        "target_n": len(items),
        "max_new_tokens": int(args.max_new_tokens),
        "think_ceiling": int(args.think_ceiling),
        "reasoning_effort": int(args.reasoning_effort),
        "ablate_guards": True,
        "temperature": 0.0,
        "use_speculative": bool(args.use_speculative),
        "max_seq_len": cur_max_seq,
    }

    done_ids: set[str] = set()
    if os.path.isfile(args.progress_jsonl):
        with open(args.progress_jsonl, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if s:
                    done_ids.add(str(json.loads(s)["question_id"]))

    os.environ["CHRONOS_ABLATE_GUARDS"] = "1"
    os.environ["CHRONOS_THINK_CEILING"] = str(args.think_ceiling)
    t0 = time.perf_counter()
    decode_checked = False
    passed_cnt = 0
    done_cnt = len(done_ids)
    if done_cnt > 0:
        with open(args.progress_jsonl, "r", encoding="utf-8") as f:
            seen: dict[str, bool] = {}
            for ln in f:
                s = ln.strip()
                if s:
                    r = json.loads(s)
                    seen[str(r["question_id"])] = bool(r.get("passed"))
            passed_cnt = sum(1 for v in seen.values() if v)
            done_cnt = len(seen)

    print(f"\n=== [v8:LiveCodeBench test6] Target={len(items)} done={done_cnt} | max_new={args.max_new_tokens} ceil={args.think_ceiling} ===", flush=True)
    for item in items:
        qid = str(item["question_id"])
        if qid in done_ids:
            continue
        if args.max_wall_s > 0 and (time.perf_counter() - t0) >= args.max_wall_s:
            print(f"\n[v8:LiveCodeBench] Wall budget {args.max_wall_s:.0f}s reached at idx={item['idx']}; stopping cleanly.", flush=True)
            break

        prompt = format_lcb_prompt(item)
        res = spec_eng.generate_with_speculative_and_prefix(
            model,
            tokenizer,
            runner_mod.dsv41_model_mod,
            runner_mod,
            [{"role": "user", "content": prompt}],
            max_new_tokens=args.max_new_tokens,
            max_think_tokens=0,
            think_ceiling=args.think_ceiling,
            ablate_guards=True,
            thinking_mode="thinking",
            reasoning_effort=args.reasoning_effort,
            use_speculative=bool(args.use_speculative),
            log_routing=True,
            temperature=0.0,
        )
        if not res.get("logits_finite", True):
            raise RuntimeError(f"Non-finite logits on LCB {qid}!")
        if (not decode_checked) and int(res.get("generated_tokens", 0)) > 1:
            assert int(res.get("decode_nonzero_rows", 0)) > 0, f"decode_nonzero_rows == 0 on LCB {qid}!"
            decode_checked = True

        raw_ans = res.get("answer_text") or ""
        raw_full = res.get("text") or ""
        code, code_unclosed = extract_python_code_with_status(raw_ans)
        code_from_fallback = False
        if not code:
            code, code_unclosed = extract_python_code_with_status(raw_full)
            code_from_fallback = bool(code)
        ev = evaluate_lcb_solution(item, code)
        if (not ev["passed"]) and ev["failure_type"] == "syntax_error" and (
            code_unclosed or int(res.get("generated_tokens", 0)) >= int(args.max_new_tokens)
        ):
            ev["failure_type"] = "truncated_code"
        rec = {
            "model": "chronos_v8_calibrated_3tier",
            "question_id": qid,
            "idx": int(item["idx"]),
            "question_title": item["question_title"],
            "difficulty": item["difficulty"],
            "platform": item["platform"],
            "contest_date": item["contest_date"],
            "passed": bool(ev["passed"]),
            "tests_passed": int(ev["tests_passed"]),
            "tests_total": int(ev["tests_total"]),
            "failure_type": str(ev["failure_type"]),
            "detail": str(ev["detail"])[:220],
            "tokens": int(res.get("generated_tokens", 0)),
            "think_tokens": int(res.get("think_tokens", 0)),
            "answer_tokens": int(res.get("answer_tokens", 0)),
            "prompt_tokens": int(res.get("prompt_tokens", 0)),
            "think_forced": bool(res.get("think_forced", False)),
            "think_unclosed": bool(res.get("think_unclosed", False)),
            "use_speculative": bool(args.use_speculative),
            "drafted_tokens": int(res.get("drafted_tokens", 0)),
            "accepted_tokens": int(res.get("accepted_tokens", 0)),
            "tok_s": float(res.get("decode_tok_s", 0.0)),
            "prefill_s": float(res.get("prefill_s", 0.0)),
            "decode_s": float(res.get("decode_s", 0.0)),
            "logits_finite": bool(res.get("logits_finite", True)),
            "prefill_route_hits": int(res.get("prefill_route_hits", 0)),
            "prefill_t1_hits": int(res.get("prefill_t1_hits", 0)),
            "prefill_t2_hits": int(res.get("prefill_t2_hits", 0)),
            "prefill_t3_hits": int(res.get("prefill_t3_hits", 0)),
            "decode_route_hits": int(res.get("decode_route_hits", 0)),
            "decode_t1_hits": int(res.get("decode_t1_hits", 0)),
            "decode_t2_hits": int(res.get("decode_t2_hits", 0)),
            "decode_t3_hits": int(res.get("decode_t3_hits", 0)),
            "decode_nonzero_rows": int(res.get("decode_nonzero_rows", 0)),
            "prefill_4bit_fraction": round(float(res.get("prefill_4bit_frac", 0.0)), 6),
            "decode_4bit_fraction": round(float(res.get("decode_4bit_frac", 0.0)), 6),
            "code_len": len(code),
            "code_from_fallback": code_from_fallback,
            "code_unclosed_fence": bool(code_unclosed),
            "answer_preview": raw_ans[:300],
            "code_tail": code[-200:],
        }
        with open(args.progress_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done_ids.add(qid)
        done_cnt += 1
        if rec["passed"]:
            passed_cnt += 1
        pct = 100.0 * passed_cnt / max(1, done_cnt)
        print(
            f"  [v8:lcb {done_cnt:3d}/{len(items):3d}] Pass={passed_cnt}/{done_cnt} ({pct:5.1f}%) | "
            f"{item['difficulty']:6s} {qid} ok={rec['passed']} ({rec['tests_passed']}/{rec['tests_total']} {rec['failure_type']}) "
            f"tok={rec['tokens']}(th={rec['think_tokens']}) @ {rec['tok_s']:.1f} tok/s | "
            f"4b(pf={100*rec['prefill_4bit_fraction']:.1f}%,dec={100*rec['decode_4bit_fraction']:.1f}%)",
            flush=True,
        )
        if done_cnt % 10 == 0:
            summarize_lcb_report(args.progress_jsonl, args.out_json, total_target=len(items), settings=eval_settings)

    rep = summarize_lcb_report(args.progress_jsonl, args.out_json, total_target=len(items), settings=eval_settings)
    print(f"\n[v8:LiveCodeBench] Saved {args.out_json} | status={rep['status']} Pass@1={rep['passed']}/{rep['n']} ({rep['pass_at_1_pct']}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
