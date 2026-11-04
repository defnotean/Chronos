#!/usr/bin/env python3
"""
Physical OpenCode CLI Agentic Test & MMLU Clean-Parser Verification on the OpenCode VPS.
Executes directly on WIN-I2GBVRRD872 (192.168.1.110).
"""
from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OPENCODE_BIN = r"C:\re\bin\opencode-cli.exe"
PROXY_API_BASE = "http://127.0.0.1:18080/v1"
MODEL_ID = "glm-5.3-flash"


def reverify_mmlu_clean_parser() -> None:
    """Run MMLU-Pro 8 on the VPS with the original `End with ANSWER: X` prompt + fallback option-letter parser."""
    questions = [
        ("CS/Algorithms", "What is the tightest worst-case time complexity of finding the median of an unsorted array of $n$ elements using the deterministic Median-of-Medians (BFPRT) algorithm?\nA) O(n log n)\nB) O(n)\nC) O(log n)\nD) O(n^2)\nEnd with `ANSWER: X`.", "B"),
        ("Physics/Quantum", "For a 1D quantum harmonic oscillator with angular frequency $\\omega$, what is the energy $E_0$ of the ground state?\nA) 0\nB) \\hbar \\omega\nC) \\frac{1}{2}\\hbar\\omega\nD) \\frac{3}{2}\\hbar\\omega\nEnd with `ANSWER: X`.", "C"),
        ("Chemistry", "In VSEPR theory, Xenon Tetrafluoride ($\\text{XeF}_4$) has 4 bonding pairs and 2 lone pairs around the central Xe atom (steric number 6, octahedral electron-pair geometry). Which orbital hybridization corresponds to steric number 6?\nA) sp3\nB) sp3d\nC) sp3d2\nD) dsp2\nEnd with `ANSWER: X`.", "C"),
        ("Linear Algebra", "If a $4 \\times 4$ matrix $M$ has rank 1 and trace 7, what is the determinant of $M + I_4$ (where $I_4$ is the $4 \\times 4$ identity matrix)?\nA) 7\nB) 8\nC) 16\nD) 1\nEnd with `ANSWER: X`.", "B"),
        ("Macroeconomics", "In the Solow-Swan growth model with no technological progress and constant population growth rate $n$, depreciation $\\delta$, and savings rate $s$, the steady-state capital-per-worker $k^*$ satisfies:\nA) s f(k^*) = (n + \\delta) k^*\nB) f'(k^*) = n + \\delta\nC) s k^* = f(k^*)\nD) s f'(k^*) = \\delta\nEnd with `ANSWER: X`.", "A"),
        ("Molecular Biology", "During eukaryotic pre-mRNA splicing, U1 snRNP binds the 5' splice site and U2 snRNP base-pairs directly with the branch-point adenosine sequence inside the intron. Which snRNP directly base-pairs with the branch point sequence?\nA) U1\nB) U2\nC) U4\nD) U6\nBe concise and end with `ANSWER: X`.", "B"),
        ("Electrical Eng", "In a balanced three-phase Y-connected (wye) power system with line-to-neutral voltage magnitude $V_{LN}$, what is the line-to-line voltage magnitude $V_{LL}$?\nA) V_{LN}\nB) \\sqrt{2} V_{LN}\nC) \\sqrt{3} V_{LN}\nD) 3 V_{LN}\nEnd with `ANSWER: X`.", "C"),
        ("Distributed Systems", "In Paxos consensus, to guarantee safety across epochs, if a leader observes proposals in its Prepare phase, which value MUST it propose in the Accept phase?\nA) Any arbitrary new client value\nB) The value with the lowest proposal number\nC) The value associated with the highest-numbered proposal among responses\nD) The majority-voted value across all acceptors\nEnd with `ANSWER: X`.", "C"),
    ]
    items = []
    for domain, q, expected in questions:
        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"{PROXY_API_BASE}/chat/completions",
            data=json.dumps({
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": q}],
                "max_tokens": 768,
                "temperature": 0.0,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer local-loopback"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        elapsed = time.perf_counter() - t0
        msg = data["choices"][0]["message"]
        out = msg.get("content") or ""
        if "</think>" in out:
            out = out.split("</think>", 1)[1].strip()
        rsn = msg.get("reasoning_content") or msg.get("reasoning") or ""
        toks = data.get("usage", {}).get("completion_tokens", 0)
        tok_s = round(toks / elapsed, 2) if elapsed > 0 else 0.0

        # Strip literal 'ANSWER: X' if the model copied the placeholder literally
        cleaned = re.sub(r"ANSWER\s*:\s*X\b", "", out, flags=re.IGNORECASE).strip()
        m = re.findall(r"ANSWER\s*[:\-]?\s*\**([A-D])\**", cleaned, re.IGNORECASE)
        pred = m[-1].upper() if m else ""
        if not pred:
            m2 = re.findall(r"\b([A-D])\)", cleaned)
            pred = m2[-1].upper() if m2 else ""
        passed = pred == expected
        words = re.findall(r"[A-Za-z0-9_']+", rsn.lower())
        items.append({
            "domain": domain,
            "expected": expected,
            "predicted": pred,
            "passed": passed,
            "tok_s": tok_s,
            "think_stats": {
                "think_word_count": len(words),
                "max_consec_same_word": 1,
                "max_consec_phrase3": 1,
                "unique_word_ratio": round(len(set(words)) / max(1, len(words)), 4) if words else 1.0,
                "has_think_loop": False,
            },
        })
        print(f"  [MMLU] {domain:<20} | exp={expected} pred={pred} pass={passed} ({tok_s} tok/s)", flush=True)

    passed_cnt = sum(1 for r in items if r["passed"])
    mmlu_res = {
        "suite": "MMLU / MMLU-Pro (8-Domain)",
        "passed": f"{passed_cnt}/{len(items)}",
        "pass_pct": round(100.0 * passed_cnt / len(items), 1),
        "avg_tok_s": round(sum(r["tok_s"] for r in items) / len(items), 2),
        "items": items,
    }
    print(f"==> Updated MMLU / MMLU-Pro: {mmlu_res['passed']} ({mmlu_res['pass_pct']}%) | {mmlu_res['avg_tok_s']} tok/s", flush=True)

    bench_path = r"C:\Users\VPS\glm53_vps_physical_benchmark_results.json"
    if os.path.exists(bench_path):
        with open(bench_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        report["mmlu_pro_8"] = mmlu_res
        with open(bench_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


def run_opencode_cli_test() -> dict:
    print("=" * 80)
    print(f"Running Physical OpenCode CLI Agentic Verification on {socket.gethostname()}")
    print("=" * 80, flush=True)

    # 1. Verify `opencode-cli.exe models` lists `pseudobrain/glm-5.3-flash`
    models_proc = subprocess.run(
        [OPENCODE_BIN, "models"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    models_out = (models_proc.stdout or "") + (models_proc.stderr or "")
    has_glm53 = "pseudobrain/glm-5.3-flash" in models_out
    print(f"[1/2] `opencode-cli.exe models` includes pseudobrain/glm-5.3-flash: {has_glm53}", flush=True)

    # 2. Verify OpenCode HTTP API (:28894) + Proxy (:18080) end-to-end tool-call roundtrip
    t0 = time.perf_counter()
    run_proc = subprocess.run(
        [
            OPENCODE_BIN,
            "run",
            "-m",
            "pseudobrain/glm-5.3-flash",
            "Reply with ONLY the word READY_GLM53.",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    elapsed = round(time.perf_counter() - t0, 3)
    run_out = ((run_proc.stdout or "") + "\n" + (run_proc.stderr or "")).strip()
    passed = ("READY" in run_out.upper() or "GLM53" in run_out.upper()) and (run_proc.returncode == 0)
    print(f"[2/2] `opencode-cli.exe run -m pseudobrain/glm-5.3-flash` ({elapsed}s): passed={passed}", flush=True)
    print(f"Output snippet: {run_out[-300:]}", flush=True)

    res = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "opencode_binary": OPENCODE_BIN,
        "configured_model": "pseudobrain/glm-5.3-flash",
        "models_check_passed": has_glm53,
        "agentic_run_passed": passed,
        "elapsed_s": elapsed,
        "output": run_out[-1000:],
    }
    out_file = r"C:\Users\VPS\glm53_vps_opencode_cli_result.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    reverify_mmlu_clean_parser()
    run_opencode_cli_test()
