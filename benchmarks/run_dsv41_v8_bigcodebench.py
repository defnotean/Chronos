#!/usr/bin/env python3
"""
Official BigCodeBench-Hard (`bigcode/bigcodebench-hard`, `v0.1.4`, n=148)
Instruct Benchmark Runner for Resident Chronos v8 (`GLOBAL_DSV41_MODEL`).

Evaluates all 148 hard complex multi-library tasks via direct isolated `unittest`
subprocess execution (no heuristics) with `RLIMIT_AS` + per-task timeout,
recording live 3-tier routing fractions, finite-logit checks, and throughput.
"""
from __future__ import annotations

import argparse
import ast
import builtins
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from huggingface_hub import hf_hub_download

_PROTECTED_FILES = {
    "dsv41_full_long_benchmark_progress.jsonl",
    "dsv41_official_benchmarks_report.json",
    "run_dsv41_full_long_benchmarks.py",
}


def _check_not_protected(path_str: str) -> None:
    if os.path.basename(os.path.abspath(path_str)) in _PROTECTED_FILES:
        raise ValueError(f"Refusing to write to protected file: {path_str}")


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


def _read_parquet_dicts(parquet_path: str) -> list[dict[str, Any]]:
    for loader in ("polars", "pyarrow", "pandas"):
        try:
            if loader == "polars":
                import polars as pl
                return pl.read_parquet(parquet_path).to_dicts()
            if loader == "pyarrow":
                import pyarrow.parquet as pq
                return pq.read_table(parquet_path).to_pylist()
            import pandas as pd
            return pd.read_parquet(parquet_path).to_dict(orient="records")
        except ImportError:
            continue
    raise RuntimeError("Need polars, pyarrow, or pandas to read bigcodebench-hard parquet")


def build_or_load_bcb_questions(
    out_json: str = "/content/dsv41_v8_bcb_hard_questions.json",
) -> list[dict[str, Any]]:
    """Download `bigcode/bigcodebench-hard` `v0.1.4` (n=148) and freeze locally."""
    out_p = Path(out_json)
    if out_p.exists() and out_p.stat().st_size > 1024:
        return json.loads(out_p.read_text(encoding="utf-8"))

    pq_path = hf_hub_download(
        "bigcode/bigcodebench-hard",
        "data/v0.1.4-00000-of-00001.parquet",
        repo_type="dataset",
    )
    rows = _read_parquet_dicts(pq_path)
    items: list[dict[str, Any]] = []
    for idx, r in enumerate(rows):
        items.append(
            {
                "idx": idx,
                "task_id": str(r["task_id"]),
                "entry_point": str(r.get("entry_point") or "task_func"),
                "instruct_prompt": str(r["instruct_prompt"]),
                "code_prompt": str(r.get("code_prompt") or ""),
                "canonical_solution": str(r.get("canonical_solution") or ""),
                "test": str(r["test"]),
                "libs": str(r.get("libs") or "[]"),
            }
        )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    return items


def format_bcb_instruct_prompt(item: dict[str, Any]) -> str:
    return (
        "Please provide a self-contained Python script that solves the following problem in a markdown code block:\n\n"
        f"{item['instruct_prompt'].strip()}\n\n"
        "Return the complete working Python code inside a single ```python``` code block."
    )


def extract_python_code_with_status(text: str, *, entry_point: str = "task_func", code_prompt: str = "") -> tuple[str, bool]:
    """Extract Python block and flag unclosed fence; restore `code_prompt` header only if body-only."""
    s = (text or "").strip()
    if not s:
        return ("", False)
    unclosed = False
    code = ""
    py_blocks = re.findall(r"```(?:python3?|py)\s*\n(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if py_blocks:
        code = py_blocks[-1].strip()
    else:
        any_blocks = re.findall(r"```\s*\n(.*?)```", s, re.DOTALL)
        if any_blocks:
            code = any_blocks[-1].strip()
        else:
            m_open = re.search(r"```(?:python3?|py)?\s*\n(.*)$", s, re.DOTALL | re.IGNORECASE)
            if m_open:
                code = m_open.group(1).strip()
                unclosed = True
            elif f"def {entry_point}" in s or "import " in s or "def " in s:
                code = s

    if code and entry_point and f"def {entry_point}" not in code and code_prompt.strip():
        lines = code.splitlines()
        first_nonempty = next((ln for ln in lines if ln.strip()), "")
        if first_nonempty.startswith(("    ", "\t")) or "return " in code:
            code = f"{code_prompt.rstrip()}\n{code}"
    return (code, unclosed)


def _hoist_future_imports(src: str) -> tuple[list[str], str]:
    """Extract top-level `from __future__ import ...` statements so harness imports stay valid."""
    if "__future__" not in src:
        return ([], src)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ([], src)

    lines = src.splitlines(keepends=True)
    remove_indices: set[int] = set()
    patched_lines: dict[int, str] = {}
    future_stmts: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            names = ", ".join(
                f"{a.name} as {a.asname}" if a.asname else a.name for a in node.names
            )
            future_stmts.append(f"from __future__ import {names}")
            start = node.lineno - 1
            end = (node.end_lineno or node.lineno) - 1
            end_col = node.end_col_offset or len(lines[end])
            tail = lines[end][end_col:].lstrip(" \t;") if end < len(lines) else ""
            if node.col_offset == 0 and (not tail.strip() or tail.strip().startswith("#")):
                for idx in range(start, end + 1):
                    remove_indices.add(idx)
            elif start == end:
                ln = lines[start]
                patched_lines[start] = ln[: node.col_offset] + ln[end_col:].lstrip(" \t;")

    out_parts = []
    for idx, ln in enumerate(lines):
        if idx in remove_indices:
            continue
        out_parts.append(patched_lines.get(idx, ln))
    return (future_stmts, "".join(out_parts))


def evaluate_bcb_solution(
    item: dict[str, Any],
    code: str,
    *,
    per_task_timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Execute `code` against official `item['test']` unittest suite in an isolated subprocess."""
    if not (code or "").strip():
        return {"passed": False, "failure_type": "no_code", "detail": "Empty code"}
    try:
        compile(code, "<solution>", "exec")
    except SyntaxError as exc:
        return {
            "passed": False,
            "failure_type": "syntax_error",
            "detail": f"SyntaxError line {exc.lineno}: {exc.msg}",
        }

    code_futures, stripped_code = _hoist_future_imports(code)
    test_futures, stripped_test = _hoist_future_imports(item["test"])
    seen_futures: list[str] = []
    for stmt in code_futures + test_futures:
        if stmt not in seen_futures:
            seen_futures.append(stmt)
    future_header = ("\n".join(seen_futures) + "\n") if seen_futures else ""

    harness = (
        f"{future_header}"
        "import os, sys, unittest\n"
        "os.environ.setdefault('MPLBACKEND', 'Agg')\n"
        "try:\n"
        "    import matplotlib\n"
        "    matplotlib.use('Agg', force=True)\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    import resource\n"
        "    lim = 32 * 1024 * 1024 * 1024\n"
        "    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))\n"
        "except Exception:\n"
        "    pass\n"
        "if hasattr(sys, 'set_int_max_str_digits'):\n"
        "    sys.set_int_max_str_digits(0)\n\n"
        f"{stripped_code}\n\n"
        f"{stripped_test}\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main(verbosity=1)\n"
    )
    try:
        compile(harness, "<test_harness>", "exec")
    except SyntaxError as exc:
        return {
            "passed": False,
            "failure_type": "syntax_error",
            "detail": f"Harness SyntaxError line {exc.lineno}: {exc.msg}",
        }

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    with tempfile.TemporaryDirectory(prefix="bcb_eval_") as tmp:
        script_p = Path(tmp) / "run_test.py"
        script_p.write_text(harness, encoding="utf-8")
        try:
            cp = subprocess.run(
                [sys.executable, str(script_p)],
                cwd=tmp,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=per_task_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "failure_type": "timeout",
                "detail": f"Timeout after {per_task_timeout_s:.0f}s",
            }
        out_comb = ((cp.stderr or "") + "\n" + (cp.stdout or "")).strip()
        if cp.returncode == 0:
            return {"passed": True, "failure_type": "pass", "detail": out_comb[-180:] or "OK"}
        if "AssertionError" in out_comb or "\nFAIL:" in out_comb:
            ft = "assertion_fail"
        elif "ModuleNotFoundError" in out_comb or "ImportError" in out_comb:
            ft = "import_error"
        else:
            ft = "runtime_error"
        return {"passed": False, "failure_type": ft, "detail": out_comb[-240:]}


def run_bcb_verifier_smoke_gate(questions_json: str) -> None:
    """Verify 0/1 sensitivity with negative stub + real canonical solution."""
    items = build_or_load_bcb_questions(out_json=questions_json)
    assert len(items) == 148, f"Expected 148 BigCodeBench-Hard items, got {len(items)}"
    it0 = items[0]
    neg = evaluate_bcb_solution(it0, f"def {it0['entry_point']}(*args, **kwargs):\n    return None")
    assert not neg["passed"], f"Smoke negative control unexpectedly passed: {neg}"
    canon_code = f"{it0['code_prompt'].rstrip()}\n{it0['canonical_solution']}"
    pos = evaluate_bcb_solution(it0, canon_code)
    assert pos["passed"], f"Smoke canonical positive control failed on {it0['task_id']}: {pos}"
    print(f"[BCB-Hard Smoke Gate] PASSED | {it0['task_id']} neg={neg['failure_type']} canon=True | n={len(items)}", flush=True)


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


def summarize_bcb_report(
    progress_jsonl: str,
    out_json: str,
    total_target: int = 148,
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
                    r = json.loads(s)
                    recs_by_id[str(r["task_id"])] = r
    recs = list(recs_by_id.values())
    n = len(recs)
    p = sum(1 for r in recs if r.get("passed"))
    lo, hi = wilson_ci_95(p, n)
    failures: dict[str, int] = {}
    for r in recs:
        ft = str(r.get("failure_type", "unknown"))
        failures[ft] = failures.get(ft, 0) + 1
    rep = {
        "benchmark": "bigcode/bigcodebench-hard (v0.1.4 Instruct)",
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
        "routing": _agg_routing(recs),
        "failure_breakdown": failures,
    }
    tmp = out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_json)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="BigCodeBench-Hard (v0.1.4 n=148) Evaluator for Chronos v8")
    ap.add_argument("--questions-json", default="/content/dsv41_v8_bcb_hard_questions.json")
    ap.add_argument("--progress-jsonl", default="/content/v8_bigcodebench_hard_progress.jsonl")
    ap.add_argument("--out-json", default="/content/dsv41_v8_bigcodebench_hard_report.json")
    ap.add_argument("--max-new-tokens", type=int, default=6144)
    ap.add_argument("--think-ceiling", type=int, default=3072)
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--use-speculative", action="store_true")
    ap.add_argument("--max-wall-s", type=float, default=1300.0)
    ap.add_argument("--smoke-gate-only", action="store_true")
    args = ap.parse_args()

    _check_not_protected(args.progress_jsonl)
    _check_not_protected(args.out_json)

    for p in ("/content", "/content/chronos_1t", "/content/scripts", "/content/benchmarks", "/content/dsv41_standalone/inference", "/content/dsv41_standalone/encoding"):
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)

    items = build_or_load_bcb_questions(out_json=args.questions_json)
    run_bcb_verifier_smoke_gate(args.questions_json)
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

    eff_raw = str(args.reasoning_effort).strip()
    eff_val = int(eff_raw) if eff_raw.isdigit() else eff_raw

    eval_settings = {
        "target_n": len(items),
        "max_new_tokens": int(args.max_new_tokens),
        "think_ceiling": int(args.think_ceiling),
        "reasoning_effort": eff_val,
        "ablate_guards": True,
        "temperature": 0.0,
        "use_speculative": bool(args.use_speculative),
        "max_seq_len": int(model.layers[0].attn.freqs_cis.size(0)),
    }

    done_ids: set[str] = set()
    passed_cnt = 0
    if os.path.isfile(args.progress_jsonl):
        seen: dict[str, bool] = {}
        with open(args.progress_jsonl, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if s:
                    r = json.loads(s)
                    seen[str(r["task_id"])] = bool(r.get("passed"))
        done_ids = set(seen.keys())
        passed_cnt = sum(1 for v in seen.values() if v)
    done_cnt = len(done_ids)

    os.environ["CHRONOS_ABLATE_GUARDS"] = "1"
    os.environ["CHRONOS_THINK_CEILING"] = str(args.think_ceiling)
    t0 = time.perf_counter()
    decode_checked = False

    print(f"\n=== [v8:BigCodeBench-Hard] Target={len(items)} done={done_cnt} | max_new={args.max_new_tokens} ceil={args.think_ceiling} eff={eff_val} ===", flush=True)
    for item in items:
        tid = str(item["task_id"])
        if tid in done_ids:
            continue
        if args.max_wall_s > 0 and (time.perf_counter() - t0) >= args.max_wall_s:
            print(f"\n[v8:BCB-Hard] Wall budget {args.max_wall_s:.0f}s reached at idx={item['idx']}; stopping cleanly.", flush=True)
            break

        prompt = format_bcb_instruct_prompt(item)
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
            reasoning_effort=eff_val,
            use_speculative=bool(args.use_speculative),
            log_routing=True,
            temperature=0.0,
        )
        if not res.get("logits_finite", True):
            raise RuntimeError(f"Non-finite logits on BCB-Hard {tid}!")
        if (not decode_checked) and int(res.get("generated_tokens", 0)) > 1:
            assert int(res.get("decode_nonzero_rows", 0)) > 0, f"decode_nonzero_rows == 0 on BCB-Hard {tid}!"
            decode_checked = True

        raw_ans = res.get("answer_text") or ""
        raw_full = res.get("text") or ""
        code, code_unclosed = extract_python_code_with_status(
            raw_ans, entry_point=item["entry_point"], code_prompt=item["code_prompt"]
        )
        code_from_fallback = False
        if not code:
            code, code_unclosed = extract_python_code_with_status(
                raw_full, entry_point=item["entry_point"], code_prompt=item["code_prompt"]
            )
            code_from_fallback = bool(code)
        ev = evaluate_bcb_solution(item, code)
        if (not ev["passed"]) and ev["failure_type"] == "syntax_error" and (
            code_unclosed or int(res.get("generated_tokens", 0)) >= int(args.max_new_tokens)
        ):
            ev["failure_type"] = "truncated_code"

        rec = {
            "model": "chronos_v8_calibrated_3tier",
            "task_id": tid,
            "idx": int(item["idx"]),
            "passed": bool(ev["passed"]),
            "failure_type": str(ev["failure_type"]),
            "detail": str(ev["detail"])[:240],
            "tokens": int(res.get("generated_tokens", 0)),
            "think_tokens": int(res.get("think_tokens", 0)),
            "answer_tokens": int(res.get("answer_tokens", 0)),
            "prompt_tokens": int(res.get("prompt_tokens", 0)),
            "think_forced": bool(res.get("think_forced", False)),
            "think_unclosed": bool(res.get("think_unclosed", False)),
            "use_speculative": bool(args.use_speculative),
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
        }
        with open(args.progress_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done_ids.add(tid)
        done_cnt += 1
        if rec["passed"]:
            passed_cnt += 1
        pct = 100.0 * passed_cnt / max(1, done_cnt)
        print(
            f"  [v8:bcb {done_cnt:3d}/{len(items):3d}] Pass={passed_cnt}/{done_cnt} ({pct:5.1f}%) | "
            f"{tid} ok={rec['passed']} ({rec['failure_type']}) tok={rec['tokens']}(th={rec['think_tokens']}) "
            f"@ {rec['tok_s']:.1f} tok/s | 4b(pf={100*rec['prefill_4bit_fraction']:.1f}%,dec={100*rec['decode_4bit_fraction']:.1f}%)",
            flush=True,
        )
        if done_cnt % 10 == 0:
            summarize_bcb_report(args.progress_jsonl, args.out_json, total_target=len(items), settings=eval_settings)

    rep = summarize_bcb_report(args.progress_jsonl, args.out_json, total_target=len(items), settings=eval_settings)
    print(f"\n[v8:BCB-Hard] Saved {args.out_json} | status={rep['status']} Pass@1={rep['passed']}/{rep['n']} ({rep['pass_at_1_pct']}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
