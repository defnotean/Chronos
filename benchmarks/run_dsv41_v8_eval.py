#!/usr/bin/env python3
"""
Controlled 600-Question Intelligence & 3-Tier Hit-Rate Evaluation for Chronos v8 (`DeepSeek-V4.1-Flash-1017`).

Matches the exact fixed 600-question set and `A_unleashed` decoding configuration
(guards off, think_ceiling=3072, kv_limit=8192, clean GSM8K prompt & parser) while
logging live 3-tier routing fractions (prefill & decode T1/T2/T3) and tagging the
strictly unseen `Val-120` held-out questions (`idx % 5 == 0`).
"""
from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import __main__
import torch

_PROTECTED_FILES = {
    "dsv41_full_long_benchmark_progress.jsonl",
    "dsv41_official_benchmarks_report.json",
    "run_dsv41_full_long_benchmarks.py",
}

_SUITE_CFG = {
    "mmlu": {"thinking_mode": "thinking", "max_new_tokens": 2048, "max_think_tokens": 0},
    "mmlu_pro": {"thinking_mode": "thinking", "max_new_tokens": 4096, "max_think_tokens": 0},
    "gsm8k": {"thinking_mode": "thinking", "max_new_tokens": 4096, "max_think_tokens": 0},
}


def _check_not_protected(path_str: str) -> None:
    if os.path.basename(os.path.abspath(path_str)) in _PROTECTED_FILES:
        raise ValueError(f"Refusing to write to protected file: {path_str}")


def _attach_v8_resident() -> tuple[Any, Any, Any, Any]:
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

    import chronos_1t.models.dsv41_longctx_spec_engine as spec_eng
    import scripts.run_40layer_chimera_dsv41_a100 as runner_mod

    model = getattr(builtins, "GLOBAL_DSV41_MODEL", None) or getattr(__main__, "GLOBAL_DSV41_MODEL", None)
    tokenizer = getattr(builtins, "GLOBAL_DSV41_TOKENIZER", None) or getattr(__main__, "GLOBAL_DSV41_TOKENIZER", None)
    if model is None or tokenizer is None:
        raise RuntimeError("GLOBAL_DSV41_MODEL not resident! Run scripts/load_dsv41_v8_calibrated_3tier.py first.")
    ver = getattr(builtins, "GLOBAL_DSV41_VERSION", "")
    if ver != "v8_calibrated_3tier":
        raise RuntimeError(f"Expected resident v8_calibrated_3tier model, got version={ver!r}")

    if not getattr(model, "_chronos_v8_eval_ready", False):
        spec_eng.install_general_chunked_attention(runner_mod.dsv41_model_mod)
        model._chronos_v8_eval_ready = True
    return model, tokenizer, spec_eng, runner_mod


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


def summarize_v8_report(progress_jsonl: str, out_json: str, elapsed_s: float = 0.0) -> dict[str, Any]:
    _check_not_protected(progress_jsonl)
    _check_not_protected(out_json)
    import benchmarks.run_dsv41_controlled_ablations as abl

    done = abl.base_bench._load_completed_records(progress_jsonl)
    by_suite: dict[str, list[dict[str, Any]]] = {"mmlu": [], "mmlu_pro": [], "gsm8k": []}
    for (s_k, _), rec in done.items():
        if s_k in by_suite:
            by_suite[s_k].append(rec)

    def _build_split_summary(subset_filter: Any = None) -> dict[str, Any]:
        suite_stats: dict[str, Any] = {}
        all_recs: list[dict[str, Any]] = []
        for s_k in ("mmlu", "mmlu_pro", "gsm8k"):
            recs = [r for r in by_suite[s_k] if (subset_filter is None or subset_filter(r))]
            n = len(recs)
            p = sum(1 for r in recs if r.get("passed"))
            lo, hi = abl.wilson_ci_95(p, n)
            suite_stats[s_k] = {
                "n": n,
                "passed": p,
                "pass_rate_pct": round(100.0 * p / max(1, n), 2),
                "wilson_ci_95_pct": [lo, hi],
                "mean_tokens": round(sum(int(r.get("tokens", 0)) for r in recs) / max(1, n), 2),
                "mean_think_tokens": round(sum(int(r.get("think_tokens", 0)) for r in recs) / max(1, n), 2),
                "think_forced_pct": round(100.0 * sum(1 for r in recs if r.get("think_forced")) / max(1, n), 2),
                "think_unclosed_pct": round(100.0 * sum(1 for r in recs if r.get("think_unclosed")) / max(1, n), 2),
                "mean_decode_tok_s": round(sum(float(r.get("tok_s", 0.0)) for r in recs) / max(1, n), 2),
                "routing": _agg_routing(recs),
            }
            all_recs.extend(recs)
        tot_n = len(all_recs)
        tot_p = sum(1 for r in all_recs if r.get("passed"))
        ov_lo, ov_hi = abl.wilson_ci_95(tot_p, tot_n)
        macro = round(
            (suite_stats["mmlu"]["pass_rate_pct"] + suite_stats["mmlu_pro"]["pass_rate_pct"] + suite_stats["gsm8k"]["pass_rate_pct"]) / 3.0,
            2,
        )
        return {
            "total_n": tot_n,
            "total_passed": tot_p,
            "micro_pass_rate_pct": round(100.0 * tot_p / max(1, tot_n), 2),
            "micro_wilson_ci_95_pct": [ov_lo, ov_hi],
            "macro_pass_rate_pct": macro,
            "mean_tokens": round(sum(int(r.get("tokens", 0)) for r in all_recs) / max(1, tot_n), 2),
            "mean_think_tokens": round(sum(int(r.get("think_tokens", 0)) for r in all_recs) / max(1, tot_n), 2),
            "think_forced_pct": round(100.0 * sum(1 for r in all_recs if r.get("think_forced")) / max(1, tot_n), 2),
            "routing": _agg_routing(all_recs),
            "suites": suite_stats,
        }

    all_sum = _build_split_summary(None)
    val_sum = _build_split_summary(lambda r: bool(r.get("is_heldout_val", False)))
    trn_sum = _build_split_summary(lambda r: not bool(r.get("is_heldout_val", False)))

    report = {
        "model_version": "chronos_v8_calibrated_3tier",
        "source_repo": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "abliterated": False,
        "status": "COMPLETE" if (all_sum["suites"]["mmlu"]["n"] == 300 and all_sum["suites"]["mmlu_pro"]["n"] == 150 and all_sum["suites"]["gsm8k"]["n"] == 150) else "IN_PROGRESS",
        "elapsed_s": round(elapsed_s, 2),
        "all_600": all_sum,
        "heldout_val_120": val_sum,
        "calib_train_480": trn_sum,
    }
    tmp = out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_json)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Chronos v8 600-Question Intelligence Benchmark (1:1 A_unleashed)")
    ap.add_argument("--suite", default="gsm8k")
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=0, help="Exclusive end index per suite (0 = full suite)")
    ap.add_argument("--questions-json", default="/content/ablate_fixed_600_questions.json")
    ap.add_argument("--progress-jsonl", default="/content/v8_eval_progress.jsonl")
    ap.add_argument("--out-json", default="/content/dsv41_v8_intelligence_report.json")
    ap.add_argument("--think-ceiling", type=int, default=3072)
    ap.add_argument("--kv-limit", type=int, default=8192)
    ap.add_argument("--max-wall-s", type=float, default=1350.0)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    _check_not_protected(args.progress_jsonl)
    _check_not_protected(args.out_json)

    for p in ("/content", "/content/scripts", "/content/benchmarks"):
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)

    import benchmarks.run_dsv41_controlled_ablations as abl

    if args.summarize_only:
        rep = summarize_v8_report(args.progress_jsonl, args.out_json)
        print(json.dumps(rep["all_600"], indent=2), flush=True)
        return 0

    model, tokenizer, spec_eng, runner_mod = _attach_v8_resident()
    q_bank = abl.build_or_load_fixed_600_questions(questions_json=args.questions_json)
    done_map = abl.base_bench._load_completed_records(args.progress_jsonl)
    os.environ["CHRONOS_ABLATE_GUARDS"] = "1"
    os.environ["CHRONOS_THINK_CEILING"] = str(args.think_ceiling)
    t0 = time.perf_counter()
    decode_checked = False

    for s_key in [s.strip().lower() for s in args.suite.split(",") if s.strip()]:
        items = q_bank[s_key]
        s_i = max(0, args.start_idx)
        e_i = len(items) if args.end_idx <= 0 else min(len(items), args.end_idx)
        s_cfg = _SUITE_CFG[s_key]
        t_mode = str(s_cfg["thinking_mode"])
        max_new = int(s_cfg["max_new_tokens"])
        max_think = int(s_cfg["max_think_tokens"])
        grader = abl._grade_gsm8k_clean if s_key == "gsm8k" else (abl._grade_mmlu_pro if s_key == "mmlu_pro" else abl._grade_mmlu)

        print(f"\n=== [v8:{s_key}] Running {s_i}..{e_i-1}/{len(items)} | max_new={max_new} ceil={args.think_ceiling} kv={args.kv_limit} ===", flush=True)
        done_cnt = sum(1 for it in items if (s_key, str(it["task_id"])) in done_map)
        pass_cnt = sum(1 for it in items if (s_key, str(it["task_id"])) in done_map and done_map[(s_key, str(it["task_id"]))].get("passed"))

        for idx_i in range(s_i, e_i):
            it = items[idx_i]
            tid = str(it["task_id"])
            if (s_key, tid) in done_map:
                continue
            if args.max_wall_s > 0 and (time.perf_counter() - t0) >= args.max_wall_s:
                print(f"\n[v8:{s_key}] Wall budget {args.max_wall_s:.0f}s reached at idx={idx_i}; stopping slice cleanly.", flush=True)
                break

            item_prompt = it["prompt"]
            if s_key == "gsm8k":
                item_prompt = item_prompt.replace(
                    r"end with \boxed{<number>}:",
                    r"put your final numerical answer inside \boxed{} (with no units):",
                ).replace(
                    r"\boxed{<number>}",
                    r"\boxed{}",
                )
            safe_prompt = abl.base_bench._fit_prompt_to_kv_budget(
                tokenizer, runner_mod._encode_prompt_safe, item_prompt, t_mode, max_new, kv_limit=args.kv_limit
            )
            res = spec_eng.generate_with_speculative_and_prefix(
                model,
                tokenizer,
                runner_mod.dsv41_model_mod,
                runner_mod,
                [{"role": "user", "content": safe_prompt}],
                max_new_tokens=max_new,
                max_think_tokens=max_think,
                think_ceiling=args.think_ceiling,
                ablate_guards=True,
                thinking_mode=t_mode,
                use_speculative=False,
                log_routing=True,
            )
            if not res.get("logits_finite", True):
                raise RuntimeError(f"Non-finite logits detected on {s_key} idx={idx_i} {tid}!")
            if (not decode_checked) and int(res.get("generated_tokens", 0)) > 1:
                assert int(res.get("decode_nonzero_rows", 0)) > 0, (
                    f"Decode routing telemetry rows were all zero on {s_key} idx={idx_i}!"
                )
                decode_checked = True

            ok, meta = grader(it, res)
            is_val = (int(idx_i) % 5) == 0
            rec = {
                "model": "chronos_v8_calibrated_3tier",
                "suite": s_key,
                "idx": idx_i,
                "task_id": tid,
                "group": it.get("group", ""),
                "is_heldout_val": is_val,
                "thinking_mode": t_mode,
                "passed": bool(ok),
                "tokens": int(res.get("generated_tokens", 0)),
                "think_tokens": int(res.get("think_tokens", 0)),
                "answer_tokens": int(res.get("answer_tokens", 0)),
                "prompt_tokens": int(res.get("prompt_tokens", 0)),
                "think_forced": bool(res.get("think_forced", False)),
                "think_unclosed": bool(res.get("think_unclosed", False)),
                "think_close_reason": str(res.get("think_close_reason", "")),
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
                "all_4bit_fraction": round(float(res.get("all_4bit_frac", 0.0)), 6),
                "detail": str(meta)[:180],
                "answer_preview": str(res.get("answer_text") or "")[:120],
            }
            abl.base_bench._append_record(args.progress_jsonl, rec)
            done_map[(s_key, tid)] = rec
            done_cnt += 1
            if ok:
                pass_cnt += 1
            pct = 100.0 * pass_cnt / max(1, done_cnt)
            print(
                f"  [v8:{s_key} {done_cnt:3d}/{len(items):3d}] Pass={pass_cnt}/{done_cnt} ({pct:5.1f}%) | "
                f"idx={idx_i:3d}{'*V' if is_val else '  '} ok={ok} tok={rec['tokens']}(th={rec['think_tokens']}) "
                f"@ {rec['tok_s']:.1f} tok/s | 4b(pf={100*rec['prefill_4bit_fraction']:.1f}%,"
                f"dec={100*rec['decode_4bit_fraction']:.1f}%) | {rec['detail']}",
                flush=True,
            )
            if done_cnt % 10 == 0:
                summarize_v8_report(args.progress_jsonl, args.out_json, time.perf_counter() - t0)

    rep = summarize_v8_report(args.progress_jsonl, args.out_json, time.perf_counter() - t0)
    print(f"\n[v8] Saved report {args.out_json} | all_n={rep['all_600']['total_n']} micro={rep['all_600']['micro_pass_rate_pct']}% val_n={rep['heldout_val_120']['total_n']} val_micro={rep['heldout_val_120']['micro_pass_rate_pct']}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
