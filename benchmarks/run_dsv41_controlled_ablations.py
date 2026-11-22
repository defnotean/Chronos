#!/usr/bin/env python3
"""
Controlled Single-Variable Ablation Runner (A0..A5) for Chronos DeepSeek-V4.1-Flash-1017.

Evaluates fixed stratified question sets matching Qwen Flash Next sample sizes:
  - MMLU:     n=300 stratified by subject
  - MMLU-Pro: n=150 stratified by category
  - GSM8K:    n=150

Strictly isolated from `results/dsv41_full_long_benchmark_progress.jsonl` and
`results/dsv41_official_benchmarks_report.json`.
"""
from __future__ import annotations

import argparse
import builtins
import json
import math
import os
import re
import sys
import time
import types
from pathlib import Path
from typing import Any

import __main__
import torch

for _p in (
    "/content",
    "/content/scripts",
    "/content/benchmarks",
    "/content/dsv41_standalone/inference",
    "/content/dsv41_standalone/encoding",
):
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import benchmarks.run_dsv41_full_long_benchmarks as base_bench

DEFAULT_QUESTIONS_JSON = "/content/ablate_fixed_600_questions.json"
DEFAULT_CACHE_DIR = base_bench.DEFAULT_CACHE_DIR

ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "A0": {
        "description": "Baseline (current settings untouched)",
        "kv_limit": 2048,
        "ablate_guards": "0",
        "think_ceiling": "",
        "log_routing": "0",
        "suites": {
            "mmlu": {"thinking_mode": "chat", "max_new_tokens": 12, "max_think_tokens": 0},
            "mmlu_pro": {"thinking_mode": "thinking", "max_new_tokens": 768, "max_think_tokens": 0},
            "gsm8k": {"thinking_mode": "thinking", "max_new_tokens": 768, "max_think_tokens": 0},
        },
    },
    "A1": {
        "description": "No think caps (max_think_tokens=0, mmlu_pro/gsm8k=2048, mmlu thinking=1024, 384 safety ceiling kept, routing logged)",
        "kv_limit": 4096,
        "ablate_guards": "0",
        "think_ceiling": "",
        "log_routing": "1",
        "suites": {
            "mmlu": {"thinking_mode": "thinking", "max_new_tokens": 1024, "max_think_tokens": 0},
            "mmlu_pro": {"thinking_mode": "thinking", "max_new_tokens": 2048, "max_think_tokens": 0},
            "gsm8k": {"thinking_mode": "thinking", "max_new_tokens": 2048, "max_think_tokens": 0},
        },
    },
    "A1_ceil1536": {
        "description": "A1 variant with <think> safety ceiling raised from 384 to 1536",
        "kv_limit": 4096,
        "ablate_guards": "0",
        "think_ceiling": "1536",
        "log_routing": "0",
        "suites": {
            "mmlu": {"thinking_mode": "thinking", "max_new_tokens": 1024, "max_think_tokens": 0},
            "mmlu_pro": {"thinking_mode": "thinking", "max_new_tokens": 2048, "max_think_tokens": 0},
            "gsm8k": {"thinking_mode": "thinking", "max_new_tokens": 2048, "max_think_tokens": 0},
        },
    },
    "A2": {
        "description": "Decoding guards off inside <think> on top of A0 baseline (CHRONOS_ABLATE_GUARDS=1)",
        "kv_limit": 2048,
        "ablate_guards": "1",
        "think_ceiling": "",
        "log_routing": "0",
        "suites": {
            "mmlu": {"thinking_mode": "chat", "max_new_tokens": 12, "max_think_tokens": 0},
            "mmlu_pro": {"thinking_mode": "thinking", "max_new_tokens": 768, "max_think_tokens": 0},
            "gsm8k": {"thinking_mode": "thinking", "max_new_tokens": 768, "max_think_tokens": 0},
        },
    },
    "A_unleashed": {
        "description": "Unleashed DSV4.1: CHRONOS_ABLATE_GUARDS=1, think_ceiling=3072, max_new_tokens=4096 (2048 MMLU), clean GSM8K prompt & parser",
        "kv_limit": 8192,
        "ablate_guards": "1",
        "think_ceiling": "3072",
        "log_routing": "0",
        "suites": {
            "mmlu": {"thinking_mode": "thinking", "max_new_tokens": 2048, "max_think_tokens": 0},
            "mmlu_pro": {"thinking_mode": "thinking", "max_new_tokens": 4096, "max_think_tokens": 0},
            "gsm8k": {"thinking_mode": "thinking", "max_new_tokens": 4096, "max_think_tokens": 0},
        },
    },
}


def wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    """Return 95% Wilson score confidence interval in percentage points."""
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / float(n)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return (round(max(0.0, 100.0 * (center - half)), 2), round(min(100.0, 100.0 * (center + half)), 2))


def get_resident_model_for_ablation() -> tuple[Any, Any, Any, Any]:
    """Attach to resident GLOBAL_DSV41_MODEL without reloading weights or invalidating CUDAGraphs."""
    srv = getattr(builtins, "_CHRONOS_HTTP_SERVER", None)
    if srv is not None:
        try:
            srv.shutdown()
            srv.server_close()
        except Exception:
            pass
        delattr(builtins, "_CHRONOS_HTTP_SERVER")

    for k in list(sys.modules.keys()):
        if k == "run_40layer_chimera_dsv41_a100" or k.startswith("scripts.run_40layer_chimera_dsv41_a100"):
            sys.modules.pop(k, None)

    import chronos_1t.models.dsv41_chronos_moe as moe_mod
    import scripts.run_40layer_chimera_dsv41_a100 as runner_mod

    model = getattr(builtins, "GLOBAL_DSV41_MODEL", None) or getattr(__main__, "GLOBAL_DSV41_MODEL", None)
    tokenizer = getattr(builtins, "GLOBAL_DSV41_TOKENIZER", None) or getattr(__main__, "GLOBAL_DSV41_TOKENIZER", None)
    if model is None or tokenizer is None:
        raise RuntimeError(
            "GLOBAL_DSV41_MODEL is not resident in memory. Run load_and_optimize.py once first."
        )

    patched = 0
    target_fwd = moe_mod.ChronosDsv41SparseMoeBlock.forward
    for layer in getattr(model, "layers", []):
        ffn = getattr(layer, "ffn", None)
        if ffn is not None and hasattr(ffn, "w1_gu_u8"):
            cur_fwd = getattr(getattr(ffn, "forward", None), "__func__", None)
            if cur_fwd is not target_fwd:
                ffn.forward = types.MethodType(target_fwd, ffn)
                patched += 1
    if patched > 0:
        cg_recaptured = runner_mod.warmup_40layer_cudagraphs(model)
        print(f"[Ablation] Hot-patched {patched} MoE layers & recaptured {cg_recaptured}/40 CUDAGraphs.", flush=True)
    else:
        print("[Ablation] Attached to resident 40-layer model (existing CUDAGraphs preserved).", flush=True)
    builtins.GLOBAL_DSV41_GENERATE = runner_mod.generate_chat_greedy
    return model, tokenizer, runner_mod.generate_chat_greedy, runner_mod._encode_prompt_safe


def build_or_load_fixed_600_questions(
    questions_json: str = DEFAULT_QUESTIONS_JSON,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> dict[str, list[dict[str, Any]]]:
    """Create or load the exact fixed 600-question set (MMLU 300, MMLU-Pro 150, GSM8K 150)."""
    if os.path.isfile(questions_json) and os.path.getsize(questions_json) > 1024:
        with open(questions_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if (
            len(data.get("mmlu", [])) == 300
            and len(data.get("mmlu_pro", [])) == 150
            and len(data.get("gsm8k", [])) == 150
        ):
            return data

    letters4 = "ABCD"
    all_mmlu = base_bench._fetch_hf_rows("cais/mmlu", "all", "test", cache_dir, 0)
    mmlu_sampled = base_bench._stratified_sample(all_mmlu, 300, group_key="subject")
    mmlu_items: list[dict[str, Any]] = []
    for i, it in enumerate(mmlu_sampled):
        opts = "\n".join(f"{letters4[j]}) {c}" for j, c in enumerate(it["choices"]))
        mmlu_items.append(
            {
                "suite": "mmlu",
                "idx": i,
                "task_id": f"mmlu_{i:03d}_{it.get('subject', 'stem')}",
                "group": str(it.get("subject") or "unknown"),
                "prompt": f"Subject: {it['subject']}\n{it['question']}\n{opts}\nReply ONLY with the single letter A, B, C, or D.",
                "gold": letters4[int(it["answer"]) % 4],
            }
        )

    letters10 = "ABCDEFGHIJ"
    mp_cache = os.path.join(cache_dir, "TIGER-Lab__MMLU-Pro__default__test.jsonl")
    if os.path.isfile(mp_cache):
        with open(mp_cache, "r", encoding="utf-8") as mf:
            if sum(1 for ln in mf if ln.strip()) < 2000:
                os.remove(mp_cache)
    all_mp = base_bench._fetch_hf_rows("TIGER-Lab/MMLU-Pro", "default", "test", cache_dir, 0)
    mp_sampled = base_bench._stratified_sample(all_mp, 150, group_key="category")
    mp_items: list[dict[str, Any]] = []
    for i, it in enumerate(mp_sampled):
        opts = "\n".join(f"{letters10[j]}) {c}" for j, c in enumerate(it["options"][:10]))
        mp_items.append(
            {
                "suite": "mmlu_pro",
                "idx": i,
                "task_id": f"mpro_{it['question_id']}",
                "group": str(it.get("category") or "unknown"),
                "prompt": f"Category: {it['category']}\n{it['question']}\n{opts}\nReason concisely and state final answer letter as ANSWER: <letter>.",
                "gold": str(it["answer"]).strip().upper(),
            }
        )

    gsm_all = base_bench._fetch_hf_rows("openai/gsm8k", "main", "test", cache_dir, 0)
    gsm_sampled = base_bench._stratified_sample(gsm_all, 150)
    gsm_items: list[dict[str, Any]] = []
    for i, it in enumerate(gsm_sampled):
        gsm_items.append(
            {
                "suite": "gsm8k",
                "idx": i,
                "task_id": f"gsm8k_{i:03d}",
                "group": "gsm8k",
                "prompt": f"Solve step by step and end with \\boxed{{<number>}}:\n{it['question']}",
                "gold": str(it["answer"]).split("####")[-1].strip().replace(",", ""),
            }
        )

    out = {"mmlu": mmlu_items, "mmlu_pro": mp_items, "gsm8k": gsm_items}
    os.makedirs(os.path.dirname(os.path.abspath(questions_json)), exist_ok=True)
    with open(questions_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(
        f"[Questions] Froze fixed dataset at {questions_json}: "
        f"mmlu={len(mmlu_items)} ({len({x['group'] for x in mmlu_items})} subjects), "
        f"mmlu_pro={len(mp_items)} ({len({x['group'] for x in mp_items})} categories), "
        f"gsm8k={len(gsm_items)}.",
        flush=True,
    )
    return out


def _grade_mmlu(item: dict[str, Any], res: dict[str, Any]) -> tuple[bool, str]:
    gold = str(item["gold"]).strip().upper()
    ans = (res.get("answer_text") or res.get("text") or "").strip()
    if "</think>" in ans:
        ans = ans.split("</think>", 1)[1].strip()
    m_exp = re.findall(r"(?:ANSWER|final)\s*[:\-]?\s*\**([A-D])\**", ans, re.IGNORECASE)
    if m_exp:
        pred = m_exp[-1].upper()
    else:
        boxed = base_bench._find_last_boxed(ans)
        if boxed and boxed.strip().upper()[:1] in "ABCD":
            pred = boxed.strip().upper()[:1]
        else:
            letters = re.findall(r"\b([A-D])\b", ans.upper())
            pred = letters[-1] if letters else (ans[:1].upper() if ans[:1].upper() in "ABCD" else "?")
    return pred == gold, f"pred={pred},gold={gold}"


def _grade_mmlu_pro(item: dict[str, Any], res: dict[str, Any]) -> tuple[bool, str]:
    gold = str(item["gold"]).strip().upper()
    ans = (res.get("answer_text") or res.get("text") or "").strip()
    if "</think>" in ans:
        ans = ans.split("</think>", 1)[1].strip()
    m = re.findall(r"(?:ANSWER|final)\s*[:\-]?\s*\**([A-J])\**", ans, re.IGNORECASE)
    if m:
        pred = m[-1].upper()
    else:
        boxed = base_bench._find_last_boxed(ans)
        if boxed and boxed.strip().upper()[:1] in "ABCDEFGHIJ":
            pred = boxed.strip().upper()[:1]
        else:
            pred = (re.findall(r"\b([A-J])\b", ans.upper()) or ["?"])[-1]
    return pred == gold, f"pred={pred},gold={gold}"


def _grade_gsm8k(item: dict[str, Any], res: dict[str, Any]) -> tuple[bool, str]:
    gold = str(item["gold"]).strip()
    pred = base_bench._extract_boxed_or_last_answer(res.get("text") or "")
    ok = base_bench._math_answers_equal(pred, gold)
    return ok, f"pred={pred!r},gold={gold!r}"


def _clean_gsm8k_pred_text(raw: str) -> str:
    import re as _re

    s = str(raw or "").strip()
    s = _re.sub(r"\\boxed\{\s*(?:<number>|\?)?\s*\}", "", s, flags=_re.IGNORECASE)
    pred = base_bench._extract_boxed_or_last_answer(s).strip()
    if pred.lower() in ("<number>", "number", ""):
        return ""
    for _ in range(4):
        new_s = _re.sub(
            r"\\(?:text|mathrm|mathbf|textbf|operatorname|mbox)\{([^{}]*)\}",
            r"\1",
            pred,
        )
        if new_s == pred:
            break
        pred = new_s
    pred = pred.replace(r"\%", "").replace("%", "").replace(r"\$", "").replace("$", "")
    pred = pred.replace(r"\left", "").replace(r"\right", "").strip()
    unit_pat = (
        r"\s*\b(?:hours?|hrs?|dollars?|minutes?|mins?|seconds?|secs?|miles?|years?|yrs?|days?|"
        r"weeks?|months?|feet|foot|ft|meters?|cm|km|grams?|kg|pounds?|lbs?|percent|percentage|"
        r"eggs?|cups?|pages?|students?|apples?|books?|people|persons?|items?|pieces?|times?|"
        r"each|total|per\s+\w+)\b\.?$"
    )
    for _ in range(3):
        stripped = _re.sub(unit_pat, "", pred, flags=_re.IGNORECASE).strip()
        if stripped == pred:
            break
        pred = stripped
    pred = pred.rstrip(".").strip()
    m_num = _re.fullmatch(r"(-?\d[\d,]*(?:\.\d+)?)\s*(?:units?)?", pred, flags=_re.IGNORECASE)
    if m_num:
        pred = m_num.group(1).replace(",", "")
    return pred


def _grade_gsm8k_clean(item: dict[str, Any], res: dict[str, Any]) -> tuple[bool, str]:
    gold = str(item["gold"]).strip()
    full_text = res.get("text") or ""
    ans_src = (res.get("answer_text") or "").strip()
    if not ans_src and "</think>" in full_text:
        ans_src = full_text.split("</think>")[-1].strip()
    candidates = []
    if ans_src:
        candidates.append(ans_src)
    candidates.append(full_text)
    pred = ""
    for src in candidates:
        cand = _clean_gsm8k_pred_text(src)
        if cand:
            pred = cand
            break
    if not pred:
        pred = base_bench._extract_boxed_or_last_answer(full_text)
    ok = base_bench._math_answers_equal(pred, gold)
    return ok, f"pred={pred!r},gold={gold!r}"


def _get_unleashed_engine(model: Any, gen_fn: Any) -> tuple[Any, Any]:
    import os as _os
    import shutil as _shutil
    import sys as _sys
    import types as _types

    for _p in ("/content", "/content/scripts", "/content/benchmarks", "/content/dsv41_standalone/inference", "/content/dsv41_standalone/encoding"):
        if _os.path.exists(_p) and _p not in _sys.path:
            _sys.path.insert(0, _p)
    for _fn in ("dsv41_chronos_moe.py", "dsv41_longctx_spec_engine.py"):
        _src = f"/content/chronos_1t/models/{_fn}"
        _dst = f"/content/{_fn}"
        if _os.path.isfile(_src) and (not _os.path.isfile(_dst) or _os.path.getmtime(_src) > _os.path.getmtime(_dst)):
            _shutil.copy2(_src, _dst)

    import chronos_1t.models.dsv41_chronos_moe as moe_mod
    import chronos_1t.models.dsv41_longctx_spec_engine as spec_eng

    runner_mod = _sys.modules[gen_fn.__module__]
    if not getattr(model, "_chronos_unleashed_ready", False):
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
        spec_eng.upgrade_resident_model_long_context(model, runner_mod.dsv41_model_mod, runner_mod, target_max_seq_len=49152)
        spec_eng.install_general_chunked_attention(runner_mod.dsv41_model_mod)
        model._chronos_unleashed_ready = True
    return spec_eng, runner_mod


_GRADERS = {
    "mmlu": _grade_mmlu,
    "mmlu_pro": _grade_mmlu_pro,
    "gsm8k": _grade_gsm8k,
}


def apply_arm_env(arm_name: str) -> dict[str, Any]:
    cfg = ARM_CONFIGS[arm_name]
    os.environ["CHRONOS_ABLATE_GUARDS"] = str(cfg["ablate_guards"])
    os.environ["CHRONOS_LOG_ROUTING"] = str(cfg["log_routing"])
    if cfg["think_ceiling"]:
        os.environ["CHRONOS_THINK_CEILING"] = str(cfg["think_ceiling"])
    else:
        os.environ.pop("CHRONOS_THINK_CEILING", None)
    return cfg


def summarize_arm_report(
    arm_name: str,
    progress_jsonl: str,
    out_json: str,
    elapsed_s: float = 0.0,
) -> dict[str, Any]:
    done = base_bench._load_completed_records(progress_jsonl)
    by_suite: dict[str, list[dict[str, Any]]] = {"mmlu": [], "mmlu_pro": [], "gsm8k": []}
    for (s_k, _), rec in done.items():
        if s_k in by_suite:
            by_suite[s_k].append(rec)

    suite_stats: dict[str, Any] = {}
    tot_n = 0
    tot_p = 0
    for s_k in ("mmlu", "mmlu_pro", "gsm8k"):
        recs = by_suite[s_k]
        n = len(recs)
        p = sum(1 for r in recs if r.get("passed"))
        toks = sum(int(r.get("tokens", 0)) for r in recs)
        th_toks = sum(int(r.get("think_tokens", 0)) for r in recs)
        n_forced = sum(1 for r in recs if r.get("think_forced"))
        n_unclosed = sum(1 for r in recs if r.get("think_unclosed"))
        n_think_mode = sum(1 for r in recs if r.get("thinking_mode") == "thinking")
        ci_lo, ci_hi = wilson_ci_95(p, n)
        suite_stats[s_k] = {
            "n": n,
            "passed": p,
            "pass_rate_pct": round(100.0 * p / max(1, n), 2),
            "wilson_ci_95_pct": [ci_lo, ci_hi],
            "mean_tokens": round(toks / max(1, n), 2),
            "mean_think_tokens": round(th_toks / max(1, n), 2),
            "think_forced_count": n_forced,
            "think_forced_pct": round(100.0 * n_forced / max(1, n), 2),
            "think_forced_among_thinking_pct": round(100.0 * n_forced / max(1, n_think_mode), 2) if n_think_mode else 0.0,
            "think_unclosed_count": n_unclosed,
            "think_unclosed_pct": round(100.0 * n_unclosed / max(1, n), 2),
            "mean_decode_tok_s": round(sum(float(r.get("tok_s", 0.0)) for r in recs) / max(1, n), 2),
        }
        tot_n += n
        tot_p += p

    all_recs = [r for s_k in ("mmlu", "mmlu_pro", "gsm8k") for r in by_suite[s_k]]
    overall_lo, overall_hi = wilson_ci_95(tot_p, tot_n)
    macro_pct = round(
        (
            suite_stats["mmlu"]["pass_rate_pct"]
            + suite_stats["mmlu_pro"]["pass_rate_pct"]
            + suite_stats["gsm8k"]["pass_rate_pct"]
        )
        / 3.0,
        2,
    )
    report = {
        "arm": arm_name,
        "config": ARM_CONFIGS[arm_name],
        "status": "COMPLETE" if (suite_stats["mmlu"]["n"] == 300 and suite_stats["mmlu_pro"]["n"] == 150 and suite_stats["gsm8k"]["n"] == 150) else "IN_PROGRESS",
        "elapsed_s": round(elapsed_s, 2),
        "total_n": tot_n,
        "total_passed": tot_p,
        "micro_pass_rate_pct": round(100.0 * tot_p / max(1, tot_n), 2),
        "micro_wilson_ci_95_pct": [overall_lo, overall_hi],
        "macro_pass_rate_pct": macro_pct,
        "overall_mean_tokens": round(sum(int(r.get("tokens", 0)) for r in all_recs) / max(1, tot_n), 2),
        "overall_mean_think_tokens": round(sum(int(r.get("think_tokens", 0)) for r in all_recs) / max(1, tot_n), 2),
        "overall_think_forced_pct": round(100.0 * sum(1 for r in all_recs if r.get("think_forced")) / max(1, tot_n), 2),
        "overall_think_unclosed_pct": round(100.0 * sum(1 for r in all_recs if r.get("think_unclosed")) / max(1, tot_n), 2),
        "suites": suite_stats,
    }
    tmp_p = out_json + ".tmp"
    with open(tmp_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(tmp_p, out_json)
    return report


def run_ablation_slice(
    arm_name: str,
    suite_keys: list[str],
    limit_end: int = 0,
    questions_json: str = DEFAULT_QUESTIONS_JSON,
    progress_jsonl: str | None = None,
    out_json: str | None = None,
    routing_dir: str | None = None,
    max_wall_s: float = 1260.0,
) -> dict[str, Any]:
    cfg = apply_arm_env(arm_name)
    progress_jsonl = progress_jsonl or f"/content/ablate_{arm_name}_progress.jsonl"
    out_json = out_json or f"/content/ablate_{arm_name}_report.json"
    routing_dir = routing_dir or f"/content/ablate_{arm_name}_routing"
    log_routing = cfg["log_routing"] == "1"
    if log_routing:
        os.makedirs(routing_dir, exist_ok=True)

    q_bank = build_or_load_fixed_600_questions(questions_json=questions_json)
    model, tokenizer, gen_fn, encode_fn = get_resident_model_for_ablation()
    done_map = base_bench._load_completed_records(progress_jsonl)
    t0 = time.perf_counter()

    for s_key in suite_keys:
        items = q_bank[s_key]
        end_idx = len(items) if limit_end <= 0 else min(len(items), limit_end)
        s_cfg = cfg["suites"][s_key]
        t_mode = str(s_cfg["thinking_mode"])
        max_new = int(s_cfg["max_new_tokens"])
        max_think = int(s_cfg["max_think_tokens"])
        kv_limit = int(cfg["kv_limit"])
        grader = _GRADERS[s_key]

        print(
            f"\n=== [{arm_name}:{s_key}] Running items 0..{end_idx - 1} / {len(items)} | "
            f"mode={t_mode} max_new={max_new} kv_limit={kv_limit} "
            f"guards_off={cfg['ablate_guards']} ceil={cfg['think_ceiling'] or 'default(384)'} ===",
            flush=True,
        )

        done_cnt = sum(1 for it in items[:end_idx] if (s_key, str(it["task_id"])) in done_map)
        pass_cnt = sum(
            1 for it in items[:end_idx]
            if (s_key, str(it["task_id"])) in done_map and done_map[(s_key, str(it["task_id"]))].get("passed")
        )

        for idx_i in range(end_idx):
            it = items[idx_i]
            tid = str(it["task_id"])
            if (s_key, tid) in done_map:
                continue
            if max_wall_s > 0 and (time.perf_counter() - t0) >= max_wall_s:
                print(
                    f"\n[{arm_name}:{s_key}] Reached slice wall budget ({max_wall_s:.0f}s) at idx={idx_i}; saving checkpoint.",
                    flush=True,
                )
                break

            item_prompt = it["prompt"]
            if arm_name == "A_unleashed" and grader is _grade_gsm8k:
                item_prompt = item_prompt.replace(
                    r"end with \boxed{<number>}:",
                    r"put your final numerical answer inside \boxed{} (with no units):",
                ).replace(
                    r"\boxed{<number>}",
                    r"\boxed{}",
                )
            safe_prompt = base_bench._fit_prompt_to_kv_budget(
                tokenizer, encode_fn, item_prompt, t_mode, max_new, kv_limit=kv_limit
            )
            if arm_name == "A_unleashed":
                spec_eng, runner_mod = _get_unleashed_engine(model, gen_fn)
                res = spec_eng.generate_with_speculative_and_prefix(
                    model,
                    tokenizer,
                    runner_mod.dsv41_model_mod,
                    runner_mod,
                    [{"role": "user", "content": safe_prompt}],
                    max_new_tokens=max_new,
                    max_think_tokens=max_think,
                    think_ceiling=3072,
                    ablate_guards=True,
                    thinking_mode=t_mode,
                    use_speculative=False,
                )
                eff_grader = _grade_gsm8k_clean if grader is _grade_gsm8k else grader
                ok, meta = eff_grader(it, res)
            else:
                res = gen_fn(
                    model,
                    tokenizer,
                    [{"role": "user", "content": safe_prompt}],
                    max_new_tokens=max_new,
                    stop_on_eos=True,
                    thinking_mode=t_mode,
                    thought_prefill="",
                    max_think_tokens=max_think,
                )
                ok, meta = grader(it, res)

            route_file = ""
            if log_routing and "route_ids_all" in res:
                route_tensor = res.pop("route_ids_all")
                route_file = os.path.join(routing_dir, f"{s_key}_{idx_i:03d}.pt")
                torch.save(
                    {
                        "suite": s_key,
                        "idx": idx_i,
                        "task_id": tid,
                        "passed": bool(ok),
                        "prompt_tokens": int(res.get("prompt_tokens", 0)),
                        "generated_tokens": int(res.get("generated_tokens", 0)),
                        "think_tokens": int(res.get("think_tokens", 0)),
                        "routed_total_hits": int(res.get("routed_total_hits", 0)),
                        "routed_cold_hits": int(res.get("routed_cold_hits", 0)),
                        "routed_cold_fraction": float(res.get("routed_cold_fraction", 0.0)),
                        "gen_cold_fraction": float(res.get("gen_cold_fraction", 0.0)),
                        "route_ids": route_tensor,
                    },
                    route_file,
                )

            rec = {
                "arm": arm_name,
                "suite": s_key,
                "idx": idx_i,
                "task_id": tid,
                "group": it.get("group", ""),
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
                "routed_total_hits": int(res.get("routed_total_hits", 0)),
                "routed_cold_hits": int(res.get("routed_cold_hits", 0)),
                "routed_cold_fraction": round(float(res.get("routed_cold_fraction", 0.0)), 6),
                "gen_cold_fraction": round(float(res.get("gen_cold_fraction", 0.0)), 6),
                "detail": str(meta)[:180],
                "answer_preview": str(res.get("answer_text") or "")[:120],
            }
            base_bench._append_record(progress_jsonl, rec)
            done_map[(s_key, tid)] = rec
            done_cnt += 1
            if ok:
                pass_cnt += 1

            pct = 100.0 * pass_cnt / max(1, done_cnt)
            cold_msg = f" cold={100.0*rec['routed_cold_fraction']:.1f}%" if log_routing else ""
            print(
                f"  [{arm_name}:{s_key} {done_cnt:3d}/{len(items):3d}] Pass={pass_cnt}/{done_cnt} ({pct:5.1f}%) | "
                f"idx={idx_i:3d} {tid}: ok={ok} tok={rec['tokens']}(th={rec['think_tokens']}, {rec['think_close_reason']}) "
                f"@ {rec['tok_s']:.1f} tok/s{cold_msg} | {rec['detail']}",
                flush=True,
            )
            if done_cnt % 5 == 0:
                summarize_arm_report(arm_name, progress_jsonl, out_json, time.perf_counter() - t0)

    rep = summarize_arm_report(arm_name, progress_jsonl, out_json, time.perf_counter() - t0)
    print(f"\n[{arm_name}] Snapshot saved to {out_json} (total_n={rep['total_n']}, macro={rep['macro_pass_rate_pct']}%)", flush=True)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="Controlled Single-Variable Ablations for Chronos DSV4.1")
    ap.add_argument("--arm", required=True, choices=sorted(ARM_CONFIGS.keys()))
    ap.add_argument("--suite", default="mmlu,mmlu_pro,gsm8k", help="Comma-separated suites from mmlu,mmlu_pro,gsm8k")
    ap.add_argument("--limit-end", type=int, default=0, help="Run indices [0, limit-end) per suite (0 = all)")
    ap.add_argument("--questions-json", default=DEFAULT_QUESTIONS_JSON)
    ap.add_argument("--progress-jsonl", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    suites = [s.strip().lower() for s in args.suite.split(",") if s.strip()]
    prog = args.progress_jsonl or f"/content/ablate_{args.arm}_progress.jsonl"
    out = args.out_json or f"/content/ablate_{args.arm}_report.json"
    run_ablation_slice(
        arm_name=args.arm,
        suite_keys=suites,
        limit_end=args.limit_end,
        questions_json=args.questions_json,
        progress_jsonl=prog,
        out_json=out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
