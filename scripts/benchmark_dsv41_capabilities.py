#!/usr/bin/env python3
"""Comprehensive Capability, Intelligence & Speed Benchmark Suite for DeepSeek-V4.1-Flash (Chronos v7.5).

Tests both direct vLLM (:8000) and Studio/OpenCode Proxy (:18080) across 6 real-world tasks:
  1. Exact Arithmetic & Step-by-Step Math (17 * 23 = 391)
  2. Multi-Step Word / Logic Problem (3-step algebra problem -> 7 pens)
  3. Algorithm & Code Generation (150+ tokens, Merge Sort + ast.parse + subprocess execution verification)
  4. Long-Form Coherent Explanation (200+ tokens, MoE + MLA, 4-gram uniqueness > 0.85)
  5. Multi-Turn Conversation Coherence (2-turn state retention: CHRONOS-9, 72 GPUs, 5760 GB)
  6. OpenCode / Tool-Calling Compatibility (OpenCode-style `tools` array with `read_file` & `bash`)

Reports TTFT (ms), decode throughput (tok/s), end-to-end throughput (tok/s), total tokens,
4-gram uniqueness ratio, and pass/fail status for each task and endpoint.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


def compute_4gram_uniqueness(text: str) -> float:
    """Compute exact word/token 4-gram uniqueness ratio in [0.0, 1.0]."""
    tokens = re.findall(r"\S+", text.lower())
    if len(tokens) < 4:
        return 1.0
    ngrams = [tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def extract_python_code(text: str) -> str:
    """Extract Python code from markdown code fences or fallback to raw text."""
    # Prefer ```python ... ``` blocks
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        # Join all blocks or pick the longest block containing 'def '
        def_blocks = [b for b in m if "def " in b]
        if def_blocks:
            return "\n\n".join(def_blocks).strip()
        return max(m, key=len).strip()
    # Handle unclosed ```python fence if output hit max_tokens after code finished
    m_open = re.search(r"```(?:python|py)?\s*\n(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if m_open:
        candidate = m_open.group(1).strip()
        # Strip trailing ``` if present
        candidate = re.sub(r"```.*$", "", candidate, flags=re.DOTALL).strip()
        return candidate
    return text.strip()


def stream_chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 512,
    temperature: float = 0.0,
    tools: Optional[List[Dict[str, Any]]] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Send a streaming chat completion request and measure TTFT and decode tok/s."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools is not None:
        body["tools"] = tools

    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-loopback"},
        method="POST",
    )

    t_start = time.perf_counter()
    t_first: Optional[float] = None
    content_chunks: List[str] = []
    reasoning_chunks: List[str] = []
    tool_calls_by_idx: Dict[int, Dict[str, Any]] = {}
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    chunk_count = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                except ValueError:
                    continue

                if obj.get("usage"):
                    usage = obj["usage"]

                choices = obj.get("choices") or []
                if not choices:
                    continue
                ch0 = choices[0]
                if ch0.get("finish_reason"):
                    finish_reason = ch0["finish_reason"]
                delta = ch0.get("delta") or {}

                c_piece = delta.get("content") or ""
                r_piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                tc_deltas = delta.get("tool_calls") or []

                if c_piece or r_piece or tc_deltas:
                    if t_first is None:
                        t_first = time.perf_counter()
                    chunk_count += 1

                if c_piece:
                    content_chunks.append(c_piece)
                if r_piece:
                    reasoning_chunks.append(r_piece)
                for tc in tc_deltas:
                    idx = tc.get("index", 0)
                    entry = tool_calls_by_idx.setdefault(
                        idx,
                        {
                            "id": tc.get("id") or f"call_{idx}",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        entry["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        entry["function"]["arguments"] += fn["arguments"]
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return {
            "ok": False,
            "error": f"HTTP {e.code}: {err_body[:500]}",
            "content": "",
            "reasoning_content": "",
            "full_text": "",
            "tool_calls": [],
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "ttft_ms": 0.0,
            "decode_tps": 0.0,
            "e2e_tps": 0.0,
            "total_time_s": time.perf_counter() - t_start,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "content": "",
            "reasoning_content": "",
            "full_text": "",
            "tool_calls": [],
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "ttft_ms": 0.0,
            "decode_tps": 0.0,
            "e2e_tps": 0.0,
            "total_time_s": time.perf_counter() - t_start,
        }

    t_end = time.perf_counter()
    content = "".join(content_chunks)
    reasoning = "".join(reasoning_chunks)
    tool_calls = [tool_calls_by_idx[k] for k in sorted(tool_calls_by_idx.keys())]
    full_text = (reasoning + "\n" + content).strip() if reasoning else content

    comp_tokens = (
        int(usage["completion_tokens"])
        if usage and usage.get("completion_tokens") is not None
        else max(chunk_count, len(re.findall(r"\S+", full_text)))
    )
    prompt_tokens = int(usage.get("prompt_tokens", 0)) if usage else 0

    total_time = max(1e-6, t_end - t_start)
    ttft_s = (t_first - t_start) if t_first is not None else total_time
    decode_time = max(1e-6, t_end - (t_first if t_first is not None else t_start))
    decode_tps = (comp_tokens - 1) / decode_time if comp_tokens > 1 else comp_tokens / total_time
    e2e_tps = comp_tokens / total_time

    return {
        "ok": True,
        "error": None,
        "content": content,
        "reasoning_content": reasoning,
        "full_text": full_text,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "completion_tokens": comp_tokens,
        "prompt_tokens": prompt_tokens,
        "ttft_ms": ttft_s * 1000.0,
        "decode_time_s": decode_time,
        "total_time_s": total_time,
        "decode_tps": decode_tps,
        "e2e_tps": e2e_tps,
        "uniqueness_4gram": compute_4gram_uniqueness(full_text),
    }


# ============================================================================
# Task Implementations
# ============================================================================

def task_1_arithmetic(base_url: str, model: str) -> Dict[str, Any]:
    prompt = "What is 17 * 23? Show each step clearly and state the final number."
    res = stream_chat_completion(
        base_url,
        model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    if not res["ok"]:
        return {**res, "task": "1_arithmetic", "passed": False, "reason": res["error"]}

    text = res["full_text"]
    has_391 = bool(re.search(r"\b391\b", text))
    u4 = res["uniqueness_4gram"]
    coherent = u4 >= 0.80 and len(text.strip()) >= 10
    passed = has_391 and coherent
    reason = f"has_391={has_391}, 4gram_uniq={u4:.3f}, tokens={res['completion_tokens']}"
    return {**res, "task": "1_arithmetic", "passed": passed, "reason": reason}


def task_2_word_logic(base_url: str, model: str) -> Dict[str, Any]:
    prompt = (
        "A store sells notebooks for $4 each and pens for $2 each. "
        "Maria buys 6 notebooks and some pens. She pays with a $50 bill and receives $12 in change. "
        "How many pens did Maria buy? Show each step clearly and state the final number."
    )
    res = stream_chat_completion(
        base_url,
        model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=320,
        temperature=0.0,
    )
    if not res["ok"]:
        return {**res, "task": "2_word_logic", "passed": False, "reason": res["error"]}

    text = res["full_text"]
    has_intermediate = any(bool(re.search(rf"\b{n}\b", text)) for n in (38, 24, 14))
    has_7 = bool(re.search(r"\b(?:7|seven)\b", text.lower()))
    u4 = res["uniqueness_4gram"]
    passed = has_intermediate and has_7 and u4 >= 0.80
    reason = f"has_7={has_7}, has_intermediate(38/24/14)={has_intermediate}, 4gram_uniq={u4:.3f}"
    return {**res, "task": "2_word_logic", "passed": passed, "reason": reason}


def task_3_merge_sort_code(base_url: str, model: str) -> Dict[str, Any]:
    prompt = (
        "Write a complete, well-commented Python implementation of merge sort and include a "
        "test case sorting [38, 27, 43, 3, 9, 82, 10]."
    )
    res = stream_chat_completion(
        base_url,
        model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=640,
        temperature=0.0,
    )
    if not res["ok"]:
        return {**res, "task": "3_merge_sort_code", "passed": False, "reason": res["error"]}

    # Prefer content, fallback to full_text if code is in reasoning
    source_text = res["content"] if "def " in res["content"] else res["full_text"]
    code = extract_python_code(source_text)
    tokens_ok = res["completion_tokens"] >= 150

    syntax_ok = False
    exec_ok = False
    exec_out = ""
    try:
        tree = ast.parse(code)
        syntax_ok = True
        # Find top-level function definitions so we can also invoke the sort function directly if needed
        fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        sort_fn = next((f for f in fn_names if "merge" in f.lower() or "sort" in f.lower()), fn_names[0] if fn_names else None)
        harness = code + "\n\n"
        if sort_fn:
            harness += (
                f"try:\n"
                f"    _r = {sort_fn}([38, 27, 43, 3, 9, 82, 10])\n"
                f"    if _r is not None:\n"
                f"        print('__SORT_RESULT__:', _r)\n"
                f"except Exception:\n"
                f"    pass\n"
            )
        proc = subprocess.run(
            [sys.executable, "-c", harness],
            capture_output=True,
            text=True,
            timeout=10,
        )
        exec_out = (proc.stdout + "\n" + proc.stderr).strip()
        exec_ok = proc.returncode == 0 and "[3, 9, 10, 27, 38, 43, 82]" in proc.stdout
    except SyntaxError as e:
        exec_out = f"SyntaxError: {e}"
    except Exception as e:
        exec_out = f"{type(e).__name__}: {e}"

    passed = tokens_ok and syntax_ok and exec_ok
    reason = (
        f"tokens={res['completion_tokens']}(>=150:{tokens_ok}), "
        f"ast_parse={syntax_ok}, exec_sorted={exec_ok}, out={exec_out[:120]!r}"
    )
    return {
        **res,
        "task": "3_merge_sort_code",
        "extracted_code": code,
        "exec_output": exec_out,
        "passed": passed,
        "reason": reason,
    }


def task_4_long_explanation(base_url: str, model: str) -> Dict[str, Any]:
    prompt = (
        "Provide a detailed, comprehensive technical explanation (at least 250 words) of how "
        "Mixture-of-Experts (MoE) routing and Multi-head Latent Attention (MLA) work in modern "
        "transformer architectures, including their mathematical intuition and memory/compute trade-offs."
    )
    res = stream_chat_completion(
        base_url,
        model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=450,
        temperature=0.0,
    )
    if not res["ok"]:
        return {**res, "task": "4_long_explanation", "passed": False, "reason": res["error"]}

    text = res["full_text"]
    low = text.lower()
    tokens_ok = res["completion_tokens"] >= 200
    u4 = res["uniqueness_4gram"]
    uniq_ok = u4 > 0.85
    has_moe = "expert" in low or "router" in low or "routing" in low
    has_mla = "latent" in low or "kv" in low or "attention" in low
    passed = tokens_ok and uniq_ok and has_moe and has_mla
    reason = (
        f"tokens={res['completion_tokens']}(>=200:{tokens_ok}), "
        f"4gram_uniq={u4:.4f}(>0.85:{uniq_ok}), has_moe={has_moe}, has_mla={has_mla}"
    )
    return {**res, "task": "4_long_explanation", "passed": passed, "reason": reason}


def task_5_multi_turn(base_url: str, model: str) -> Dict[str, Any]:
    turn1_user = (
        "We are configuring a server cluster codenamed 'CHRONOS-9'. "
        "It has 12 nodes, and each node has 8 GPUs with 80 GB of VRAM per GPU. "
        "What is the total GPU count and total VRAM (in GB) across the entire cluster? Keep your answer concise."
    )
    res1 = stream_chat_completion(
        base_url,
        model,
        messages=[{"role": "user", "content": turn1_user}],
        max_tokens=256,
        temperature=0.0,
    )
    if not res1["ok"]:
        return {**res1, "task": "5_multi_turn", "passed": False, "reason": f"Turn 1 failed: {res1['error']}"}

    turn1_reply = res1["content"] if res1["content"].strip() else res1["full_text"]
    turn2_user = (
        "If we take 3 nodes of that cluster offline for maintenance, what is the codename "
        "of the cluster and how many active GPUs and total active VRAM (in GB) remain?"
    )
    messages_t2 = [
        {"role": "user", "content": turn1_user},
        {"role": "assistant", "content": turn1_reply},
        {"role": "user", "content": turn2_user},
    ]
    res2 = stream_chat_completion(
        base_url,
        model,
        messages=messages_t2,
        max_tokens=256,
        temperature=0.0,
    )
    if not res2["ok"]:
        return {**res2, "task": "5_multi_turn", "passed": False, "reason": f"Turn 2 failed: {res2['error']}"}

    t2_text = res2["full_text"]
    has_codename = "chronos-9" in t2_text.lower()
    has_72 = bool(re.search(r"\b72\b", t2_text))
    has_5760 = bool(re.search(r"\b5[,]?760\b", t2_text))
    u4 = res2["uniqueness_4gram"]
    passed = has_codename and has_72 and has_5760 and u4 >= 0.80
    reason = (
        f"has_chronos9={has_codename}, has_72_gpus={has_72}, "
        f"has_5760_gb={has_5760}, 4gram_uniq={u4:.3f}"
    )
    return {
        **res2,
        "task": "5_multi_turn",
        "turn1_response": turn1_reply,
        "turn1_ttft_ms": res1["ttft_ms"],
        "turn1_decode_tps": res1["decode_tps"],
        "passed": passed,
        "reason": reason,
    }


def task_6_opencode_tools(base_url: str, model: str) -> Dict[str, Any]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file at the given absolute path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {
                            "type": "string",
                            "description": "The absolute path to the file to read.",
                        }
                    },
                    "required": ["filePath"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to run."}
                    },
                    "required": ["command"],
                },
            },
        },
    ]
    prompt = "Use the `read_file` tool to read the file `/etc/hostname`."
    res = stream_chat_completion(
        base_url,
        model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
        tools=tools,
    )
    if not res["ok"]:
        return {**res, "task": "6_opencode_tools", "passed": False, "reason": res["error"]}

    tcs = res["tool_calls"]
    full = res["full_text"]
    valid_tc = False
    tc_summary = "none"
    if tcs:
        tc0 = tcs[0]
        fn_name = tc0.get("function", {}).get("name", "")
        raw_args = tc0.get("function", {}).get("arguments", "")
        try:
            parsed_args = json.loads(raw_args) if raw_args else {}
            valid_tc = fn_name == "read_file" and "/etc/hostname" in json.dumps(parsed_args)
            tc_summary = f"{fn_name}({raw_args})"
        except Exception:
            valid_tc = fn_name == "read_file" and "/etc/hostname" in raw_args
            tc_summary = f"{fn_name}(raw={raw_args})"
    elif "read_file" in full and "/etc/hostname" in full:
        # Also accept DSML / inline tool invocation if not parsed into delta.tool_calls
        valid_tc = True
        tc_summary = "inline_dsml_or_text_reference"

    passed = valid_tc
    reason = f"tool_call_ok={valid_tc}, tool_call={tc_summary}, finish_reason={res.get('finish_reason')}"
    return {**res, "task": "6_opencode_tools", "passed": passed, "reason": reason}


# ============================================================================
# Runner
# ============================================================================

ALL_TASKS = [
    (1, "Exact Arithmetic (17 * 23)", task_1_arithmetic),
    (2, "Multi-Step Word / Logic Problem", task_2_word_logic),
    (3, "Algorithm & Code Gen (Merge Sort 150+ tok)", task_3_merge_sort_code),
    (4, "Long-Form Explanation (MoE+MLA 200+ tok)", task_4_long_explanation),
    (5, "Multi-Turn Conversation Coherence", task_5_multi_turn),
    (6, "OpenCode Tool-Calling Compatibility", task_6_opencode_tools),
]


def run_suite_for_port(
    host: str,
    port: int,
    model: str,
    selected_ids: List[int],
) -> Dict[str, Any]:
    base_url = f"http://{host}:{port}"
    label = "direct_vllm" if port == 8000 else ("studio_proxy" if port == 18080 else f"port_{port}")
    print(f"\n{'=' * 84}")
    print(f"  ENDPOINT: {base_url} ({label}) | model={model}")
    print(f"{'=' * 84}")

    # Warmup request
    print("  [Warmup] Sending warmup request...", end=" ", flush=True)
    w = stream_chat_completion(
        base_url,
        model,
        messages=[{"role": "user", "content": "Reply with the single word READY."}],
        max_tokens=16,
        temperature=0.0,
    )
    if w["ok"]:
        print(f"OK (TTFT={w['ttft_ms']:.1f}ms, tokens={w['completion_tokens']})")
    else:
        print(f"FAILED ({w['error']})")

    results: List[Dict[str, Any]] = []
    for tid, tname, tfn in ALL_TASKS:
        if tid not in selected_ids:
            continue
        print(f"\n  -> Task {tid}: {tname}")
        r = tfn(base_url, model)
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"     [{status}] TTFT={r['ttft_ms']:.1f}ms | "
            f"Decode={r['decode_tps']:.2f} tok/s | "
            f"E2E={r['e2e_tps']:.2f} tok/s | "
            f"Tokens={r['completion_tokens']} | "
            f"{r['reason']}"
        )
        preview = (r.get("content") or r.get("full_text") or "").replace("\n", " ")[:180]
        if preview:
            print(f"     Sample: {preview!r}")

    passed_count = sum(1 for r in results if r["passed"])
    valid_tps = [r["decode_tps"] for r in results if r["ok"] and r["completion_tokens"] > 1]
    valid_ttft = [r["ttft_ms"] for r in results if r["ok"]]
    avg_decode_tps = sum(valid_tps) / len(valid_tps) if valid_tps else 0.0
    avg_ttft_ms = sum(valid_ttft) / len(valid_ttft) if valid_ttft else 0.0

    print(f"\n  --- Summary for :{port} ({label}) ---")
    print(
        f"  Passed: {passed_count}/{len(results)} | "
        f"Avg TTFT: {avg_ttft_ms:.1f} ms | "
        f"Avg Decode Throughput: {avg_decode_tps:.2f} tok/s"
    )

    return {
        "endpoint": base_url,
        "port": port,
        "label": label,
        "passed": passed_count,
        "total": len(results),
        "all_passed": passed_count == len(results) and len(results) > 0,
        "avg_ttft_ms": avg_ttft_ms,
        "avg_decode_tps": avg_decode_tps,
        "tasks": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="DeepSeek-V4.1-Flash End-to-End Capability & Speed Benchmark")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ports", default="8000,18080", help="Comma-separated ports to test (default: 8000,18080)")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--tasks", default="1,2,3,4,5,6", help="Comma-separated task numbers to run")
    ap.add_argument("--output", default="/home/defnotean/pb27/benchmark_dsv41_results.json")
    args = ap.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
    selected_ids = [int(t.strip()) for t in args.tasks.split(",") if t.strip()]

    suite_report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "endpoints": {},
    }

    overall_pass = True
    for port in ports:
        rep = run_suite_for_port(args.host, port, args.model, selected_ids)
        suite_report["endpoints"][str(port)] = rep
        if not rep["all_passed"]:
            overall_pass = False

    suite_report["overall_pass"] = overall_pass

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(suite_report, f, indent=2, ensure_ascii=False)
            print(f"\nSaved full benchmark report to: {args.output}")
        except Exception as e:
            print(f"\nWarning: could not write report to {args.output}: {e}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
