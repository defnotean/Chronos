#!/usr/bin/env python3
"""
GLM-5.3 Chronos v7.5 on a single Colab GPU (A100-80GB / H100-80GB / G4 RTX PRO 6000) — load, chat, benchmark, serve.

    # stream the converted checkpoint from the Hub (shards are deleted after each layer is resident)
    python scripts/run_glm53_chronos_colab.py --repo defnoteatin/PseudoBrain-GLM-5.3-Chronos-v7.5 \
        --model-dir /content/glm53_chronos --stream-delete --benchmark

    # OpenAI-compatible endpoint (http://127.0.0.1:8000/v1/chat/completions)
    python scripts/run_glm53_chronos_colab.py --model-dir /content/glm53_chronos --serve --port 8000

The residency plan (which experts are T1/T2/T3 on the GPU vs. pinned host RAM vs. not resident) is printed at start;
override with CHRONOS_GLM53_T1_SLOTS / _T2_SLOTS / _T3_SLOTS / _T3_HOST_SLOTS. Set CHRONOS_GLM53_TIER_CACHE=/content/
glm53_tiers to persist built tiers on disk so a runtime restart skips the reconstruction (~1 min on A100).
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from chronos_1t.models.glm53_chronos_spec import GLM53_CHRONOS_REPO_ID, PROFILES, detect_profile_name  # noqa: E402
from chronos_1t.models.glm53_engine import Glm53ChronosEngine  # noqa: E402

BENCH_PROMPTS = [
    "Write a Python function that merges overlapping intervals and explain its complexity.",
    "A train travels 300 km at 75 km/h and returns at 100 km/h. What is the average speed for the round trip?",
    "Explain the difference between a mutex and a semaphore with a short C example.",
    "Return a JSON object with keys name, version and dependencies for a minimal FastAPI project.",
]


def load_tokenizer(model_dir: Path, repo: str | None):
    from transformers import AutoTokenizer

    if repo:
        from huggingface_hub import hf_hub_download

        for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
            if not (model_dir / f).exists():
                try:
                    hf_hub_download(repo_id=repo, filename=f, local_dir=str(model_dir))
                except Exception:
                    pass
    return AutoTokenizer.from_pretrained(str(model_dir))


def chat_ids(tok, messages: list[dict], enable_thinking: bool = False) -> list[int]:
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=enable_thinking)
    except TypeError:
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=GLM53_CHRONOS_REPO_ID, help="HF repo to stream shards from ('' = local only)")
    ap.add_argument("--model-dir", default="/content/glm53_chronos")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--profile", default="", choices=[""] + list(PROFILES))
    ap.add_argument("--stream-delete", action="store_true", help="Delete each shard after its layer is resident")
    ap.add_argument("--layers", type=int, default=0, help="Debug: only build the first N layers")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="Record routing counts -> chronos_glm53_expert_stats.json")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--served-model-name", default="glm-5.3-chronos")
    args = ap.parse_args()

    if args.profile:
        os.environ["CHRONOS_GLM53_PROFILE"] = args.profile
    if args.calibrate:
        os.environ["CHRONOS_GLM53_TRACK"] = "1"
    os.environ.setdefault("CHRONOS_GLM53_CTX", str(args.ctx))
    model_dir = Path(args.model_dir)
    repo = args.repo or None
    print(f"[colab] device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} profile={detect_profile_name()}", flush=True)

    tok = load_tokenizer(model_dir, repo) if (repo or (model_dir / "tokenizer.json").exists()) else None
    eng = Glm53ChronosEngine(
        model_dir, max_len=args.ctx, num_layers=args.layers or None, repo_id=repo, delete_shards_after_load=args.stream_delete,
    )
    print(f"[colab] memory: {json.dumps(eng.memory_report())}", flush=True)
    lock = threading.Lock()

    def run(messages: list[dict], max_new: int, temperature: float, on_tok=None) -> tuple[str, dict]:
        ids = chat_ids(tok, messages, args.thinking)
        with lock:
            res = eng.generate(ids, max_new_tokens=max_new, temperature=temperature, on_token=on_tok)
        return tok.decode(res["tokens"], skip_special_tokens=True), res

    if args.prompt:
        text, res = run([{"role": "user", "content": args.prompt}], args.max_new_tokens, args.temperature)
        print(text)
        print(json.dumps({k: v for k, v in res.items() if k != "tokens"}), flush=True)

    if args.benchmark:
        rows = []
        for p in BENCH_PROMPTS:
            text, res = run([{"role": "user", "content": p}], args.max_new_tokens, 0.0)
            rows.append({k: v for k, v in res.items() if k != "tokens"})
            print(f"--- {p}\n{text[:600]}\n{json.dumps(rows[-1])}", flush=True)
        mean = sum(r["decode_tok_s"] for r in rows) / len(rows)
        report = {"profile": detect_profile_name(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                  "mean_decode_tok_s": round(mean, 2), "runs": rows, "memory": eng.memory_report()}
        out = model_dir / "glm53_chronos_colab_benchmark.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"[bench] mean decode {mean:.2f} tok/s -> {out}", flush=True)

    if args.calibrate:
        from chronos_1t.models.glm53_chronos_moe import save_expert_stats

        print(f"[calibrate] routing stats -> {save_expert_stats(model_dir)} (next load ranks tiers by real usage)", flush=True)

    if args.serve:
        if tok is None:
            raise SystemExit("tokenizer files are required for --serve")

        class Handler(BaseHTTPRequestHandler):
            def _json(self, code: int, obj: dict) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                if self.path.rstrip("/") in ("/v1/models", "/models"):
                    self._json(200, {"object": "list", "data": [{"id": args.served_model_name, "object": "model",
                                                                 "context_length": args.ctx}]})
                elif self.path.rstrip("/") == "/health":
                    self._json(200, {"status": "ok"})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                    return self._json(404, {"error": "not found"})
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                msgs = req.get("messages", [])
                max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 512)
                temp = float(req.get("temperature", 0.0) or 0.0)
                rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
                if req.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    sent = [0]
                    toks: list[int] = []

                    def on_tok(t: int) -> None:
                        toks.append(t)
                        txt = tok.decode(toks, skip_special_tokens=True)
                        delta, sent[0] = txt[sent[0]:], len(txt)
                        if delta:
                            chunk = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": args.served_model_name,
                                     "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            self.wfile.flush()

                    run(msgs, max_new, temp, on_tok)
                    end = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": args.served_model_name,
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())
                    return
                text, res = run(msgs, max_new, temp)
                self._json(200, {
                    "id": rid, "object": "chat.completion", "created": created, "model": args.served_model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": res["prefill_tokens"], "completion_tokens": res["decode_tokens"],
                              "total_tokens": res["prefill_tokens"] + res["decode_tokens"]},
                })

            def log_message(self, *a):  # quiet
                return

        print(f"[serve] OpenAI-compatible endpoint on http://{args.host}:{args.port}/v1 (model={args.served_model_name})", flush=True)
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
