#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/defnotean/pb27
MODEL_DIR=/home/defnotean/pb27/hf/dsv41_chronos_v75_abliterated
SERVING_DIR="$ROOT/serving/pb27_vllm"
SERVED_NAME=${1:-deepseek-v4.1-flash}
PORT=${2:-8000}
WAIT_READY=${3:-0}

# 1. Ensure newest dsv41_chronos_moe.py and patch_dsv41_vllm.py are synced to serving plugin dir
for f in dsv41_chronos_moe.py patch_dsv41_vllm.py; do
  newest=""
  for cand in "$SERVING_DIR/chronos_1t/models/$f" "$MODEL_DIR/$f" "$MODEL_DIR/chronos_1t/models/$f"; do
    if [ -f "$cand" ]; then
      if [ -z "$newest" ] || [ "$cand" -nt "$newest" ]; then
        newest="$cand"
      fi
    fi
  done
  if [ -n "$newest" ]; then
    for dest in "$SERVING_DIR/chronos_1t/models/$f" "$MODEL_DIR/$f" "$MODEL_DIR/chronos_1t/models/$f"; do
      if [ "$newest" != "$dest" ] && [ -d "$(dirname "$dest")" ]; then
        cp -f "$newest" "$dest"
      fi
    done
  fi
done
rm -rf "$SERVING_DIR/chronos_1t/models/__pycache__" "$SERVING_DIR/pb27_vllm/__pycache__" 2>/dev/null || true

# 2. Stop any existing pb27-vllm container
docker rm -f pb27-vllm >/dev/null 2>&1 || true
sleep 2

# 3. Launch pb27-vllm container on :$PORT (hard-capped at 108g RAM / 0 swap to protect host sshd/OS)
nohup bash /home/defnotean/gpu_run.sh docker run --rm --name pb27-vllm --gpus all --ipc host --network host \
  --memory=108g --memory-swap=108g \
  -v /home/defnotean/.cache/huggingface:/root/.cache/huggingface \
  -v /home/defnotean/.cache/vllm:/root/.cache/vllm \
  -v /home/defnotean/.cache/flashinfer:/root/.cache/flashinfer \
  -v /home/defnotean/.cache/triton:/root/.triton \
  -v /home/defnotean/.cache/huggingface:/home/defnotean/.cache/huggingface:ro \
  -v "$MODEL_DIR:$MODEL_DIR:ro" \
  -v "$SERVING_DIR:/pb27_vllm" \
  -e PYTHONFAULTHANDLER=1 \
  -e PYTHONUNBUFFERED=1 \
  -e HF_HUB_OFFLINE=1 \
  -e PB27_MEMORY=0 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e CHRONOS_TIER1_4BIT_SLOTS=${CHRONOS_TIER1_4BIT_SLOTS:-16} \
  -e CHRONOS_TIER2_2BIT_SLOTS=${CHRONOS_TIER2_2BIT_SLOTS:-80} \
  -e CHRONOS_TIER3_1BIT_SLOTS=${CHRONOS_TIER3_1BIT_SLOTS:-288} \
  -e CHRONOS_DECODE_TOP_K=${CHRONOS_DECODE_TOP_K:-6} \
  --entrypoint bash vllm/vllm-openai:v0.31.0-aarch64 \
  -c "pip install --no-deps --no-build-isolation -q -e /pb27_vllm && exec vllm serve $MODEL_DIR \
    --served-model-name $SERVED_NAME --port $PORT --host 127.0.0.1 \
    --max-model-len 1048576 --max-num-batched-tokens 1024 --max-num-seqs 4 \
    --gpu-memory-utilization 0.88 --kv-cache-memory-bytes 2G --block-size 64 \
    --no-enable-prefix-caching \
    --compilation-config '{\"cudagraph_capture_sizes\":[1,2,4,8]}' \
    --reasoning-parser deepseek_v41 --enable-auto-tool-choice --tool-call-parser deepseek_v41" \
  > "$ROOT/vllm_dsv41.log" 2>&1 < /dev/null &

# 4. Ensure studio_proxy.py is running on :18080
if ! ss -ltn 2>/dev/null | grep -q ":18080 "; then
  nohup python3 "$SERVING_DIR/studio_proxy.py" --listen 18080 \
    --upstream "http://127.0.0.1:$PORT" \
    --model "$SERVED_NAME" --public-model "$SERVED_NAME" \
    --checkpoint "dsv41_chronos_v75_abliterated" \
    --context 1048576 --max-output 65536 \
    --decoder "vLLM Chronos v7.5 + Upgrade-2 Hybrid FP4/FP8 (Abliterated)" \
    > "$ROOT/studio_front.log" 2>&1 < /dev/null &
fi

cp -f "$ROOT/start_dsv41_vllm.sh" "$SERVING_DIR/start_dsv41_studio.sh"
echo "Launched pb27-vllm ($SERVED_NAME) on :$PORT (log: $ROOT/vllm_dsv41.log) and studio_proxy on :18080"

if [ "$WAIT_READY" = "--wait" ] || [ "$WAIT_READY" = "1" ]; then
  echo "Waiting for vLLM API server on http://127.0.0.1:$PORT/v1/models ..."
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "vLLM is READY on :$PORT (and :18080 health: $(curl -s http://127.0.0.1:18080/health))"
      exit 0
    fi
    sleep 5
  done
  echo "Timed out waiting for vLLM on :$PORT"
  exit 1
fi
