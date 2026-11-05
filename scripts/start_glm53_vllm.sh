#!/usr/bin/env bash
# GLM-5.3 Chronos v7.5 on the DGX Spark (GB10, 128 GB unified) via pb27-vllm (vLLM v0.31, DSA sparse MLA on sm_121).
#
#   bash scripts/start_glm53_vllm.sh [served-name] [port] [--wait]
#
# Safety: refuses to start while another pb27-vllm* container is running (e.g. the DeepSeek-V4.1 server started by
# start_dsv41_vllm.sh) because both models need most of the 128 GB unified memory. Set FORCE=1 to stop it first.
set -euo pipefail
ROOT=${ROOT:-/home/defnotean/pb27}
MODEL_DIR=${MODEL_DIR:-$ROOT/hf/glm53_chronos_v75}
TIER_CACHE=${TIER_CACHE:-$ROOT/hf/glm53_chronos_tiers}
SERVING_DIR=${SERVING_DIR:-$ROOT/serving/pb27_vllm}
CHRONOS_SRC=${CHRONOS_SRC:-$(cd "$(dirname "$0")/.." && pwd)}
SERVED_NAME=${1:-glm-5.3}
PORT=${2:-8001}
WAIT_READY=${3:-0}
CTX=${CTX:-65536}
CONTAINER=pb27-vllm-glm53

if [ ! -f "$MODEL_DIR/config.json" ] || ! grep -q '"arch": "glm53"' "$MODEL_DIR/config.json"; then
  echo "No GLM-5.3 Chronos checkpoint in $MODEL_DIR (expected config.json with chronos_holographic_moe.arch=glm53)."
  echo "Download it with: huggingface-cli download defnoteatin/PseudoBrain-GLM-5.3-Chronos-v7.5 --local-dir $MODEL_DIR"
  exit 1
fi

running=$(docker ps --format '{{.Names}}' | grep -E '^pb27-vllm' || true)
if [ -n "$running" ]; then
  if [ "${FORCE:-0}" != "1" ]; then
    echo "Refusing to start: pb27-vllm container(s) already running: $running (set FORCE=1 to stop them first)."
    exit 2
  fi
  for c in $running; do docker rm -f "$c" >/dev/null 2>&1 || true; done
fi

# 1. Sync the GLM-5.3 Chronos modules into the serving plugin tree (same mechanism as start_dsv41_vllm.sh)
mkdir -p "$SERVING_DIR/chronos_1t/models" "$SERVING_DIR/chronos_1t/converter" "$TIER_CACHE"
for f in glm53_chronos_spec.py glm53_tiers.py glm53_holo.py glm53_chronos_moe.py glm53_engine.py patch_glm53_vllm.py; do
  cp -f "$CHRONOS_SRC/chronos_1t/models/$f" "$SERVING_DIR/chronos_1t/models/$f"
done
cp -f "$CHRONOS_SRC/chronos_1t/converter/convert_glm53_flash_colab.py" "$SERVING_DIR/chronos_1t/converter/"
[ -f "$SERVING_DIR/chronos_1t/converter/__init__.py" ] || touch "$SERVING_DIR/chronos_1t/converter/__init__.py"
PLUGIN="$SERVING_DIR/chronos_1t/vllm_plugin.py"
if [ -f "$PLUGIN" ] && ! grep -q patch_glm53_vllm "$PLUGIN"; then
  cat >> "$PLUGIN" <<'EOF'


def _register_glm53() -> None:
    try:
        from chronos_1t.models.patch_glm53_vllm import enable_chronos_glm53_vllm_hook
        enable_chronos_glm53_vllm_hook()
    except Exception:
        pass


_orig_register_chronos = register


def register() -> None:  # noqa: F811
    _orig_register_chronos()
    _register_glm53()
EOF
elif [ ! -f "$PLUGIN" ]; then
  cp -f "$CHRONOS_SRC/chronos_1t/vllm_plugin.py" "$PLUGIN"
fi
rm -rf "$SERVING_DIR/chronos_1t/models/__pycache__" 2>/dev/null || true

# 2. Unified-memory hygiene (same as the DSV4.1 launcher)
sudo -n swapoff -a >/dev/null 2>&1 || true
sudo -n sysctl -w vm.min_free_kbytes=1048576 >/dev/null 2>&1 || true

# 3. Launch (container hard-capped at 108g; the residency planner assumes exactly this budget: profile spark_gb10)
nohup bash "$ROOT/../gpu_run.sh" docker run --rm --name "$CONTAINER" --gpus all --ipc host --network host \
  --memory=108g --memory-swap=108g --oom-score-adj=500 \
  -v /home/defnotean/.cache/vllm:/root/.cache/vllm \
  -v /home/defnotean/.cache/flashinfer:/root/.cache/flashinfer \
  -v /home/defnotean/.cache/triton:/root/.triton \
  -v "$MODEL_DIR:$MODEL_DIR:ro" \
  -v "$TIER_CACHE:$TIER_CACHE" \
  -v "$SERVING_DIR:/pb27_vllm" \
  -e PYTHONUNBUFFERED=1 -e HF_HUB_OFFLINE=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e TRITON_DEFAULT_FP_FUSION=1 \
  -e CHRONOS_GLM53_PROFILE=spark_gb10 \
  -e CHRONOS_GLM53_CTX="$CTX" \
  -e CHRONOS_GLM53_TIER_CACHE="$TIER_CACHE" \
  -e CHRONOS_GLM53_T1_SLOTS="${CHRONOS_GLM53_T1_SLOTS:-}" \
  -e CHRONOS_GLM53_T2_SLOTS="${CHRONOS_GLM53_T2_SLOTS:-}" \
  -e CHRONOS_GLM53_T3_SLOTS="${CHRONOS_GLM53_T3_SLOTS:-}" \
  --entrypoint bash vllm/vllm-openai:v0.31.0-aarch64 \
  -c "pip install --no-deps --no-build-isolation -q -e /pb27_vllm && exec vllm serve $MODEL_DIR \
    --served-model-name $SERVED_NAME --port $PORT --host 127.0.0.1 \
    --trust-remote-code --max-model-len $CTX --max-num-batched-tokens 2048 --max-num-seqs 2 \
    --gpu-memory-utilization 0.88 --kv-cache-memory-bytes 3G --block-size 64 \
    --no-enable-prefix-caching \
    --compilation-config '{\"cudagraph_capture_sizes\":[1,2,4,8],\"cudagraph_mode\":\"PIECEWISE\"}' \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47" \
  > "$ROOT/vllm_glm53.log" 2>&1 < /dev/null &

echo "Launched $CONTAINER ($SERVED_NAME) on :$PORT (log: $ROOT/vllm_glm53.log). First start reconstructs ~19k experts"
echo "into tier slots (a few minutes); restarts reuse $TIER_CACHE."

if [ "$WAIT_READY" = "--wait" ] || [ "$WAIT_READY" = "1" ]; then
  for i in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "vLLM is READY on :$PORT"; exit 0
    fi
    sleep 5
  done
  echo "Timed out waiting for vLLM on :$PORT"; exit 1
fi
