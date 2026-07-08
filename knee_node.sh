#!/usr/bin/env bash
# KNEE SWEEP (on-node): launch ONE server with the FROZEN eval flags (+ extra policy args / env),
# then run the mix bench at several request-rates (lambda) to trace the goodput-under-SLO curve.
# This is OFF the fixed per-version protocol (which is lambda=3) — used only to VERIFY the curve shift.
# Reuses eval.sh's exact launch flags + bench invocation; varies only --request-rate.
# Usage: knee_node.sh <name> <label> <lambdas_csv> [extra launch args...]
set -uo pipefail
ROOT="${AUTORESEARCH_ROOT:-/home/junyanch_google_com/autoresearch}"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
SGL_HOME="${SGL_HOME:-$ROOT/programs/sgl/v0.25_ablations/base_free}"
SGL_WORKSPACE="${SGL_WORKSPACE:-$ROOT/workspace/sgl/v0.25_ablations/base_free}"
NAME="${1:?name}"; LABEL="${2:?label}"; LAMS="${3:?lambdas csv e.g. 3,4,5}"; shift 3
EXTRA=("$@")
WORK=$SGL_WORKSPACE/researchers/$NAME
PORT=${PORT:-30000}; OUT="$WORK/runs/knee-$LABEL"; mkdir -p "$OUT"
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
source "$WORK/.venv/bin/activate"; export PYTHONPATH="$WORK/python"; cd "$WORK"
export TRITON_CACHE_DIR="$WORK/.cache/triton" CUDA_CACHE_PATH="$WORK/.cache/nv" \
       FLASHINFER_CACHE_DIR="$WORK/.cache/flashinfer" XDG_CACHE_HOME="$WORK/.cache"
mkdir -p "$WORK/.cache"
MODEL=Qwen/Qwen3.5-122B-A10B-FP8
MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
DRAMG=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
[ "${DRAMG:-0}" -lt 1300 ] && { echo "DRAM_TOO_LOW ${DRAMG}G on $(hostname)"; exit 2; }
# same FROZEN launch as eval.sh (+ EXTRA last-wins)
LAUNCH=(python3 -m sglang.launch_server
  --model-path "$MODEL" --tp 8 --trust-remote-code
  --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85
  --enable-hierarchical-cache --hicache-size 96 --page-size 64
  --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through
  --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion
  --enable-metrics --enable-cache-report --port "$PORT")
[ ${#EXTRA[@]} -gt 0 ] && LAUNCH+=("${EXTRA[@]}")
echo ">>> knee launch ($LABEL); extra=[${EXTRA[*]:-none}] XTIER_LAZY=${SGLANG_XTIER_LAZY:-0} WM=${SGLANG_XTIER_WM_FRAC:-} "
"${LAUNCH[@]}" > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; sleep 5; pkill -9 -f "[s]glang.launch_server" 2>/dev/null' EXIT
for i in $(seq 1 900); do
  curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { echo "server ready (~$((i*3))s)"; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED"; tail -40 "$OUT/server.log"; exit 3; }
  sleep 3
done
curl -sf "localhost:$PORT/health" >/dev/null 2>&1 || { echo "SERVER_TIMEOUT"; exit 3; }
IFS=',' read -ra LAMARR <<< "$LAMS"
for LAM in "${LAMARR[@]}"; do
  echo ">>> bench lambda=$LAM"; curl -s -X POST "localhost:$PORT/flush_cache" >/dev/null 2>&1; sleep 5
  python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
    --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
    --request-rate "$LAM" --max-concurrency 128 --num-prompts 1553 --port $PORT \
    --output-file "$OUT/res_lam$LAM.json" 2>&1 | tee "$OUT/bench_lam$LAM.txt" | grep -E "Request throughput|Median TTFT|P99 TTFT|Mean TTFT|Successful"
done
echo "================ KNEE_DONE label=$LABEL ================"
