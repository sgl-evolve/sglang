#!/usr/bin/env bash
# Runs ON the profiler node. Launches the accel server (frozen config), drives a saturation load,
# triggers profile_by_stage, collects stage-separated chrome traces, cleans up.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
WORK=$ROOT/workspace/sgl/v0.31/research/researchers/turing
PORT=30056
PROFDIR=$WORK/runs/prof_accel_sat2
CACHE=/tmp/turing_prof_cache            # node-local, SEPARATE flashinfer/triton cache => NO race w/ dose-response
MODEL=Qwen/Qwen3.5-122B-A10B-FP8
MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
mkdir -p "$PROFDIR" "$CACHE"

set -a; . "$ROOT/.env" 2>/dev/null; set +a
export HF_TOKEN="${HF_API_KEY:-}" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
source "$WORK/.venv/bin/activate"
export PYTHONPATH="$WORK/python"; cd "$WORK"
export TRITON_CACHE_DIR="$CACHE/triton" CUDA_CACHE_PATH="$CACHE/nv" FLASHINFER_CACHE_DIR="$CACHE/flashinfer" XDG_CACHE_HOME="$CACHE"
export SGLANG_TORCH_PROFILER_DIR="$PROFDIR"
export SGLANG_TURING_GIANT_ACCEL=1 SGLANG_TURING_ACCEL_THETA=0.85 SGLANG_TURING_ACCEL_FACTOR=2.0

echo "[prof] launch server $(date -u +%H:%M:%S)"
pkill -9 -f "sglang.launch_server|bench_serving" 2>/dev/null; sleep 2
python3 -m sglang.launch_server --model-path "$MODEL" --tp 8 --trust-remote-code \
  --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85 \
  --enable-hierarchical-cache --hicache-size 96 --page-size 64 \
  --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through \
  --enable-metrics --enable-cache-report --port "$PORT" > "$PROFDIR/server.log" 2>&1 &
SRV=$!
for i in $(seq 1 1200); do
  curl -sf localhost:$PORT/health >/dev/null 2>&1 && { echo "[prof] ready ~$((i*3))s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "[prof] SERVER_DIED"; tail -40 "$PROFDIR/server.log"; exit 3; }
  sleep 3
done
curl -sf localhost:$PORT/health >/dev/null 2>&1 || { echo "[prof] SERVER_TIMEOUT"; exit 3; }

echo "[prof] start saturation load (λ=10) $(date -u +%H:%M:%S)"
python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
  --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
  --request-rate 10 --max-concurrency 256 --num-prompts 1553 --port "$PORT" > "$PROFDIR/load.log" 2>&1 &
LOAD=$!

echo "[prof] wait 160s for saturation $(date -u +%H:%M:%S)"; sleep 160
echo "[prof] running-req snapshot:"; grep -oE "#running-req: [0-9]+" "$PROFDIR/server.log" | tail -3
echo "[prof] trigger profile_by_stage (40 steps/stage, GPU) $(date -u +%H:%M:%S)"
curl -s -X POST localhost:$PORT/start_profile -H "Content-Type: application/json" \
  -d '{"num_steps":40,"profile_by_stage":true,"activities":["GPU"]}' | tee "$PROFDIR/profstart.log"; echo

echo "[prof] wait for BOTH stages (gate on DECODE, which exports AFTER EXTEND)"
for i in $(seq 1 180); do
  ls "$PROFDIR"/*DECODE*.trace.json* >/dev/null 2>&1 && { echo "[prof] DECODE traces present ($((i*5))s)"; break; }
  ls "$PROFDIR"/*EXTEND*.trace.json* >/dev/null 2>&1 && echo "[prof] EXTEND present, awaiting DECODE... ($((i*5))s)"
  sleep 5
done
sleep 30   # let DECODE export flush fully
echo "[prof] cleanup $(date -u +%H:%M:%S)"; kill $LOAD $SRV 2>/dev/null; sleep 3; pkill -9 -f "sglang.launch_server|bench_serving" 2>/dev/null
echo "[prof] DONE $(date -u +%H:%M:%S)"; ls -la "$PROFDIR"
