#!/usr/bin/env bash
#SBATCH -p a3
#SBATCH -N1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH -t 1:40:00
#SBATCH -J mixthru-drift-3e7
# drift-3e7 mixed-chunk THROUGHPUT scan (NOT the fixed eval; not on the curve; fast-scan per
# submit-gpu-job -> ANY a3 node). Two purposes while the certified pool is starved by quartz-7m3:
#  (1) confirm mixed-chunk LOADS at the eval's full GPU config (ctx 262144, mem-frac 0.85) on an
#      EXCLUSIVE node -> proves the v19 OOM was contention (quartz co-tenant), not a mixed-chunk cost.
#  (2) measure the throughput delta (mixed OFF vs ON) that underpins the v19 hypothesis
#      (mixed-chunk -> higher throughput -> shorter queue -> lower mean TTFT in the overloaded regime).
# Disk L3 tier omitted (runs on disk-full nodes too); GPU-memory config identical to the eval, so the
# mamba-cache OOM test is faithful. Reduced num-prompts for speed. Same lambda/concurrency as the eval.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
WORK=$ROOT/workspace/sgl/researchers/drift-3e7
source "$WORK/.venv/bin/activate"; export PYTHONPATH="$WORK/python"; cd "$WORK"
export TRITON_CACHE_DIR="$WORK/.cache/triton" CUDA_CACHE_PATH="$WORK/.cache/nv" \
       FLASHINFER_CACHE_DIR="$WORK/.cache/flashinfer" XDG_CACHE_HOME="$WORK/.cache" \
       FLASHINFER_WORKSPACE_BASE="$WORK"
mkdir -p "$WORK/.cache"
export SGLANG_DRIFT_MIXCHUNK_TRACE=1

MODEL=Qwen/Qwen3.5-122B-A10B-FP8
MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
OUT="$WORK/runs/_mixchunk_throughput"; mkdir -p "$OUT"
NPROMPTS=250   # reduced from 1553 for a fast scan; same lambda/concurrency regime

echo ">>> mixed-chunk throughput scan on $(hostname) at $(date +%T)"

launch() {  # $1=port $2=log $3...=extra
  local port="$1" log="$2"; shift 2
  python3 -m sglang.launch_server \
    --model-path "$MODEL" --tp 8 --trust-remote-code --dist-timeout 5400 \
    --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85 \
    --enable-hierarchical-cache --hicache-size 96 \
    --page-size 64 --hicache-io-backend direct --hicache-mem-layout page_first_direct \
    --hicache-write-policy write_through \
    --schedule-policy srpf --radix-eviction-policy lfu \
    --enable-metrics --enable-cache-report --log-level info --port "$port" "$@" > "$log" 2>&1 &
  echo $!
}
wait_ready() {  # $1=port $2=pid $3=log
  local port="$1" pid="$2" log="$3" i
  for i in $(seq 1 900); do
    curl -sf "localhost:$port/health" >/dev/null 2>&1 && { echo ">>> ready port $port (~$((i*3))s)"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo ">>> SERVER DIED port $port:"; grep -iE "error|oom|out of memory|mamba" "$log"|tail -6; return 3; }
    sleep 3
  done
  echo ">>> TIMEOUT port $port"; return 4
}
bench() {  # $1=port $2=tag
  local port="$1" tag="$2"
  python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
    --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
    --request-rate 3.5 --max-concurrency 128 --num-prompts "$NPROMPTS" --port "$port" \
    2>&1 | tee "$OUT/bench_$tag.txt"
}
killsrv() { kill "$1" 2>/dev/null; sleep 3; pkill -9 -f sglang.launch_server 2>/dev/null; sleep 5; }
trap 'pkill -9 -f sglang.launch_server 2>/dev/null' EXIT

# ---- mixed-chunk OFF ----
echo ">>> [1/2] baseline (mixed OFF)"
PID=$(launch 31501 "$OUT/srv_off.log")
if wait_ready 31501 "$PID" "$OUT/srv_off.log"; then bench 31501 off; else echo ">>> OFF load failed"; fi
killsrv "$PID"

# ---- mixed-chunk ON (also the full-config exclusive-node OOM test) ----
echo ">>> [2/2] mixed (mixed ON) — also confirms full-config load on exclusive node"
PID=$(launch 31502 "$OUT/srv_on.log" --enable-mixed-chunk)
if wait_ready 31502 "$PID" "$OUT/srv_on.log"; then bench 31502 on; else echo ">>> ON load FAILED (mamba OOM at full config even exclusive?)"; fi
killsrv "$PID"

NMIX=$(grep -c "MIXCHUNK-TRACE" "$OUT/srv_on.log" 2>/dev/null); NMIX=${NMIX:-0}
echo ">>> MIXED batches fired: $NMIX"
echo "=========== THROUGHPUT COMPARISON (mixed OFF vs ON) ==========="
for t in off on; do
  echo "--- $t ---"
  grep -iE "Request throughput|Output token throughput|Mean TTFT|Median TTFT|P99 TTFT|Successful requests|Benchmark duration" "$OUT/bench_$t.txt" 2>/dev/null
done
echo ">>> MIXCHUNK_THROUGHPUT_DONE nmix=$NMIX"
