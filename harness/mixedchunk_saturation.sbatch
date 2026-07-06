#!/usr/bin/env bash
#SBATCH -p a3
#SBATCH -N1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH -t 1:20:00
#SBATCH -J mixsat-drift-3e7
# drift-3e7 mixed-chunk SATURATION scan (non-eval fast-scan; ANY a3 node). Unlike the lambda=3.5
# throughput scan (which the server kept up with -> no max-throughput signal), this OVERLOADS the
# server (request-rate >> max throughput) so throughput == the server's ceiling. Measures whether
# mixed-chunk RAISES the max throughput ceiling (the mechanism for a v19 headline win in the eval's
# overloaded/queue-dominated regime). No disk L3 (runs on disk-full nodes); GPU config = eval's.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
WORK=$ROOT/workspace/sgl/researchers/drift-3e7
source "$WORK/.venv/bin/activate"; export PYTHONPATH="$WORK/python"; cd "$WORK"
export TRITON_CACHE_DIR="$WORK/.cache/triton" CUDA_CACHE_PATH="$WORK/.cache/nv" \
       FLASHINFER_CACHE_DIR="$WORK/.cache/flashinfer" XDG_CACHE_HOME="$WORK/.cache" \
       FLASHINFER_WORKSPACE_BASE="$WORK"
mkdir -p "$WORK/.cache"; export SGLANG_DRIFT_MIXCHUNK_TRACE=1
MODEL=Qwen/Qwen3.5-122B-A10B-FP8
MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
OUT="$WORK/runs/_mixchunk_saturation"; mkdir -p "$OUT"
RATE=14; NPROMPTS=300   # rate >> ~3.5 max -> saturate; enough convs to reach steady-state ceiling

launch() { local port="$1" log="$2"; shift 2
  python3 -m sglang.launch_server --model-path "$MODEL" --tp 8 --trust-remote-code --dist-timeout 5400 \
    --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85 \
    --enable-hierarchical-cache --hicache-size 96 --page-size 64 --hicache-io-backend direct \
    --hicache-mem-layout page_first_direct --hicache-write-policy write_through \
    --schedule-policy srpf --radix-eviction-policy lfu \
    --enable-metrics --enable-cache-report --log-level info --port "$port" "$@" > "$log" 2>&1 & echo $!; }
wait_ready() { local port="$1" pid="$2" log="$3" i; for i in $(seq 1 900); do
    curl -sf "localhost:$port/health" >/dev/null 2>&1 && { echo ">>> ready $port (~$((i*3))s)"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo ">>> DIED $port"; grep -aiE "out of memory|Not enough GPU" "$log"|tail -3; return 3; }
    sleep 3; done; echo ">>> TIMEOUT $port"; return 4; }
bench() { python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
    --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
    --request-rate "$RATE" --max-concurrency 128 --num-prompts "$NPROMPTS" --port "$1" > "$OUT/bench_$2.txt" 2>&1; }
killsrv() { kill "$1" 2>/dev/null; sleep 3; pkill -9 -f sglang.launch_server 2>/dev/null; sleep 5; }
trap 'pkill -9 -f sglang.launch_server 2>/dev/null' EXIT

echo ">>> [1/2] saturate mixed OFF"; PID=$(launch 31601 "$OUT/srv_off.log")
if wait_ready 31601 "$PID" "$OUT/srv_off.log"; then bench 31601 off; fi; killsrv "$PID"
echo ">>> [2/2] saturate mixed ON"; PID=$(launch 31602 "$OUT/srv_on.log" --enable-mixed-chunk)
if wait_ready 31602 "$PID" "$OUT/srv_on.log"; then bench 31602 on; fi; killsrv "$PID"

NMIX=$(grep -ac "MIXCHUNK-TRACE" "$OUT/srv_on.log" 2>/dev/null); NMIX=${NMIX:-0}
echo "======= SATURATION COMPARISON (mixed OFF vs ON) ======="
for t in off on; do echo "--- $t ---"; grep -aE "Successful requests:|Request throughput|Output token throughput|Mean TTFT|Median TTFT|P99 TTFT|Benchmark duration" "$OUT/bench_$t.txt" 2>/dev/null; done
echo ">>> MIXSAT_DONE nmix=$NMIX"
