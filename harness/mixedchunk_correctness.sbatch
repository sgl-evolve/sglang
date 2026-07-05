#!/usr/bin/env bash
#SBATCH -p a3
#SBATCH -N1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH -t 2:40:00
#SBATCH -J mixchk-drift-3e7
# drift-3e7 mixed-chunk OUTPUT-CORRECTNESS harness (NOT the fixed eval; not on the curve).
# Runs on ANY healthy a3 node (submit-gpu-job: non-eval GPU work -> any node). Disk-independent
# (no L3 storage tier) so it is immune to the /mnt/localssd 1.8T gate blocking the eval pool.
# Launches the 122B model TWICE on one node -- mixed-chunk OFF then ON -- and diffs greedy
# outputs to certify the v19 hybrid-SSM mixed-chunk fix is lossless (no request failures + no
# corruption vs the mix-inactive reference) before it is ever run through the full protocol.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
WORK=$ROOT/workspace/sgl/researchers/drift-3e7
source "$WORK/.venv/bin/activate"; export PYTHONPATH="$WORK/python"; cd "$WORK"

# Isolated JIT caches (warmed by rewarm_cache.sh) so model load is fast + collision-safe.
export TRITON_CACHE_DIR="$WORK/.cache/triton" CUDA_CACHE_PATH="$WORK/.cache/nv" \
       FLASHINFER_CACHE_DIR="$WORK/.cache/flashinfer" XDG_CACHE_HOME="$WORK/.cache" \
       FLASHINFER_WORKSPACE_BASE="$WORK"
mkdir -p "$WORK/.cache"
export SGLANG_DRIFT_MIXCHUNK_TRACE=1   # confirm MIXED batches actually form (test validity)

MODEL=Qwen/Qwen3.5-122B-A10B-FP8
OUT="$WORK/runs/_mixchunk_correctness"; mkdir -p "$OUT"
CTX=65536; CHUNK=2048; MNT=256

echo ">>> mixed-chunk correctness harness on $(hostname) at $(date +%T)"

launch() {  # $1=port  $2=logfile  $3...=extra flags
  local port="$1" log="$2"; shift 2
  python3 -m sglang.launch_server \
    --model-path "$MODEL" --tp 8 --trust-remote-code --dist-timeout 5400 \
    --context-length "$CTX" --chunked-prefill-size "$CHUNK" --mem-fraction-static 0.85 \
    --log-level info --port "$port" "$@" > "$log" 2>&1 &
  echo $!
}
wait_ready() {  # $1=port $2=pid $3=log
  local port="$1" pid="$2" log="$3" i
  for i in $(seq 1 2000); do
    if curl -sf "localhost:$port/health" >/dev/null 2>&1; then echo ">>> ready port $port after ~$((i*3))s"; return 0; fi
    kill -0 "$pid" 2>/dev/null || { echo ">>> SERVER DIED (port $port):"; tail -40 "$log"; return 3; }
    sleep 3
  done
  echo ">>> TIMEOUT port $port"; tail -40 "$log"; return 4
}
killsrv() { kill "$1" 2>/dev/null; sleep 3; pkill -9 -f sglang.launch_server 2>/dev/null; sleep 5; }
trap 'pkill -9 -f sglang.launch_server 2>/dev/null' EXIT

# ---------- 1) baseline: mixed-chunk OFF ----------
echo ">>> [1/2] launching baseline (mixed-chunk OFF) :31401"
PID=$(launch 31401 "$OUT/server_baseline.log")
wait_ready 31401 "$PID" "$OUT/server_baseline.log" || exit 3
python3 harness/mixedchunk_client.py --port 31401 --tag baseline --max-new-tokens "$MNT" --out "$OUT/baseline.json"
killsrv "$PID"

# ---------- 2) mixed-chunk ON ----------
echo ">>> [2/2] launching mixed (mixed-chunk ON) :31402"
PID=$(launch 31402 "$OUT/server_mixed.log" --enable-mixed-chunk)
wait_ready 31402 "$PID" "$OUT/server_mixed.log" || exit 3
python3 harness/mixedchunk_client.py --port 31402 --tag mixed --max-new-tokens "$MNT" --out "$OUT/mixed.json"
killsrv "$PID"

# ---------- confirm MIXED batches fired ----------
NMIX=$(grep -c "MIXCHUNK-TRACE" "$OUT/server_mixed.log" 2>/dev/null || echo 0)
echo ">>> MIXED batches formed (mix_with_running fired): $NMIX times"
if [ "$NMIX" -eq 0 ]; then
  echo ">>> WARNING: no MIXED batches formed -> correctness test is INCONCLUSIVE (raise concurrency / lower chunk size)"
fi

# ---------- verdict ----------
echo ">>> comparing..."
python3 harness/mixedchunk_compare.py "$OUT/baseline.json" "$OUT/mixed.json"
RC=$?
echo ">>> MIXCHUNK_CORRECTNESS_DONE rc=$RC nmix=$NMIX"
exit $RC
