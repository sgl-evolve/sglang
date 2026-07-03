#!/usr/bin/env bash
# diag_capture.sh — is CUDA-graph capture (flashinfer allreduce / NCCL) systemic or node-flaky?
# Launches the 122B server with the eval's model/tp/capture config but NO hicache (so no
# 1.8TB disk needed), on whatever node this runs on, and reports whether it reaches /health
# (capture passed) or crashes/hangs. NOT a logged eval — pure diagnostic.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
WORK=$ROOT/workspace/sgl/researchers/quill-7m3
set -a; source "$ROOT/.env"; set +a; export HF_TOKEN="${HF_API_KEY:-}"
source "$WORK/.venv/bin/activate"; export PYTHONPATH="$WORK/python"
export HF_HOME=/rmeng_data/junyanch-data/hf_cache
PORT=31007
LOG=$WORK/diag_server.log
echo "diag on $(hostname) @ $(date +%H:%M:%S)"
nvidia-smi --query-gpu=index,name --format=csv,noheader | head -8
python3 -m sglang.launch_server --model-path Qwen/Qwen3.5-122B-A10B-FP8 --tp 8 \
  --trust-remote-code --context-length 262144 --chunked-prefill-size 6144 \
  --mem-fraction-static 0.85 --page-size 64 --enable-metrics --port $PORT > "$LOG" 2>&1 &
SRV=$!
res="TIMEOUT"
for i in $(seq 1 700); do
  if curl -sf "localhost:$PORT/health" >/dev/null 2>&1; then res="CAPTURE_PASS(~$((i*3))s)"; break; fi
  kill -0 $SRV 2>/dev/null || { res="SERVER_DIED(check diag_server.log)"; break; }
  sleep 3
done
echo "DIAG_RESULT: $res @ $(date +%H:%M:%S)"
grep -iE "Bus error|Traceback|Guessing device|fired up|SIGBUS|died" "$LOG" 2>/dev/null | tail -8
kill $SRV 2>/dev/null; sleep 3; pkill -9 -f sglang.launch_server 2>/dev/null || true
echo "diag done @ $(date +%H:%M:%S)"
