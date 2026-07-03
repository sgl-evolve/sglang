#!/usr/bin/env bash
# diag_warmup.sh — distinguish SLOW-COLD-COMPILE vs true HANG, and warm the JIT cache.
# Launches the 122B server (eval model/tp/capture config, NO hicache) with a LONG (75 min)
# readiness wait, logging GPU util + JIT cache growth every ~90s. If it reaches /health, the
# earlier "hangs" were just cold JIT compile exceeding eval.sh's 45min timeout, and the shared
# ~/.cache is now warm -> real evals will start fast. If it never progresses (cache static,
# GPU 0%, no compilers) it's a true hang.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
WORK=$ROOT/workspace/sgl/researchers/quill-7m3
set -a; source "$ROOT/.env"; set +a; export HF_TOKEN="${HF_API_KEY:-}"
source "$WORK/.venv/bin/activate"; export PYTHONPATH="$WORK/python"
export HF_HOME=/rmeng_data/junyanch-data/hf_cache
PORT=31009; LOG=$WORK/diag_warmup_server.log
echo "warmup diag on $(hostname) @ $(date +%H:%M:%S)"
python3 -m sglang.launch_server --model-path Qwen/Qwen3.5-122B-A10B-FP8 --tp 8 \
  --trust-remote-code --context-length 262144 --chunked-prefill-size 6144 \
  --mem-fraction-static 0.85 --page-size 64 --enable-metrics --port $PORT > "$LOG" 2>&1 &
SRV=$!
res="TIMEOUT_75min"
for i in $(seq 1 1500); do
  if curl -sf "localhost:$PORT/health" >/dev/null 2>&1; then res="READY(~$((i*3))s = $((i*3/60))min)"; break; fi
  kill -0 $SRV 2>/dev/null || { res="SERVER_DIED"; break; }
  if (( i % 30 == 0 )); then
    gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | paste -sd, -)
    comp=$(ps -eo comm | grep -icE "cicc|ptxas|cc1|nvcc" || true)
    dg=$(find ~/.cache/deep_gemm -type f 2>/dev/null | wc -l)
    tri=$(du -sm ~/.triton 2>/dev/null | cut -f1)
    echo "[$((i*3/60))min @ $(date +%H:%M:%S)] gpu=[$gpu] compilers=$comp deepgemm_files=$dg triton_MB=$tri"
  fi
  sleep 3
done
echo "WARMUP_RESULT: $res @ $(date +%H:%M:%S)"
grep -iE "Bus error|Traceback|fired up|DeepGEMM warmup|Capture" "$LOG" 2>/dev/null | tail -6
kill $SRV 2>/dev/null; sleep 3; pkill -9 -f sglang.launch_server 2>/dev/null || true
echo "warmup diag done @ $(date +%H:%M:%S)"
