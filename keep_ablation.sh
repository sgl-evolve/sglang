#!/usr/bin/env bash
# Same-node keep_hits sensitivity ablation for BM_EXCL: keep_hits=1 (win) vs keep_hits=2
# (predicted-worse) at the per-version rate λ=3. Grabs ONE held certified node's flock and
# holds it across both phases so the comparison is on the identical node/session.
# Reports hit (via BM_DIAG unbiased prefill hit) + p99/median/req_s. Usage: keep_ablation.sh <tag> [node]
set -uo pipefail
ROOT="/home/junyanch_google_com/autoresearch"
WORK="$ROOT/workspace/sgl/v0.25_ablations/base_mech/researchers/base_mech"
POOL="${SGL_POOL_DIR:-$ROOT/workspace/sgl/v0.25_ablations/_pool}"
MODEL=Qwen/Qwen3.5-122B-A10B-FP8
MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
TAG="${1:?usage: keep_ablation.sh <tag> [node]}"; PREF="${2:-}"
OUT="$WORK/runs/keepabl-$TAG"; mkdir -p "$OUT"
PORT=31900
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
held_nodes(){ compgen -G "$POOL/held/*" >/dev/null 2>&1 && for f in "$POOL/held"/*; do basename "$f"; done; }

run_phase(){  # $1=node $2=jid $3=keep
  local node="$1" jid="$2" keep="$3"
  srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash -c "
set -uo pipefail
export BM_EXCL=1 BM_EXCL_KEEP_HITS=$keep
export HF_TOKEN='$HF_API_KEY' HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
source '$WORK/.venv/bin/activate'; export PYTHONPATH='$WORK/python'; cd '$WORK'
export TRITON_CACHE_DIR='$WORK/.cache/triton' CUDA_CACHE_PATH='$WORK/.cache/nv' FLASHINFER_CACHE_DIR='$WORK/.cache/flashinfer' XDG_CACHE_HOME='$WORK/.cache'
python3 -m sglang.launch_server --model-path '$MODEL' --tp 8 --trust-remote-code \
  --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85 \
  --enable-hierarchical-cache --hicache-size 96 --page-size 64 \
  --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through \
  --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion \
  --enable-metrics --enable-cache-report --port $PORT > '$OUT/server-keep$keep.log' 2>&1 &
SRV=\$!
trap 'kill \$SRV 2>/dev/null; sleep 3; pkill -9 -f sglang.launch_server 2>/dev/null' EXIT
for i in \$(seq 1 900); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && { echo READY-keep$keep \$((i*3))s; break; }; kill -0 \$SRV 2>/dev/null || { echo SERVER_DIED-keep$keep; tail -25 '$OUT/server-keep$keep.log'; exit 3; }; sleep 3; done
curl -sf localhost:$PORT/health >/dev/null 2>&1 || { echo SERVER_TIMEOUT-keep$keep; exit 3; }
curl -s -X POST localhost:$PORT/flush_cache >/dev/null 2>&1; sleep 5
echo \">>> keep$keep rate=3\"
python3 benchmark/hicache/bench_serving.py --backend sglang --model '$MODEL' \
  --dataset-name loogle --dataset-path '$MIX' --enable-multiturn --disable-shuffle \
  --request-rate 3 --max-concurrency 128 --num-prompts 1553 --port $PORT \
  --output-file '$OUT/keep$keep-r3.json' 2>&1 | grep -E 'P99 TTFT|Median TTFT|Mean TTFT|Request throughput|Successful' | sed \"s/^/[keep$keep] /\"
kill \$SRV 2>/dev/null; sleep 5; pkill -9 -f sglang.launch_server 2>/dev/null; sleep 5
echo PHASE_DONE-keep$keep
"
}

while :; do
  for node in $(held_nodes); do
    [ -n "$PREF" ] && [ "$node" != "$PREF" ] && continue
    jid=$(cat "$POOL/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    exec 200>"$POOL/locks/$node.lock"
    if flock -n 200; then
      echo "[keepabl] got $node (hold $jid); DRAM check..."
      ok=0; for i in $(seq 1 8); do d=$(srun --jobid="$jid" --overlap -N1 -w "$node" awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo 2>/dev/null); echo "  DRAM=${d:-?}G try $i"; [ "${d:-0}" -ge 1300 ] && { ok=1; break; }; sleep 15; done
      if [ "$ok" -eq 1 ]; then
        echo "[keepabl] === keep_hits=1 phase on $node ==="; run_phase "$node" "$jid" 1
        echo "[keepabl] === keep_hits=2 phase on $node ==="; run_phase "$node" "$jid" 2
        flock -u 200; exec 200>&-; echo "[keepabl] done on $node"; exit 0
      fi
      flock -u 200
    fi
    exec 200>&-
  done
  [ -n "$PREF" ] && echo "[keepabl] preferred $PREF not grabbable yet; retrying"
  sleep 20
done
