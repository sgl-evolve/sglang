#!/usr/bin/env bash
# kleinrock SESSION-VARIANCE diagnostic. Question: does the goodput@SLO regime vary across SERVER LAUNCHES
# on ONE node, controlling the cross-allocation node-state confound flagged in paper §5.5/§7?
# Design: N server sessions back-to-back in ONE exclusive allocation. Each session = launch server ->
# warmup -> K stock lambda=3 draws -> FULLY tear down server -> next session. Same node, no other jobs,
# minimal time gap => isolates launch-level (session) metastability from node heterogeneity AND cross-
# allocation drift (residual: slow thermal drift over the allocation, noted as a caveat). Server flags are
# BYTE-IDENTICAL to eval.sh (stock write_through) so every draw is the frozen eval's lambda=3 point (on-contract).
# Output sessvar.csv (session,rep,p99,...) -> analyze within-session vs across-session variance.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env" 2>/dev/null; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
MAIN=$ROOT/workspace/sgl/v0.31/research/researchers/kleinrock
source "$MAIN/.venv/bin/activate"
export PYTHONPATH="$MAIN/python"; cd "$MAIN"
export TRITON_CACHE_DIR="$MAIN/.cache/triton" CUDA_CACHE_PATH="$MAIN/.cache/nv" FLASHINFER_CACHE_DIR="$MAIN/.cache/flashinfer" XDG_CACHE_HOME="$MAIN/.cache"; mkdir -p "$MAIN/.cache"
OUT="$MAIN/runs/v0-sessvar"; mkdir -p "$OUT"; PORT=${PORT:-30021}
N=${SESS_N:-3}; K=${SESS_K:-3}; RATE=3; NUMP=1553; WARMUP_NUMP=300; MAXC=256; SLO_MS=8000
MODEL=Qwen/Qwen3.5-122B-A10B-FP8; MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
COMMIT=$(git -C "$MAIN" rev-parse --short HEAD 2>/dev/null || echo "?")
NODE=$(hostname -s 2>/dev/null || hostname)
DRAMG=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); [ "${DRAMG:-0}" -lt 1300 ] && { echo "DRAM_TOO_LOW ${DRAMG}G"; exit 2; }
# frozen server flags (BYTE-IDENTICAL to eval.sh) -- STOCK write_through
LAUNCH_FLAGS=(--model-path "$MODEL" --tp 8 --trust-remote-code
  --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85
  --enable-hierarchical-cache --hicache-size 96 --page-size 64
  --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through
  --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion
  --enable-metrics --enable-cache-report --port "$PORT")
SRV=0
cleanup(){ kill ${SRV} 2>/dev/null; sleep 4; pkill -9 -f sglang.launch_server 2>/dev/null; sleep 2; }
trap cleanup EXIT
teardown(){ kill ${SRV} 2>/dev/null; sleep 8; pkill -9 -f sglang.launch_server 2>/dev/null; sleep 16; }  # full GPU free before relaunch
echo "label,node,session,rep,rate,req_throughput,out_tok_s,ttft_p50_ms,ttft_p99_ms,e2e_p99_ms,hit_rate" > "$OUT/sessvar.csv"
echo ">>> [SESSVAR] node=$NODE commit=$COMMIT N=$N K=$K start $(date -u)"

for SESS in $(seq 1 "$N"); do
  echo "==== SESSION $SESS/$N launch $(date -u +%H:%M:%S) ===="
  python3 -m sglang.launch_server "${LAUNCH_FLAGS[@]}" > "$OUT/server_s$SESS.log" 2>&1 & SRV=$!
  ok=0
  for i in $(seq 1 900); do
    curl -sf localhost:$PORT/health >/dev/null 2>&1 && { echo "session $SESS ready ~$((i*3))s"; ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "SESSION $SESS SERVER_DIED"; tail -30 "$OUT/server_s$SESS.log"; break; }
    sleep 3
  done
  [ "$ok" = 1 ] || { echo "SESSION $SESS skipped (no health)"; teardown; continue; }
  # warmup (compute-warm; cache flushed per bench invocation)
  python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
    --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
    --request-rate "$RATE" --max-concurrency "$MAXC" --num-prompts "$WARMUP_NUMP" --port $PORT \
    --output-file "$OUT/warmup_s$SESS.json" > "$OUT/warmup_s$SESS.txt" 2>&1 || true
  for REP in $(seq 1 "$K"); do
    echo ">>> session $SESS rep $REP/$K $(date -u +%H:%M:%S)"
    python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
      --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
      --request-rate "$RATE" --max-concurrency "$MAXC" --num-prompts "$NUMP" --port $PORT \
      --output-file "$OUT/bench_s${SESS}_r${REP}.json" > "$OUT/bench_s${SESS}_r${REP}.txt" 2>&1
    curl -s localhost:$PORT/metrics > "$OUT/metrics_s${SESS}_r${REP}.txt" 2>/dev/null
    python3 - "$OUT" "$SESS" "$REP" "$RATE" "$NODE" >> "$OUT/sessvar.csv" <<'PY'
import sys,re
o,SESS,REP,R,node=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
t=open(f"{o}/bench_s{SESS}_r{REP}.txt",errors="ignore").read(); m=open(f"{o}/metrics_s{SESS}_r{REP}.txt",errors="ignore").read()
g=lambda p:(re.search(p,t) or [None,None])[1]
req=g(r"Request throughput \(req/s\):\s*([0-9.]+)"); tok=g(r"Output token throughput \(tok/s\):\s*([0-9.]+)")
p50=g(r"Median TTFT \(ms\):\s*([0-9.]+)"); p99=g(r"P99 TTFT \(ms\):\s*([0-9.]+)"); e2e=g(r"P99 E2E Latency \(ms\):\s*([0-9.]+)")
def praw(metric,**lab):
  best=None
  for l,v in re.findall(r"(?m)^"+re.escape(metric)+r"\{([^}]*)\}\s+([0-9.eE+]+)",m):
    if all(f'{k}="{val}"' in l for k,val in lab.items()): best=float(v)
  return best
def sraw(metric):
  vs=re.findall(r"(?m)^"+re.escape(metric)+r"\{[^}]*\}\s+([0-9.eE+]+)",m); return sum(float(x) for x in vs) if vs else None
dev=praw("sglang:cached_tokens_total",cache_source="device") or 0; host=praw("sglang:cached_tokens_total",cache_source="host") or 0
prm=sraw("sglang:prompt_tokens_total"); hit=round((dev+host)/prm,4) if prm else ""
print(f"sessvar,{node},{SESS},{REP},{R},{req or ''},{tok or ''},{p50 or ''},{p99 or ''},{e2e or ''},{hit}")
PY
    tail -1 "$OUT/sessvar.csv"
  done
  echo "==== SESSION $SESS done; tearing down $(date -u +%H:%M:%S) ===="
  teardown
done
echo "==== SESSVAR_DONE $(date -u) — sessvar.csv in $OUT ===="
