#!/usr/bin/env bash
# kleinrock PHASE-MAP diagnostic (OFF-CONTRACT rate points, like warmdiag/medk — does NOT touch the headline
# eval). Purpose: empirically map the LOWER half of the coin-flip phase diagram on the REAL system. The fixed
# eval's rate set starts at lambda=3 (coin-flip) and lambda>=5 reliably fails; the reliable-PASS regime below
# lambda=3 predicted by the minimal model (§5.5) is never measured. Here we sweep LOW lambda in {1.5, 2.5},
# K reps each, same node/server (stock, byte-identical flags), to show reliable-pass at low load and the
# transition into the coin-flip band — completing the empirical phase diagram (pass -> coin-flip -> fail).
# Each draw is a full NUMP=1553 run at that lambda (cold-flush per bench, faithful to how the eval measures a
# rate point) — only the RATE differs from the contract, and it is a labeled diagnostic, not the headline metric.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env" 2>/dev/null; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
MAIN=$ROOT/workspace/sgl/v0.31/research/researchers/kleinrock
source "$MAIN/.venv/bin/activate"
export PYTHONPATH="$MAIN/python"; cd "$MAIN"
export TRITON_CACHE_DIR="$MAIN/.cache/triton" CUDA_CACHE_PATH="$MAIN/.cache/nv" FLASHINFER_CACHE_DIR="$MAIN/.cache/flashinfer" XDG_CACHE_HOME="$MAIN/.cache"; mkdir -p "$MAIN/.cache"
OUT="$MAIN/runs/v0-phasemap"; mkdir -p "$OUT"; PORT=${PORT:-30031}
LAMBDAS=${PHASE_LAMBDAS:-"1.5 2.5"}; K=${PHASE_K:-3}; NUMP=1553; WARMUP_NUMP=300; MAXC=256; SLO_MS=8000
MODEL=Qwen/Qwen3.5-122B-A10B-FP8; MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
COMMIT=$(git -C "$MAIN" rev-parse --short HEAD 2>/dev/null || echo "?"); NODE=$(hostname -s 2>/dev/null || hostname)
DRAMG=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); [ "${DRAMG:-0}" -lt 1300 ] && { echo "DRAM_TOO_LOW ${DRAMG}G"; exit 2; }
# frozen server flags (BYTE-IDENTICAL to eval.sh) -- STOCK write_through
LAUNCH=(python3 -m sglang.launch_server --model-path "$MODEL" --tp 8 --trust-remote-code
  --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85
  --enable-hierarchical-cache --hicache-size 96 --page-size 64
  --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through
  --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion
  --enable-metrics --enable-cache-report --port "$PORT")
"${LAUNCH[@]}" > "$OUT/server.log" 2>&1 & SRV=$!
trap 'kill $SRV 2>/dev/null; sleep 4; pkill -9 -f sglang.launch_server 2>/dev/null' EXIT
for i in $(seq 1 900); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && { echo "ready ~$((i*3))s"; break; }; kill -0 $SRV 2>/dev/null || { echo SERVER_DIED; tail -40 "$OUT/server.log"; exit 3; }; sleep 3; done
curl -sf localhost:$PORT/health >/dev/null 2>&1 || { echo SERVER_TIMEOUT; exit 3; }
echo ">>> [PHASEMAP] node=$NODE commit=$COMMIT lambdas=[$LAMBDAS] K=$K warmup $(date -u +%H:%M:%S)"
python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
  --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
  --request-rate 3 --max-concurrency "$MAXC" --num-prompts "$WARMUP_NUMP" --port $PORT \
  --output-file "$OUT/bench_warmup.json" > "$OUT/bench_warmup.txt" 2>&1 || true
echo "label,node,lambda,rep,req_throughput,out_tok_s,ttft_p50_ms,ttft_p99_ms,e2e_p99_ms,hit_rate" > "$OUT/phasemap.csv"
for LAM in $LAMBDAS; do
  for REP in $(seq 1 "$K"); do
    tag="l${LAM}_r${REP}"
    echo ">>> [PHASEMAP] lambda=$LAM rep=$REP/$K $(date -u +%H:%M:%S)"
    python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
      --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
      --request-rate "$LAM" --max-concurrency "$MAXC" --num-prompts "$NUMP" --port $PORT \
      --output-file "$OUT/bench_${tag}.json" > "$OUT/bench_${tag}.txt" 2>&1
    curl -s localhost:$PORT/metrics > "$OUT/metrics_${tag}.txt" 2>/dev/null
    python3 - "$OUT" "$tag" "$LAM" "$REP" "$NODE" >> "$OUT/phasemap.csv" <<'PY'
import sys,re
o,tag,LAM,REP,node=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
t=open(f"{o}/bench_{tag}.txt",errors="ignore").read(); m=open(f"{o}/metrics_{tag}.txt",errors="ignore").read()
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
print(f"phasemap,{node},{LAM},{REP},{req or ''},{tok or ''},{p50 or ''},{p99 or ''},{e2e or ''},{hit}")
PY
    tail -1 "$OUT/phasemap.csv"
  done
done
echo "==== PHASEMAP_DONE $(date -u) — phasemap.csv in $OUT ===="
