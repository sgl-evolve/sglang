#!/usr/bin/env bash
# kleinrock WARM-STEADY-STATE DIAGNOSTIC (OFF-CONTRACT — not the frozen eval).
# Identical to the frozen eval.sh EXCEPT the per-rate cache flush is disabled (KLEINROCK_NOFLUSH=1),
# so the cache persists across rates -> measures the TRUE warm-steady-state goodput@SLO. This isolates
# the per-rate-flush cold-start's contribution to goodput@SLO=0 (resolves flush-vs-structural).
# Runs the WARMDIAG WORKTREE's code (stock python + no-flush bench). Output -> runs/v0-warmdiag/.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env" 2>/dev/null; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
MAIN=$ROOT/workspace/sgl/v0.31/research/researchers/kleinrock
WT=$ROOT/workspace/sgl/v0.31/research/kleinrock-warmdiag-wt
source "$MAIN/.venv/bin/activate"
export PYTHONPATH="$WT/python"; cd "$WT"
export KLEINROCK_NOFLUSH=1     # <<< the diagnostic knob: disable per-rate flush (warm steady-state)
export TRITON_CACHE_DIR="$MAIN/.cache/triton" CUDA_CACHE_PATH="$MAIN/.cache/nv" FLASHINFER_CACHE_DIR="$MAIN/.cache/flashinfer" XDG_CACHE_HOME="$MAIN/.cache"; mkdir -p "$MAIN/.cache"
OUT="$MAIN/runs/v0-warmdiag"; mkdir -p "$OUT"; PORT=${PORT:-30001}
RATES="3 5 7 10"; NUMP=1553; WARMUP_NUMP=300; MAXC=256; SLO_MS=8000
MODEL=Qwen/Qwen3.5-122B-A10B-FP8; MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
COMMIT=$(git -C "$WT" rev-parse --short HEAD 2>/dev/null || echo "?")
DRAMG=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); [ "${DRAMG:-0}" -lt 1300 ] && { echo "DRAM_TOO_LOW ${DRAMG}G"; exit 2; }
# frozen server flags (identical to eval.sh)
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
echo ">>> [WARMDIAG] warmup ($WARMUP_NUMP convs @ rate 3) $(date -u +%H:%M:%S)  (KLEINROCK_NOFLUSH=$KLEINROCK_NOFLUSH)"
python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
  --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
  --request-rate 3 --max-concurrency "$MAXC" --num-prompts "$WARMUP_NUMP" --port $PORT \
  --output-file "$OUT/bench_warmup.json" > "$OUT/bench_warmup.txt" 2>&1 || true
echo "label,rate,req_throughput,out_tok_s,ttft_p50_ms,ttft_p99_ms,e2e_p99_ms,hit_rate" > "$OUT/curve.csv"
for R in $RATES; do
  echo ">>> [WARMDIAG] rate=$R $(date -u +%H:%M:%S)"
  python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
    --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
    --request-rate "$R" --max-concurrency "$MAXC" --num-prompts "$NUMP" --port $PORT \
    --output-file "$OUT/bench_r$R.json" > "$OUT/bench_r$R.txt" 2>&1
  curl -s localhost:$PORT/metrics > "$OUT/metrics_r$R.txt" 2>/dev/null
  python3 - "$OUT" "$R" >> "$OUT/curve.csv" <<'PY'
import sys,re
o,R=sys.argv[1],sys.argv[2]; t=open(f"{o}/bench_r{R}.txt",errors="ignore").read(); m=open(f"{o}/metrics_r{R}.txt",errors="ignore").read()
g=lambda p:(re.search(p,t) or [None,None])[1]
req=g(r"Request throughput \(req/s\):\s*([0-9.]+)"); tok=g(r"Output token throughput \(tok/s\):\s*([0-9.]+)")
p50=g(r"Median TTFT \(ms\):\s*([0-9.]+)"); p99=g(r"P99 TTFT \(ms\):\s*([0-9.]+)"); e2e=g(r"P99 E2E Latency \(ms\):\s*([0-9.]+)")
def praw(metric,**lab):
  for l,v in re.findall(r"(?m)^"+re.escape(metric)+r"\{([^}]*)\}\s+([0-9.eE+]+)",m):
    if all(f'{k}="{val}"' in l for k,val in lab.items()): return float(v)
def sraw(metric):
  vs=re.findall(r"(?m)^"+re.escape(metric)+r"\{[^}]*\}\s+([0-9.eE+]+)",m); return sum(float(x) for x in vs) if vs else None
dev=praw("sglang:cached_tokens_total",cache_source="device") or 0; host=praw("sglang:cached_tokens_total",cache_source="host") or 0
prm=sraw("sglang:prompt_tokens_total"); hit=round((dev+host)/prm,4) if prm else ""
print(f"warmdiag,{R},{req or ''},{tok or ''},{p50 or ''},{p99 or ''},{e2e or ''},{hit}")
PY
done
python3 - "$OUT" v0-warmdiag "$COMMIT" "$SLO_MS" <<'PY'
import sys,csv,json
o,ver,commit,slo=sys.argv[1],sys.argv[2],sys.argv[3],float(sys.argv[4])
rows=[r for r in csv.DictReader(open(f"{o}/curve.csv")) if r["req_throughput"]]
f=lambda r,k:(float(r[k]) if r.get(k) not in (None,"") else None)
good=max([f(r,"req_throughput") for r in rows if f(r,"ttft_p99_ms") and f(r,"ttft_p99_ms")<=slo] or [0])
panel={"overall/goodput_reqs_at_SLO":good,"overall/slo_ms":slo}
for r in rows: R=r["rate"]; panel[f"curve/req_s@{R}"]=f(r,"req_throughput"); panel[f"curve/ttft_p99@{R}"]=f(r,"ttft_p99_ms")
json.dump({"version":ver,"commit":commit,"panel":panel,"curve":rows},open(f"{o}/summary.json","w"),indent=2)
print(f"[WARMDIAG] GOODPUT@{int(slo)}ms = {good:.2f} req/s (NO per-rate flush)")
PY
echo "==== WARMDIAG_DONE — curve.csv + summary.json in $OUT ===="
