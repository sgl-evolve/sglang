#!/usr/bin/env bash
# v0.3 FIXED evaluator (DRAFT) — Strata-aligned: full-decode Poisson rate SWEEP -> throughput-latency curve.
# One 2-tier server load (frozen v0.25 config), swept over request-rate; headline = goodput under a TTFT-SLO.
# Usage (on an --exclusive a3 node):  eval.sh <name> <version> [extra policy args...]
set -uo pipefail
ROOT="${AUTORESEARCH_ROOT:-/home/junyanch_google_com/autoresearch}"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
SGL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd 2>/dev/null || echo "$ROOT/programs/sgl/v0.3")"
SGL_WORKSPACE="$ROOT/workspace/sgl/v0.3_ablations/$(basename "$SGL_HOME" 2>/dev/null || echo v0.3)"
NAME="${1:?usage: eval.sh <name> <version> [extra]}"; VER="${2:?usage}"; shift 2 || true; EXTRA=("$@")
WORK=$SGL_WORKSPACE/researchers/$NAME; PORT=${PORT:-30000}; OUT="$WORK/runs/$VER"; mkdir -p "$OUT"

# --- FROZEN load knobs (the contract) ---
RATES="3 5"                # SCREEN: goodput@SLO provably determined by lambda in {3,5} (cap ~3.7 req/s => 7,10 always FAIL). ~2x faster.
NUMP=1553                 # full working set (~19M tok >> L1+L2 ~10.7M) => genuine memory pressure (hit ~0.62,
                          # not the under-provisioned 0.84 a smaller set gives) — so the cache actually binds.
WARMUP_NUMP=300           # a warmup burst first: warms the decode pipeline + fills the cache, so the measured
                          # points are WARM STEADY-STATE, not cold-start-metastable (fixes the bimodal goodput@SLO).
MAXC=256                  # concurrency cap (>= peak so it doesn't clip the sweep)
SLO_MS=8000               # TTFT p99 SLO for the goodput headline

FORBIDDEN='--model-path|--tokenizer|--tp|--context-length|--mem-fraction-static|--hicache-size|--hicache-storage-backend|--enable-hierarchical-cache|--hicache-mem-layout|--chunked-prefill-size|--page-size|--port|--request-rate|--num-prompts|--max-concurrency|--random-output-len|--output-len'
for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf '%s' "$a" | grep -qE "^($FORBIDDEN)(=.*)?$" && { echo "REFUSED '$a' — changes the FROZEN eval. Change engine code."; exit 5; }; done

set -a; source "$ROOT/.env"; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
source "$WORK/.venv/bin/activate"; export PYTHONPATH="$WORK/python"; cd "$WORK"
export TRITON_CACHE_DIR="$WORK/.cache/triton" CUDA_CACHE_PATH="$WORK/.cache/nv" FLASHINFER_CACHE_DIR="$WORK/.cache/flashinfer" XDG_CACHE_HOME="$WORK/.cache"; mkdir -p "$WORK/.cache"
MODEL=Qwen/Qwen3.5-122B-A10B-FP8; MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
COMMIT=$(git -C "$WORK" rev-parse --short HEAD 2>/dev/null || echo "?")
DRAMG=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); [ "${DRAMG:-0}" -lt 1300 ] && { echo "DRAM_TOO_LOW ${DRAMG}G"; exit 2; }

# --- one 2-tier server load (frozen v0.25 flags + EXTRA) ---
LAUNCH=(python3 -m sglang.launch_server --model-path "$MODEL" --tp 8 --trust-remote-code
  --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85
  --enable-hierarchical-cache --hicache-size 96 --page-size 64
  --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through
  --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion
  --enable-metrics --enable-cache-report --port "$PORT")
[ ${#EXTRA[@]} -gt 0 ] && LAUNCH+=("${EXTRA[@]}")
"${LAUNCH[@]}" > "$OUT/server.log" 2>&1 & SRV=$!
trap 'kill $SRV 2>/dev/null; sleep 4; pkill -9 -f sglang.launch_server 2>/dev/null' EXIT
for i in $(seq 1 900); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && { echo "ready ~$((i*3))s"; break; }; kill -0 $SRV 2>/dev/null || { echo SERVER_DIED; tail -40 "$OUT/server.log"; exit 3; }; sleep 3; done
curl -sf localhost:$PORT/health >/dev/null 2>&1 || { echo SERVER_TIMEOUT; exit 3; }

# --- WARMUP: prime the decode pipeline + fill the cache to steady state (kills the cold-start prefill-queue
#     metastability that made goodput@SLO bimodal). Discarded — not a measured point. ---
echo ">>> warmup ($WARMUP_NUMP convs @ rate 3) $(date -u +%H:%M:%S)"
python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
  --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
  --request-rate 3 --max-concurrency "$MAXC" --num-prompts "$WARMUP_NUMP" --port $PORT \
  --output-file "$OUT/bench_warmup.json" > "$OUT/bench_warmup.txt" 2>&1 || true

# --- rate sweep (full decode; WARM STEADY-STATE — NO flush between rates, so the cache stays hot and each
#     point is a realistic steady-state operating point, not cold-start-noise-bound) ---
echo "label,rate,req_throughput,out_tok_s,ttft_p50_ms,ttft_p99_ms,e2e_p99_ms,hit_rate" > "$OUT/curve.csv"
for R in $RATES; do
  echo ">>> rate=$R $(date -u +%H:%M:%S)"
  python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
    --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
    --request-rate "$R" --max-concurrency "$MAXC" --num-prompts "$NUMP" --port $PORT \
    --output-file "$OUT/bench_r$R.json" > "$OUT/bench_r$R.txt" 2>&1
  curl -s localhost:$PORT/metrics > "$OUT/metrics_r$R.txt" 2>/dev/null
  python3 - "$OUT" "$R" >> "$OUT/curve.csv" <<'PY'
import sys,re,os
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
print(f"sweep,{R},{req or ''},{tok or ''},{p50 or ''},{p99 or ''},{e2e or ''},{hit}")
PY
done

# --- headline: goodput @ TTFT-SLO + peak throughput (decode-bound control) -> summary.json ---
python3 - "$OUT" "$VER" "$COMMIT" "$SLO_MS" <<'PY'
import sys,csv,json
o,ver,commit,slo=sys.argv[1],sys.argv[2],sys.argv[3],float(sys.argv[4])
rows=[r for r in csv.DictReader(open(f"{o}/curve.csv")) if r["req_throughput"]]
f=lambda r,k:(float(r[k]) if r.get(k) not in (None,"") else None)
good=max([f(r,"req_throughput") for r in rows if f(r,"ttft_p99_ms") and f(r,"ttft_p99_ms")<=slo] or [0])
peak_tok=max([f(r,"out_tok_s") or 0 for r in rows] or [0]); peak_req=max([f(r,"req_throughput") or 0 for r in rows] or [0])
best_hit=max([f(r,"hit_rate") or 0 for r in rows] or [0])
panel={"overall/goodput_reqs_at_SLO":good,"overall/peak_out_tok_s":peak_tok,"overall/peak_req_s":peak_req,
       "overall/hit_rate":best_hit,"overall/slo_ms":slo}
for r in rows:  # per-rate curve points for the dashboard
    R=r["rate"]; panel[f"curve/req_s@{R}"]=f(r,"req_throughput"); panel[f"curve/ttft_p99@{R}"]=f(r,"ttft_p99_ms")
json.dump({"version":ver,"commit":commit,"panel":panel,"curve":rows},open(f"{o}/summary.json","w"),indent=2)
print(f"GOODPUT@{int(slo)}ms = {good:.2f} req/s | peak {peak_tok:.0f} tok/s | hit {best_hit:.3f}")
PY
echo "==== EVAL_DONE $VER ($COMMIT) — curve.csv + summary.json in $OUT ===="
