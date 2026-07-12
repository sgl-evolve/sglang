#!/usr/bin/env bash
# kleinrock SAME-NODE MEDIAN-OF-K (write_back) — methodology-validation control (off-contract measurement of the FROZEN eval's noise).
# Purpose: resolve the node confound in the goodput@SLO coin-flip claim. The frozen eval measures the
# lambda=3 point ONCE, cross-node runs are node-confounded (+/-45% node var). Here we launch the STOCK
# frozen server ONCE and re-run the eval's lambda=3 measurement K times BACK-TO-BACK on the SAME node.
# Each bench_serving invocation issues /flush_cache at its start (bench_serving.py:447), so every
# replicate is an i.i.d. COLD-CACHE lambda=3 draw IDENTICAL to the eval's lambda=3 point -- but all on
# one node. Since lambda>=5 reliably fails the SLO, goodput@SLO in {0,3} is decided entirely by lambda=3,
# so the spread of these K p99 values IS the goodput coin-flip, with node held constant.
# Server flags = eval.sh frozen flags EXCEPT --hicache-write-policy write_back (methodology-validation control, matches v1-writeback). write_back is a CONFIG (not a mechanism claim); used only to test whether same-node median-of-k resolves a known config diff.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env" 2>/dev/null; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
MAIN=$ROOT/workspace/sgl/v0.31/research/researchers/kleinrock
source "$MAIN/.venv/bin/activate"
export PYTHONPATH="$MAIN/python"; cd "$MAIN"     # STOCK bench (per-invocation flush = faithful to eval lambda=3)
export TRITON_CACHE_DIR="$MAIN/.cache/triton" CUDA_CACHE_PATH="$MAIN/.cache/nv" FLASHINFER_CACHE_DIR="$MAIN/.cache/flashinfer" XDG_CACHE_HOME="$MAIN/.cache"; mkdir -p "$MAIN/.cache"
OUT="$MAIN/runs/v0-medk-wb"; mkdir -p "$OUT"; PORT=${PORT:-30003}
K=${MEDK_K:-5}; RATE=3; NUMP=1553; WARMUP_NUMP=300; MAXC=256; SLO_MS=8000
MODEL=Qwen/Qwen3.5-122B-A10B-FP8; MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
COMMIT=$(git -C "$MAIN" rev-parse --short HEAD 2>/dev/null || echo "?")
NODE=$(hostname -s 2>/dev/null || hostname)
DRAMG=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); [ "${DRAMG:-0}" -lt 1300 ] && { echo "DRAM_TOO_LOW ${DRAMG}G"; exit 2; }
# frozen server flags + write_back (config control), hicache 96
LAUNCH=(python3 -m sglang.launch_server --model-path "$MODEL" --tp 8 --trust-remote-code
  --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85
  --enable-hierarchical-cache --hicache-size 96 --page-size 64
  --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_back
  --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion
  --enable-metrics --enable-cache-report --port "$PORT")
"${LAUNCH[@]}" > "$OUT/server.log" 2>&1 & SRV=$!
trap 'kill $SRV 2>/dev/null; sleep 4; pkill -9 -f sglang.launch_server 2>/dev/null' EXIT
for i in $(seq 1 900); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && { echo "ready ~$((i*3))s"; break; }; kill -0 $SRV 2>/dev/null || { echo SERVER_DIED; tail -40 "$OUT/server.log"; exit 3; }; sleep 3; done
curl -sf localhost:$PORT/health >/dev/null 2>&1 || { echo SERVER_TIMEOUT; exit 3; }
echo ">>> [MEDK] node=$NODE commit=$COMMIT K=$K  warmup ($WARMUP_NUMP convs @ rate $RATE) $(date -u +%H:%M:%S)"
python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
  --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
  --request-rate "$RATE" --max-concurrency "$MAXC" --num-prompts "$WARMUP_NUMP" --port $PORT \
  --output-file "$OUT/bench_warmup.json" > "$OUT/bench_warmup.txt" 2>&1 || true
echo "label,node,rep,rate,req_throughput,out_tok_s,ttft_p50_ms,ttft_p99_ms,e2e_p99_ms,hit_rate" > "$OUT/medk.csv"
for REP in $(seq 1 "$K"); do
  echo ">>> [MEDK] rep=$REP/$K rate=$RATE $(date -u +%H:%M:%S)"
  python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
    --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
    --request-rate "$RATE" --max-concurrency "$MAXC" --num-prompts "$NUMP" --port $PORT \
    --output-file "$OUT/bench_rep$REP.json" > "$OUT/bench_rep$REP.txt" 2>&1
  curl -s localhost:$PORT/metrics > "$OUT/metrics_rep$REP.txt" 2>/dev/null
  python3 - "$OUT" "$REP" "$RATE" "$NODE" >> "$OUT/medk.csv" <<'PY'
import sys,re
o,REP,R,node=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
t=open(f"{o}/bench_rep{REP}.txt",errors="ignore").read(); m=open(f"{o}/metrics_rep{REP}.txt",errors="ignore").read()
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
print(f"medk,{node},{REP},{R},{req or ''},{tok or ''},{p50 or ''},{p99 or ''},{e2e or ''},{hit}")
PY
  tail -1 "$OUT/medk.csv"
done
# summarize: spread of lambda=3 p99 across K same-node draws -> the goodput coin-flip
python3 - "$OUT" v0-medk "$COMMIT" "$SLO_MS" "$NODE" "$K" <<'PY'
import sys,csv,json,statistics
o,ver,commit,slo,node,K=sys.argv[1],sys.argv[2],sys.argv[3],float(sys.argv[4]),sys.argv[5],int(sys.argv[6])
rows=[r for r in csv.DictReader(open(f"{o}/medk.csv")) if r.get("ttft_p99_ms")]
p99=[float(r["ttft_p99_ms"]) for r in rows]
n_pass=sum(1 for x in p99 if x<=slo); n=len(p99)
panel={"medk/node":node,"medk/K_requested":K,"medk/n":n,"medk/slo_ms":slo,
       "medk/p99_min":min(p99) if p99 else None,"medk/p99_median":statistics.median(p99) if p99 else None,
       "medk/p99_max":max(p99) if p99 else None,
       "medk/p99_mean":round(statistics.mean(p99),1) if p99 else None,
       "medk/p99_stdev":round(statistics.pstdev(p99),1) if len(p99)>1 else None,
       "medk/n_pass_SLO":n_pass,"medk/n_fail_SLO":n-n_pass,
       # goodput@SLO per replicate = 3.0 if lambda=3 passes else 0 (lambda>=5 reliably fails)
       "medk/goodput_median":(3.0 if (statistics.median(p99)<=slo) else 0.0) if p99 else None,
       "medk/coin_flip": (0<n_pass<n)}
json.dump({"version":ver,"commit":commit,"node":node,"panel":panel,"rows":rows},open(f"{o}/summary.json","w"),indent=2)
print(f"[MEDK] node={node} n={n}/{K}  p99(ms) min/med/max = "
      f"{min(p99) if p99 else '-'}/{statistics.median(p99) if p99 else '-'}/{max(p99) if p99 else '-'}  "
      f"pass_SLO={n_pass}/{n}  COIN_FLIP={0<n_pass<n}")
PY
echo "==== MEDK_DONE — medk.csv + summary.json in $OUT ===="
