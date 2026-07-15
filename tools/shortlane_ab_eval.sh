#!/usr/bin/env bash
# kleinrock SHORT-LANE A/B (Paper 2) — same-node paired comparison of stock (reserve=0) vs the reserved
# short-prefill lane (--prefill-short-lane-reserve N). Tests whether holding back N chunk-budget tokens from an
# in-flight chunked mega-doc, so short waiting reqs prefill alongside it, mitigates prefill head-of-line blocking
# and shifts the p99 TTFT / goodput. Both arms run the SAME shortlane binary; the ONLY difference is the reserve
# flag (opt-in mechanism; reserve=0 => byte-identical stock => lossless-by-default). Per my methodology: same
# node, warmup->steady-state, median-of-k, report the p99 DISTRIBUTION + goodput-pass-count. FAST SCREEN first
# (lambda=5, the decisive HOL rate: stock reliably-fails; a reliable pass under reserve = clean signal).
# ISOLATED flashinfer/triton cache (.cache-shortlane) so a concurrent SRPF eval on another node does NOT race
# the shared JIT cache (see sgl-flashinfer-jit-race).
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env" 2>/dev/null; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
MAIN=$ROOT/workspace/sgl/v0.31/research/researchers/kleinrock
SLWT=$ROOT/workspace/sgl/v0.31/research/researchers/kleinrock-shortlane-wt      # engine: main + short-lane
source "$MAIN/.venv/bin/activate"
export PYTHONPATH="$SLWT/python"
cd "$SLWT"
CACHE="$MAIN/${SL_CACHE:-.cache-shortlane}"; mkdir -p "$CACHE"
export TRITON_CACHE_DIR="$CACHE/triton" CUDA_CACHE_PATH="$CACHE/nv" FLASHINFER_CACHE_DIR="$CACHE/flashinfer" XDG_CACHE_HOME="$CACHE"
OUT="$MAIN/${SL_OUT:-runs/v4-shortlane-ab}"; mkdir -p "$OUT"; PORT=${PORT:-30041}
NUMP=1553; WARMUP_NUMP=300; MAXC=256; SLO_MS=8000
MODEL=Qwen/Qwen3.5-122B-A10B-FP8; MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
NODE=$(hostname -s 2>/dev/null || hostname)
SL_COMMIT=$(git -C "$SLWT" rev-parse --short HEAD 2>/dev/null || echo "?")
DRAMG=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); [ "${DRAMG:-0}" -lt 1300 ] && { echo "DRAM_TOO_LOW ${DRAMG}G"; exit 2; }

BASE_LAUNCH=(python3 -m sglang.launch_server --model-path "$MODEL" --tp 8 --trust-remote-code
  --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85
  --enable-hierarchical-cache --hicache-size 96 --page-size 64
  --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through
  --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion
  --enable-metrics --enable-cache-report --port "$PORT")

trap 'pkill -9 -f sglang.launch_server 2>/dev/null' EXIT
echo "arm,node,lambda,rep,req_throughput,out_tok_s,ttft_p50_ms,ttft_p99_ms,e2e_p99_ms,hit_rate,pass" > "$OUT/shortlane_ab.csv"

run_arm () {   # $1=arm_label  $2=extra_flag(s)  $3="lam:K ..."
  local ALABEL="$1" EXTRA="$2"; shift 2; local SPECS="$*"
  local LAUNCH=("${BASE_LAUNCH[@]}"); [ -n "$EXTRA" ] && LAUNCH+=($EXTRA)
  echo ">>> [SHORTLANE] launch arm=$ALABEL extra='$EXTRA' node=$NODE $(date -u +%H:%M:%S)"
  "${LAUNCH[@]}" > "$OUT/server_${ALABEL}.log" 2>&1 & local SRV=$!
  trap 'kill $SRV 2>/dev/null; sleep 4; pkill -9 -f sglang.launch_server 2>/dev/null' RETURN
  local ok=0 i
  for i in $(seq 1 900); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && { echo "ready ~$((i*3))s"; ok=1; break; }; kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED arm=$ALABEL"; tail -40 "$OUT/server_${ALABEL}.log"; break; }; sleep 3; done
  [ "$ok" = 1 ] || { echo "SERVER_TIMEOUT arm=$ALABEL"; kill $SRV 2>/dev/null; sleep 5; pkill -9 -f sglang.launch_server 2>/dev/null; sleep 5; trap - RETURN; return 3; }
  echo ">>> [SHORTLANE] warmup arm=$ALABEL $(date -u +%H:%M:%S)"
  python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
    --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
    --request-rate 3 --max-concurrency "$MAXC" --num-prompts "$WARMUP_NUMP" --port $PORT \
    --output-file "$OUT/bench_${ALABEL}_warmup.json" > "$OUT/bench_${ALABEL}_warmup.txt" 2>&1 || true
  local spec LAM K REP
  for spec in $SPECS; do
    LAM="${spec%%:*}"; K="${spec##*:}"
    for REP in $(seq 1 "$K"); do
      local tag="${ALABEL}_l${LAM}_r${REP}"
      echo ">>> [SHORTLANE] arm=$ALABEL lambda=$LAM rep=$REP/$K $(date -u +%H:%M:%S)"
      [ -n "${SL_PERREQ:-}" ] && export KLEINROCK_PERREQ_DUMP="$OUT/perreq_${tag}.csv" || unset KLEINROCK_PERREQ_DUMP
      python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
        --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
        --request-rate "$LAM" --max-concurrency "$MAXC" --num-prompts "$NUMP" --port $PORT \
        --output-file "$OUT/bench_${tag}.json" > "$OUT/bench_${tag}.txt" 2>&1
      curl -s localhost:$PORT/metrics > "$OUT/metrics_${tag}.txt" 2>/dev/null
      python3 - "$OUT" "$tag" "$ALABEL" "$LAM" "$REP" "$NODE" "$SLO_MS" >> "$OUT/shortlane_ab.csv" <<'PY'
import sys,re
o,tag,arm,LAM,REP,node,slo=sys.argv[1:8]; slo=float(slo)
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
p99f=float(p99) if p99 else None
pas=1 if (p99f is not None and p99f<=slo) else 0
print(f"{arm},{node},{LAM},{REP},{req or ''},{tok or ''},{p50 or ''},{p99 or ''},{e2e or ''},{hit},{pas}")
PY
      tail -1 "$OUT/shortlane_ab.csv"
    done
  done
  kill $SRV 2>/dev/null; sleep 5; pkill -9 -f sglang.launch_server 2>/dev/null; sleep 5
  trap - RETURN
}

SPECS=${SL_SPECS:-"5:3"}                 # fast screen: lambda=5 (decisive HOL rate), K=3
RESERVES=${SL_RESERVES:-${SL_RESERVE:-2048}}   # space-sep list of reserve sizes to ablate (reserve0 always run first)
echo "==== SHORTLANE-AB START node=$NODE sl_commit=$SL_COMMIT reserves=[$RESERVES] $(date -u) | specs=[$SPECS] ===="
run_arm "reserve0"        ""                                    $SPECS   # stock (byte-identical)
for R in $RESERVES; do
  run_arm "reserve${R}" "--prefill-short-lane-reserve $R" $SPECS
done
echo "==== SHORTLANE-AB DONE $(date -u) — shortlane_ab.csv in $OUT ===="
column -t -s, "$OUT/shortlane_ab.csv" 2>/dev/null || cat "$OUT/shortlane_ab.csv"
