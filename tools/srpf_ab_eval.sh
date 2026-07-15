#!/usr/bin/env bash
# kleinrock SRPF A/B — same-node PAIRED comparison of the DEFAULT scheduler (fcfs, what the fixed eval uses)
# vs my SRPF (shortest-remaining-prefill-first) engine mechanism, on stable metrics + goodput-pass-count, at
# the rates that matter (lambda=5 decisive [stock reliably-fails -> clean fail->pass detection]; lambda=3 the
# coin-flip knee). This is CONTRACT-LEGAL: --schedule-policy is NOT in eval.sh's FORBIDDEN list; SRPF is a real
# engine mechanism I added (2 commits on evolve/kleinrock-srpf), selected via that arg. Both phases run the SAME
# binary (srpf-worktree python = main + only the SRPF files) so the ONLY difference is the flag toggled -> the
# cleanest possible control. Per my own methodology: same node (controls heterogeneity), warmup->steady-state,
# median-of-k, report the p99 TTFT DISTRIBUTION (not a single coin-flip draw).
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env" 2>/dev/null; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
MAIN=$ROOT/workspace/sgl/v0.31/research/researchers/kleinrock
SRPFWT=$ROOT/workspace/sgl/v0.31/research/kleinrock-srpf-wt        # engine source (main + SRPF only)
source "$MAIN/.venv/bin/activate"
export PYTHONPATH="$SRPFWT/python"                                 # <-- SRPF-capable engine for BOTH phases
cd "$SRPFWT"
export TRITON_CACHE_DIR="$MAIN/.cache/triton" CUDA_CACHE_PATH="$MAIN/.cache/nv" FLASHINFER_CACHE_DIR="$MAIN/.cache/flashinfer" XDG_CACHE_HOME="$MAIN/.cache"; mkdir -p "$MAIN/.cache"
OUT="$MAIN/${SRPF_OUT:-runs/v3-srpf-ab}"; mkdir -p "$OUT"; PORT=${PORT:-30037}
NUMP=1553; WARMUP_NUMP=300; MAXC=256; SLO_MS=8000
MODEL=Qwen/Qwen3.5-122B-A10B-FP8; MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
NODE=$(hostname -s 2>/dev/null || hostname)
SRPF_COMMIT=$(git -C "$SRPFWT" rev-parse --short HEAD 2>/dev/null || echo "?")
DRAMG=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); [ "${DRAMG:-0}" -lt 1300 ] && { echo "DRAM_TOO_LOW ${DRAMG}G"; exit 2; }

# frozen server flags -- BYTE-IDENTICAL to protocol eval.sh (stock write_through). Per-phase EXTRA appended.
BASE_LAUNCH=(python3 -m sglang.launch_server --model-path "$MODEL" --tp 8 --trust-remote-code
  --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85
  --enable-hierarchical-cache --hicache-size 96 --page-size 64
  --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through
  --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion
  --enable-metrics --enable-cache-report --port "$PORT")

trap 'pkill -9 -f sglang.launch_server 2>/dev/null' EXIT   # global safety net on abnormal exit
echo "policy,node,lambda,rep,req_throughput,out_tok_s,ttft_p50_ms,ttft_p99_ms,e2e_p99_ms,hit_rate,pass" > "$OUT/srpf_ab.csv"

run_phase () {   # $1=policy_label  $2=extra_flag(s, space-sep or empty)  $3="lam:K lam:K ..." (decisive first)
  local PLABEL="$1" EXTRA="$2"; shift 2; local SPECS="$*"
  local LAUNCH=("${BASE_LAUNCH[@]}"); [ -n "$EXTRA" ] && LAUNCH+=($EXTRA)
  echo ">>> [SRPF-AB] launch policy=$PLABEL extra='$EXTRA' node=$NODE $(date -u +%H:%M:%S)"
  "${LAUNCH[@]}" > "$OUT/server_${PLABEL}.log" 2>&1 & local SRV=$!
  trap 'kill $SRV 2>/dev/null; sleep 4; pkill -9 -f sglang.launch_server 2>/dev/null' RETURN
  local ok=0 i
  for i in $(seq 1 900); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && { echo "ready ~$((i*3))s"; ok=1; break; }; kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED policy=$PLABEL"; tail -40 "$OUT/server_${PLABEL}.log"; break; }; sleep 3; done
  [ "$ok" = 1 ] || { echo "SERVER_TIMEOUT policy=$PLABEL"; kill $SRV 2>/dev/null; sleep 5; pkill -9 -f sglang.launch_server 2>/dev/null; sleep 5; trap - RETURN; return 3; }
  echo ">>> [SRPF-AB] warmup policy=$PLABEL ($WARMUP_NUMP @ rate 3) $(date -u +%H:%M:%S)"
  python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
    --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
    --request-rate 3 --max-concurrency "$MAXC" --num-prompts "$WARMUP_NUMP" --port $PORT \
    --output-file "$OUT/bench_${PLABEL}_warmup.json" > "$OUT/bench_${PLABEL}_warmup.txt" 2>&1 || true
  local spec LAM K REP
  for spec in $SPECS; do
    LAM="${spec%%:*}"; K="${spec##*:}"
    for REP in $(seq 1 "$K"); do
      local tag="${PLABEL}_l${LAM}_r${REP}"
      echo ">>> [SRPF-AB] policy=$PLABEL lambda=$LAM rep=$REP/$K $(date -u +%H:%M:%S)"
      python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
        --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
        --request-rate "$LAM" --max-concurrency "$MAXC" --num-prompts "$NUMP" --port $PORT \
        --output-file "$OUT/bench_${tag}.json" > "$OUT/bench_${tag}.txt" 2>&1
      curl -s localhost:$PORT/metrics > "$OUT/metrics_${tag}.txt" 2>/dev/null
      python3 - "$OUT" "$tag" "$PLABEL" "$LAM" "$REP" "$NODE" "$SLO_MS" >> "$OUT/srpf_ab.csv" <<'PY'
import sys,re
o,tag,pol,LAM,REP,node,slo=sys.argv[1:8]; slo=float(slo)
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
print(f"{pol},{node},{LAM},{REP},{req or ''},{tok or ''},{p50 or ''},{p99 or ''},{e2e or ''},{hit},{pas}")
PY
      tail -1 "$OUT/srpf_ab.csv"
    done
  done
  kill $SRV 2>/dev/null; sleep 5; pkill -9 -f sglang.launch_server 2>/dev/null; sleep 5
  trap - RETURN
}

SPECS=${SRPF_SPECS:-"5:5 3:3"}     # decisive lambda=5 (K5) first, then knee lambda=3 (K3), per phase
POLICIES=${SRPF_POLICIES:-"srpf"}  # policies to test vs fcfs baseline (space-sep). e.g. "lof srpf" for deferral-specificity
echo "==== SRPF-AB START node=$NODE srpf_commit=$SRPF_COMMIT $(date -u) | specs=[$SPECS] policies=[$POLICIES] ===="
run_phase "fcfs" "" $SPECS         # Phase A: default scheduler (what the fixed eval uses)
for P in $POLICIES; do
  run_phase "$P" "--schedule-policy $P" $SPECS
done
echo "==== SRPF-AB DONE $(date -u) — srpf_ab.csv in $OUT ===="
column -t -s, "$OUT/srpf_ab.csv" 2>/dev/null || cat "$OUT/srpf_ab.csv"
