#!/usr/bin/env bash
# DIAGNOSTIC (off-contract research, NOT a formal eval): boot the frozen 2-tier server and run
# bench at a configurable NUMP / rate to probe the host-pressure regime the charter's premise
# assumes (working set ~19M >> cache) but the contract eval.sh (NUMP=500) does not reach.
# Frozen server flags identical to eval.sh; only NUMP/RATES vary. Instrumentation via sentinel.
# Usage: diag_pressure.sh <label> <numprompts> "<rates>" [extra server args...]
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; source "$ROOT/.env"; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
WORK=$ROOT/workspace/sgl/v0.3/research/researchers/lamport
LABEL="${1:?label}"; NUMP="${2:?numprompts}"; RATES="${3:?rates}"; shift 3; EXTRA=("$@")
OUT="$WORK/runs/$LABEL"; mkdir -p "$OUT"; PORT=${PORT:-30001}
source "$WORK/.venv/bin/activate"; export PYTHONPATH="$WORK/python"; cd "$WORK"
export TRITON_CACHE_DIR="$WORK/.cache/triton" CUDA_CACHE_PATH="$WORK/.cache/nv" FLASHINFER_CACHE_DIR="$WORK/.cache/flashinfer" XDG_CACHE_HOME="$WORK/.cache"
MODEL=Qwen/Qwen3.5-122B-A10B-FP8; MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
MAXC=256
DRAMG=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); [ "${DRAMG:-0}" -lt 1300 ] && { echo "DRAM_TOO_LOW ${DRAMG}G"; exit 2; }
LAUNCH=(python3 -m sglang.launch_server --model-path "$MODEL" --tp 8 --trust-remote-code
  --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85
  --enable-hierarchical-cache --hicache-size 96 --page-size 64
  --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through
  --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion
  --enable-metrics --enable-cache-report --port "$PORT")
[ ${#EXTRA[@]} -gt 0 ] && LAUNCH+=("${EXTRA[@]}")
"${LAUNCH[@]}" > "$OUT/server.log" 2>&1 & SRV=$!
trap 'kill $SRV 2>/dev/null; sleep 4; pkill -9 -f sglang.launch_server 2>/dev/null' EXIT
for i in $(seq 1 900); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && { echo "ready ~$((i*3))s"; break; }; kill -0 $SRV 2>/dev/null || { echo SERVER_DIED; tail -30 "$OUT/server.log"; exit 3; }; sleep 3; done
curl -sf localhost:$PORT/health >/dev/null 2>&1 || { echo SERVER_TIMEOUT; exit 3; }
echo "label,rate,req_throughput,out_tok_s,ttft_p50_ms,ttft_p99_ms,e2e_p99_ms,hit_rate" > "$OUT/curve.csv"
for R in $RATES; do
  curl -s -X POST localhost:$PORT/flush_cache >/dev/null 2>&1; sleep 5
  echo ">>> rate=$R numprompts=$NUMP $(date -u +%H:%M:%S)"
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
def sraw(metric):
  vs=re.findall(r"(?m)^"+re.escape(metric)+r"\{[^}]*\}\s+([0-9.eE+]+)",m); return sum(float(x) for x in vs) if vs else None
def praw(metric,**lab):
  for l,v in re.findall(r"(?m)^"+re.escape(metric)+r"\{([^}]*)\}\s+([0-9.eE+]+)",m):
    if all(f'{k}="{val}"' in l for k,val in lab.items()): return float(v)
dev=praw("sglang:cached_tokens_total",cache_source="device") or 0; host=praw("sglang:cached_tokens_total",cache_source="host") or 0
prm=sraw("sglang:prompt_tokens_total"); hit=round((dev+host)/prm,4) if prm else ""
hu=sraw("sglang:hicache_host_used_tokens"); ht=sraw("sglang:hicache_host_total_tokens"); hutil=round(hu/ht,3) if hu and ht else ""
print(f"sweep,{R},{req or ''},{tok or ''},{p50 or ''},{p99 or ''},{e2e or ''},{hit}  # host_util={hutil}")
PY
done
echo "==== DIAG_DONE $LABEL (NUMP=$NUMP) — curve.csv in $OUT ===="
