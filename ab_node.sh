#!/usr/bin/env bash
# SAME-NODE A/B (on-node): on ONE node, run TWO configs back-to-back (fresh server each) at λ set,
# to compare XTIER-v4 vs stock write_back WITHOUT node-variance confounding. Off-protocol verification.
# Configs: "wb" = --hicache-write-policy write_back ; "xt" = env SGLANG_XTIER_LAZY=1 (write_through base).
# Usage: ab_node.sh <lambdas_csv>   (e.g. 3,4)
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
SGL_WORKSPACE="$ROOT/workspace/sgl/v0.25_ablations/base_free"
WORK=$SGL_WORKSPACE/researchers/base_free
LAMS="${1:?lambdas csv}"; PORT=${PORT:-30000}
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
export PATH="$WORK/.venv/bin:$PATH"; PY="$WORK/.venv/bin/python3"
export PYTHONPATH="$WORK/python"; cd "$WORK"
export TRITON_CACHE_DIR="$WORK/.cache/triton" CUDA_CACHE_PATH="$WORK/.cache/nv" \
       FLASHINFER_CACHE_DIR="$WORK/.cache/flashinfer" XDG_CACHE_HOME="$WORK/.cache"
MODEL=Qwen/Qwen3.5-122B-A10B-FP8; MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
DRAMG=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
[ "${DRAMG:-0}" -lt 1450 ] && { echo "DRAM_TOO_LOW ${DRAMG}G"; exit 2; }
GMAX=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|sort -n|tail -1)
[ "${GMAX:-0}" -gt 5000 ] && { echo "GPU_BUSY ${GMAX}MiB — skip"; exit 2; }
# NCCL preflight (like eval.sh) — bail fast on a flaky node before the expensive launch
mkdir -p "$WORK/runs/_ab_probe"; cat > "$WORK/runs/_ab_probe/nccl.py" <<'PY'
import torch, torch.distributed as d
d.init_process_group("nccl"); r=d.get_rank(); torch.cuda.set_device(r)
x=torch.ones(8,device="cuda"); d.all_reduce(x); torch.cuda.synchronize(); print("NCCL_OK",r); d.destroy_process_group()
PY
torchrun --standalone --nproc_per_node=8 "$WORK/runs/_ab_probe/nccl.py" > "$WORK/runs/_ab_probe/log" 2>&1 || { echo "NODE_NCCL_FAIL on $(hostname) — skip"; exit 6; }
echo "NCCL preflight OK"

run_cfg(){ # $1=label  $2..=extra launch args ; XTIER via env exported by caller per label
  local label="$1"; shift; local extra=("$@")
  local OUT="$WORK/runs/ab-$label"; mkdir -p "$OUT"
  local LAUNCH=("$PY" -m sglang.launch_server --model-path "$MODEL" --tp 8 --trust-remote-code
    --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85
    --enable-hierarchical-cache --hicache-size 96 --page-size 64
    --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through
    --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion
    --enable-metrics --enable-cache-report --port "$PORT")
  [ ${#extra[@]} -gt 0 ] && LAUNCH+=("${extra[@]}")
  echo ">>> AB[$label] launch (XTIER_LAZY=${SGLANG_XTIER_LAZY:-0} WM=${SGLANG_XTIER_WM_FRAC:-} extra=[${extra[*]:-none}])"
  "${LAUNCH[@]}" > "$OUT/server.log" 2>&1 & local SRV=$!
  for i in $(seq 1 900); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo "AB[$label] SERVER_DIED"; return 3; }; sleep 3; done
  curl -sf localhost:$PORT/health >/dev/null 2>&1 || { echo "AB[$label] TIMEOUT"; kill $SRV 2>/dev/null; return 3; }
  IFS=',' read -ra LA <<< "$LAMS"
  for L in "${LA[@]}"; do
    curl -s -X POST localhost:$PORT/flush_cache >/dev/null 2>&1; sleep 5
    "$PY" benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
      --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
      --request-rate "$L" --max-concurrency 128 --num-prompts 1553 --port $PORT \
      --output-file "$OUT/res_lam$L.json" 2>&1 | tee "$OUT/bench_lam$L.txt" | grep -E "Request throughput|Median TTFT|P99 TTFT|Mean TTFT"
  done
  kill $SRV 2>/dev/null; sleep 8; pkill -9 -f "[s]glang.launch_server" 2>/dev/null; sleep 10
}

# A: stock write_back (config). Ensure XTIER off.
SGLANG_XTIER_LAZY=0 run_cfg wb --hicache-write-policy write_back
# B: XTIER-v4 (mechanism) on write_through base.
export SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_XTIER_BATCH=128 SGLANG_XTIER_PERIOD=4 SGLANG_XTIER_LOG=0
run_cfg xt
echo "================ AB_DONE ================"
