#!/usr/bin/env bash
# FIXED evaluation protocol runner for researcher kv-onyx-mf76.
# Runs ON the designated a3 node (invoke via: srun --jobid=$HOLD --overlap -N1 bash run_eval.sh <version>).
# Two benchmarks with a MANDATORY full reset between them. Identical server launch flags for both.
# Budget flags are FIXED (never change): --tp 8, --mem-fraction-static 0.85, --hicache-size 1024,
#   --context-length 65536, --enable-hierarchical-cache. Mechanism flags are overridable via env.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
NAME=kv-onyx-mf76
VERSION="${1:?usage: run_eval.sh <version-label>}"
PORT="${PORT:-30000}"
WORK=$ROOT/programs/sglang/research-$NAME
set -a; source "$ROOT/.env"; set +a
export HF_TOKEN="${HF_API_KEY:-}" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.8; export PATH="$CUDA_HOME/bin:$PATH"
# venv pip-CUDA libs. deep_gemm needs cu13 (.so.13 sonames); torch needs cu12 (.so.12/.10).
# ORDER MATTERS: cu12 + torch FIRST so their .so.12/.10 win; cu13 LAST (its .so.13 are unique,
# but its .so.12/.10 builds of cufft/cusparse/curand must NOT shadow torch's CUDA-12 libs -> NCCL/CUDA errors).
SP="$WORK/.venv/lib/python3.12/site-packages"
NVLIBS_CU12=$(echo "$SP"/nvidia/*/lib | tr ' ' '\n' | grep -v '/cu13/' | tr '\n' ':')
export LD_LIBRARY_PATH="$SP/torch/lib:$NVLIBS_CU12:$CUDA_HOME/lib64:$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
# deep_gemm ships only a cu13 build whose kernels require a CUDA-13 driver; this node's driver is
# 570.195 (CUDA 12.8). deep_gemm still IMPORTS (cu13 libs on path) but we must NOT run its kernels,
# else cuda-graph capture / forward hit cudaErrorInsufficientDriver. Disable -> model uses cu12 FP8 path.
export SGLANG_ENABLE_JIT_DEEPGEMM=0
# flashinfer norm defaults to a CUTLASS-DSL (cu13) kernel that needs a CUDA-13 driver; force its
# precompiled cu12 CUDA norm kernel instead (lossless: same RMSNorm math, different kernel).
export FLASHINFER_USE_CUDA_NORM=1
source "$WORK/.venv/bin/activate"
export PYTHONPATH="$WORK/python"
PY="$WORK/.venv/bin/python"
cd "$WORK"
SSD=/mnt/localssd/$NAME; mkdir -p "$SSD"; rm -rf "$SSD"/* 2>/dev/null
# CRITICAL: HiCacheFile reads its storage dir from this env var (or defaults to /tmp/hicache on the
# 194 GB root fs!). The factory creates HiCacheFile(storage_config) WITHOUT file_path, so the
# extra_config "file_path" is IGNORED. Must set the env var to put the L3 disk tier on /mnt/localssd.
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$SSD"
OUT="$WORK/runs/$VERSION"; mkdir -p "$OUT"
MODEL=Qwen/Qwen3.5-397B-A17B-FP8
# LooGLE: the cached longdep_qa.jsonl uses the new HF schema (context/question/answer); the bench
# loader expects the old schema (input/qa_pairs). Reformat the SAME data (grouped by doc -> multiturn)
# so the UNMODIFIED loader runs. Benchmark code untouched; data content identical.
LOOGLE_SRC=$(find "$HF_HOME" ~/.cache/huggingface -name longdep_qa.jsonl 2>/dev/null | head -1)
LOOGLE="$WORK/runs/loogle_longdep_oldfmt.jsonl"
[ -f "$LOOGLE" ] || python3 "$WORK/progress/harness/convert_loogle.py" "$LOOGLE_SRC" "$LOOGLE"
SHAREGPT=/rmeng_data/junyanch-data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json

# ---- mechanism flags (baseline defaults; override via env per version) ----
PAGE_SIZE="${PAGE_SIZE:-64}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-6144}"
IO_BACKEND="${IO_BACKEND:-kernel}"
# Qwen3.5-397B is a HYBRID (Mamba/linear-attn) MoE; its MambaPoolHost only supports
# 'page_first_direct' (the protocol's 'page_first' raises ValueError). Forced by the model arch;
# both attention & mamba host pools support page_first_direct. Same for baseline + all versions.
MEM_LAYOUT="${MEM_LAYOUT:-page_first_direct}"
WRITE_POLICY="${WRITE_POLICY:-write_through}"
STORAGE_BACKEND="${STORAGE_BACKEND:-file}"
PREFETCH_POLICY="${PREFETCH_POLICY:-wait_complete}"
EXTRA_SERVER_FLAGS="${EXTRA_SERVER_FLAGS:-}"
# Disk-tier (L3) budget: the file backend defaults to NO cap (MAX_SIZE=None, MIN_FREE_SPACE=0),
# so write_through fills the shared /mnt/localssd -> ENOSPC -> scheduler crash. Enforce the 2.1 TB
# disk BUDGET via the LRU evictor (cap) + a filesystem free-space guard (shared disk also holds
# other users' data). Identical for baseline + all versions.
DISK_MAX_SIZE="${DISK_MAX_SIZE:-2000G}"   # ~2.0 TiB cap (under the 2.1 TB disk budget)
DISK_MIN_FREE="${DISK_MIN_FREE:-300G}"    # keep >=300 GB free on the shared filesystem
FILE_IO_THREADS="${FILE_IO_THREADS:-}"    # v1 lever: parallelize file-L3 page IO (unset=serial baseline)
STORAGE_CFG="{\"file_path\":\"$SSD\",\"max_size\":\"$DISK_MAX_SIZE\",\"min_free_space\":\"$DISK_MIN_FREE\""
[ -n "$FILE_IO_THREADS" ] && STORAGE_CFG="$STORAGE_CFG,\"file_io_threads\":$FILE_IO_THREADS"
STORAGE_CFG="$STORAGE_CFG}"
# ---- FIXED budget (never change) ----
CONTEXT_LENGTH=65536
MEM_FRACTION=0.85
# --hicache-size is PER TP RANK and applied to EACH host pool. Qwen3.5 is hybrid, so HiCache
# allocates TWO host pools (attention KV + Mamba state), each = hicache_size GB/rank. On TP=8 the
# total host RAM = 2 pools x hicache_size x 8. The protocol's literal "1024" => 16 TB (absurd);
# even 128 => 2 TB (OOM on 1.86 TB). The 1 TB TOTAL host BUDGET is realized as 64 GB/rank
# (2 x 64 x 8 = 1024 GB = 1 TB total). Identical for baseline and every version; respects the ceiling.
HICACHE_SIZE="${HICACHE_SIZE:-64}"

launch_server() {  # $1 = logfile
  "$PY" -m sglang.launch_server --model-path "$MODEL" --tp 8 --trust-remote-code \
    --page-size "$PAGE_SIZE" --context-length "$CONTEXT_LENGTH" --chunked-prefill-size "$CHUNKED_PREFILL" \
    --mem-fraction-static "$MEM_FRACTION" \
    --enable-hierarchical-cache --hicache-size "$HICACHE_SIZE" \
    --hicache-io-backend "$IO_BACKEND" --hicache-mem-layout "$MEM_LAYOUT" --hicache-write-policy "$WRITE_POLICY" \
    --hicache-storage-backend "$STORAGE_BACKEND" --hicache-storage-prefetch-policy "$PREFETCH_POLICY" \
    --hicache-storage-backend-extra-config "$STORAGE_CFG" \
    $EXTRA_SERVER_FLAGS \
    --enable-metrics --enable-cache-report --port "$PORT" > "$1" 2>&1 &
  echo $!
}
wait_health() {  # $1 = server pid
  local srv=$1 i
  for i in $(seq 1 700); do
    curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { echo "READY after ~$((i*3))s"; return 0; }
    kill -0 "$srv" 2>/dev/null || { echo "SERVER_DIED during load"; return 1; }
    sleep 3
  done
  echo "HEALTH_TIMEOUT"; return 1
}
snap() { curl -s "localhost:$PORT/metrics" 2>/dev/null > "$1"; }
reset_server() {  # $1 = server pid
  kill "$1" 2>/dev/null; sleep 5
  pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 10
  rm -rf "$SSD"/* 2>/dev/null
}

{ echo "version=$VERSION"; echo "commit=$(git rev-parse HEAD)"; echo "date=$(date -u)"; echo "host=$(hostname)";
  echo "page_size=$PAGE_SIZE chunked_prefill=$CHUNKED_PREFILL io_backend=$IO_BACKEND mem_layout=$MEM_LAYOUT";
  echo "write_policy=$WRITE_POLICY prefetch_policy=$PREFETCH_POLICY storage_backend=$STORAGE_BACKEND";
  echo "extra='$EXTRA_SERVER_FLAGS'  (FIXED: tp8 ctx=$CONTEXT_LENGTH memfrac=$MEM_FRACTION hicache=$HICACHE_SIZE)";
} | tee "$OUT/config.txt"

# free disk sanity
FREEG=$(df -BG /mnt/localssd 2>/dev/null | tail -1 | awk '{gsub(/G/,"",$4);print $4}')
echo "localssd_free=${FREEG}G"
[ "${FREEG:-0}" -lt 2100 ] && echo "WARN: <2.1T free on /mnt/localssd"

# ============================== (A) LooGLE — bench_serving ==============================
echo ">>> [A] LooGLE: launching server $(date -u)"
SRV=$(launch_server "$OUT/server_loogle.log")
if wait_health "$SRV"; then
  snap "$OUT/loogle_metrics_pre.txt"
  echo ">>> [A] LooGLE: running bench_serving $(date -u)"
  "$PY" benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
    --dataset-name loogle --dataset-path "$LOOGLE" --enable-multiturn --disable-shuffle \
    --request-rate 10 --max-concurrency 128 --num-prompts 200 --port "$PORT" \
    --output-file "$OUT/loogle_result.json" 2>&1 | tee "$OUT/loogle.txt"
  snap "$OUT/loogle_metrics_post.txt"
else
  echo "LOOGLE_LAUNCH_FAILED"; tail -100 "$OUT/server_loogle.log"
fi
echo ">>> RESET after LooGLE $(date -u)"
reset_server "$SRV"

# ============================== (B) ShareGPT — bench_mix.py ==============================
echo ">>> [B] ShareGPT: launching server $(date -u)"
SRV=$(launch_server "$OUT/server_sharegpt.log")
if wait_health "$SRV"; then
  snap "$OUT/sharegpt_metrics_pre.txt"
  export CONFIG_PATH="$OUT/bench_mix_config.json"
  cat > "$CONFIG_PATH" <<'JSON'
{
  "num_rounds": 10,
  "num_clients": 60,
  "round_ratios": [50, 25, 15, 15, 10, 10, 9, 8, 7, 6],
  "mean_new_tokens_per_round": [1000, 400, 350, 300, 280, 260, 240, 220, 210, 200],
  "mean_return_tokens_per_round": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
  "mean_inter_round_interval": [60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
}
JSON
  echo ">>> [B] ShareGPT: running bench_mix.py (duration 600s) $(date -u)"
  "$PY" benchmark/hicache/bench_mix.py --model-path "$MODEL" --dataset-path "$SHAREGPT" \
    --port "$PORT" --duration 600 2>&1 | tee "$OUT/bench_mix.out"
  snap "$OUT/sharegpt_metrics_post.txt"
else
  echo "SHAREGPT_LAUNCH_FAILED"; tail -100 "$OUT/server_sharegpt.log"
fi
echo ">>> FINAL RESET $(date -u)"
reset_server "$SRV"
echo "EVAL_DONE version=$VERSION out=$OUT $(date -u)"
