#!/usr/bin/env bash
#SBATCH -J kvheron-spf-ab
#SBATCH -p a3
#SBATCH -N1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH -t 3:00:00
#SBATCH -o /home/junyanch_google_com/autoresearch/workspace/sgl/researchers/kv-heron-eb9/spf-ab-%j.out
# ---------------------------------------------------------------------------------------------------
# OFF-CONTRACT scheduler A/B (NOT a formal eval; results are NOT loggable to the W&B curve).
# Purpose: get real, node-confound-immune fcfs-vs-spf signal for v16 WITHOUT a certified node, since
# my regime is host-dominated with the disk (L3) tier INERT (l3_hit_frac=0, host_util=0.97). SPF acts
# purely on the waiting-queue order, so it is faithful even with a shrunk (inert) disk reservation.
# Same server config as v6 EXCEPT: disk max_size 1800G->200G (fits a spare node), my own SSD subdir,
# port 30055. best_effort + my concurrent-IO code (via PYTHONPATH) = the v6 base for BOTH arms.
# Runs fcfs then spf back-to-back on ONE node (same-node A/B). Reduced num-prompts for faster signal;
# concurrency 128 + rate 3.5 preserves the queue pressure SPF targets.
# ---------------------------------------------------------------------------------------------------
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
NAME=kv-heron-eb9
WORK=$ROOT/workspace/sgl/researchers/$NAME
SSD=/mnt/localssd/$NAME-ab
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$SSD"
source "$WORK/.venv/bin/activate"; export PYTHONPATH="$WORK/python"; cd "$WORK"
export TRITON_CACHE_DIR="$WORK/.cache/triton" CUDA_CACHE_PATH="$WORK/.cache/nv" \
       FLASHINFER_CACHE_DIR="$WORK/.cache/flashinfer" XDG_CACHE_HOME="$WORK/.cache"
mkdir -p "$WORK/.cache"
MODEL=Qwen/Qwen3.5-122B-A10B-FP8
MIX=/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl
PORT=30055
ABROOT="$WORK/runs/_spf_ab"; mkdir -p "$ABROOT"
NPROMPTS="${NPROMPTS:-800}"
echo "[ab] node=$(hostname) commit=$(git -C "$WORK" rev-parse --short HEAD) nprompts=$NPROMPTS"

wipe(){ rm -rf "$SSD" 2>/dev/null; mkdir -p "$SSD"; }

# NCCL preflight (uncertified node -> verify 8-GPU fabric before a ~5-10min load)
mkdir -p "$ABROOT"
cat > "$ABROOT/_nccl.py" <<'PY'
import torch, torch.distributed as d
d.init_process_group("nccl"); r=d.get_rank(); torch.cuda.set_device(r)
x=torch.ones(8,device="cuda"); d.all_reduce(x); torch.cuda.synchronize()
print("NCCL_OK", r); d.destroy_process_group()
PY
if ! torchrun --standalone --nproc_per_node=8 "$ABROOT/_nccl.py" > "$ABROOT/nccl.log" 2>&1; then
  echo "[ab] NODE_NCCL_FAIL on $(hostname) — aborting A/B"; grep -iE "insuffic|error|abort" "$ABROOT/nccl.log" | head -3; exit 6
fi
echo "[ab] NCCL preflight OK"

run_policy(){  # $1 = fcfs|spf
  local pol="$1" OUT="$ABROOT/$1"; mkdir -p "$OUT"; wipe
  local LAUNCH=(python3 -m sglang.launch_server
    --model-path "$MODEL" --tp 8 --trust-remote-code
    --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85
    --enable-hierarchical-cache --hicache-size 96
    --hicache-storage-backend file --hicache-storage-backend-extra-config "{\"file_path\":\"$SSD\",\"max_size\":\"200G\",\"min_free_space\":\"20G\"}"
    --page-size 64 --hicache-io-backend direct --hicache-mem-layout page_first_direct
    --hicache-write-policy write_through --hicache-storage-prefetch-policy best_effort
    --schedule-policy "$pol"
    --enable-metrics --enable-cache-report --port "$PORT")
  printf '%q ' "${LAUNCH[@]}" > "$OUT/launch_cmd.txt"; echo >> "$OUT/launch_cmd.txt"
  echo "[ab] === launch server policy=$pol ==="
  "${LAUNCH[@]}" > "$OUT/server.log" 2>&1 &
  local SRV=$!
  local ready=0
  for i in $(seq 1 900); do
    curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { ready=1; echo "[ab] $pol server ready (~$((i*3))s)"; break; }
    kill -0 $SRV 2>/dev/null || { echo "[ab] $pol SERVER_DIED:"; tail -40 "$OUT/server.log"; break; }
    sleep 3
  done
  if [ "$ready" != 1 ]; then echo "[ab] $pol server not ready — skip"; kill $SRV 2>/dev/null; pkill -9 -f "port $PORT" 2>/dev/null; sleep 5; return 1; fi
  # confirm the policy actually resolved (SPF must not silently fall back)
  grep -oE "schedule_policy='?[a-z-]+'?" "$OUT/server.log" | head -1
  curl -s -X POST "localhost:$PORT/flush_cache" >/dev/null 2>&1; sleep 5; wipe
  echo "[ab] === bench policy=$pol nprompts=$NPROMPTS ==="
  python3 benchmark/hicache/bench_serving.py --backend sglang --model "$MODEL" \
    --dataset-name loogle --dataset-path "$MIX" --enable-multiturn --disable-shuffle \
    --request-rate 3.5 --max-concurrency 128 --num-prompts "$NPROMPTS" --port "$PORT" \
    --output-file "$OUT/mix_result.json" 2>&1 | tee "$OUT/mix.txt"
  kill $SRV 2>/dev/null; sleep 5; pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 3; wipe
}

run_policy fcfs
run_policy spf

echo "==================== SPF A/B SUMMARY (off-contract, node=$(hostname)) ===================="
python3 - "$ABROOT/fcfs/mix_result.json" "$ABROOT/spf/mix_result.json" <<'PY'
import json,sys
def g(p):
    try: j=json.load(open(p))
    except Exception as e: return None
    return j
def pick(j,*names):
    for n in names:
        if n in j and isinstance(j[n],(int,float)): return j[n]
    return None
labels=["fcfs","spf"]; rows={}
for lab,p in zip(labels,sys.argv[1:3]):
    j=g(p)
    if not j: print(f"{lab}: NO RESULT ({p})"); continue
    rows[lab]=dict(
      mean=pick(j,"mean_ttft_ms","ttft_mean_ms"),
      median=pick(j,"median_ttft_ms","ttft_median_ms","p50_ttft_ms"),
      p99=pick(j,"p99_ttft_ms","ttft_p99_ms"),
      out_tps=pick(j,"output_throughput","output_throughput_tok_s"),
      completed=pick(j,"completed","num_prompts"))
    print(f"{lab}: mean_ttft={rows[lab]['mean']} median={rows[lab]['median']} p99={rows[lab]['p99']} out_tps={rows[lab]['out_tps']} completed={rows[lab]['completed']}")
if "fcfs" in rows and "spf" in rows and rows["fcfs"]["mean"] and rows["spf"]["mean"]:
    d=100*(rows["fcfs"]["mean"]-rows["spf"]["mean"])/rows["fcfs"]["mean"]
    print(f">>> SPF mean-TTFT change vs fcfs: {d:+.1f}%  (positive = SPF faster)")
PY
echo "==================== A/B DONE ===================="
