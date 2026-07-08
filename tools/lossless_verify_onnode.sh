#!/usr/bin/env bash
# lossless_verify_onnode.sh <mode>  (mode: exc | stock | nocache) — runs ON a compute node.
# Launches a server in the given cache mode, runs a 2-round greedy client over long docs (round1 fresh;
# later docs evict earlier to host; round2 = host hits -> load_back), saves ALL outputs. Purpose: compare
# exclusive vs STOCK vs no-cache to determine whether the exclusive mechanism adds ANY loss BEYOND the
# stock cache (the correct bar). Hybrid-Mamba caches are inherently non-bit-exact fresh-vs-cached (mamba
# checkpoint reconstruction), so the real test is exc-r2 == stock-r2 (my change preserves stock behavior).
set -uo pipefail
MODE="${1:?usage: lossless_verify_onnode.sh <exc|stock|nocache>}"
ROOT=/home/junyanch_google_com/autoresearch
WS="$ROOT/workspace/sgl/v0.25_ablations/sgl_free/researchers/sgl_free"
set -a; source "$ROOT/.env"; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
source "$WS/.venv/bin/activate"; export PYTHONPATH="$WS/python"; cd "$WS"
export TRITON_CACHE_DIR="$WS/.cache/triton" CUDA_CACHE_PATH="$WS/.cache/nv" \
       FLASHINFER_CACHE_DIR="$WS/.cache/flashinfer" XDG_CACHE_HOME="$WS/.cache"
MODEL=Qwen/Qwen3.5-122B-A10B-FP8
PORT=${PORT:-30077}; OUT="$WS/runs/lossless_verify"; mkdir -p "$OUT"
DRAMG=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
[ "${DRAMG:-0}" -lt 1300 ] && { echo "DRAM_TOO_LOW ${DRAMG}G on $(hostname) — skip"; exit 2; }

COMMON=(--model-path "$MODEL" --tp 8 --trust-remote-code --context-length 262144 --chunked-prefill-size 6144
  --mem-fraction-static 0.85 --page-size 64 --disable-custom-all-reduce
  --enforce-disable-flashinfer-allreduce-fusion --enable-metrics --enable-cache-report --port "$PORT")
case "$MODE" in
  exc)     export SGLANG_HICACHE_EXCLUSIVE=1
           EXTRA=(--enable-hierarchical-cache --hicache-size 96 --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through);;
  stock)   unset SGLANG_HICACHE_EXCLUSIVE
           EXTRA=(--enable-hierarchical-cache --hicache-size 96 --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-write-policy write_through);;
  nocache) unset SGLANG_HICACHE_EXCLUSIVE
           EXTRA=(--disable-radix-cache);;
esac
echo ">>> mode=$MODE on $(hostname)"
python3 -m sglang.launch_server "${COMMON[@]}" "${EXTRA[@]}" > "$OUT/server_$MODE.log" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null; sleep 5; pkill -9 -f sglang.launch_server 2>/dev/null' EXIT
for i in $(seq 1 900); do curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { echo "ready ~$((i*3))s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED"; tail -40 "$OUT/server_$MODE.log"; exit 3; }; sleep 3; done

python3 - "$PORT" "$OUT" "$MODE" <<'PY'
import sys, json, requests
port, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
url = f"http://localhost:{port}/v1/chat/completions"
docs=[]
for line in open("/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"):
    d=json.loads(line); t=d.get("input","")
    if len(t)>60000: docs.append(t[:600000])
    if len(docs)>=24: break
prompts=[f"Input: {t}\nQuestion: Summarize the key facts in one sentence." for t in docs]
def gen(p):
    r=requests.post(url, json={"model":"Qwen/Qwen3.5-122B-A10B-FP8","messages":[{"role":"user","content":p}],
        "temperature":0.0,"max_tokens":48,"stream":False}, timeout=600)
    r.raise_for_status(); return r.json()["choices"][0]["message"]["content"]
r1=[gen(p) for p in prompts]; print(f"[{mode}] round1 done")
r2=[gen(p) for p in prompts]; print(f"[{mode}] round2 done")
json.dump({"mode":mode,"r1":r1,"r2":r2}, open(f"{out}/outputs_{mode}.json","w"))
mism=[i for i in range(len(prompts)) if r1[i]!=r2[i]]
print(f"[{mode}] r1-vs-r2 self-consistency: {len(prompts)-len(mism)}/{len(prompts)} match; mism={mism}")
PY
echo "=== $MODE done ==="
