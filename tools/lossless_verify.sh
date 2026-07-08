#!/usr/bin/env bash
# flock a held node; run exc, stock, nocache legs back-to-back; compare (exc-r2 vs stock-r2 = my mechanism
# vs stock cache; stock r1-vs-r2 = is fresh-vs-cache mismatch inherent; nocache-r1 = fresh ground truth).
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
POOL="$ROOT/workspace/sgl/v0.25_ablations/_pool"
WS="$ROOT/workspace/sgl/v0.25_ablations/sgl_free/researchers/sgl_free"
ON="$WS/tools/lossless_verify_onnode.sh"
held_nodes(){ for f in "$POOL"/held/*; do [ -e "$f" ] && basename "$f"; done; }
done_all(){ [ -f "$WS/runs/lossless_verify/outputs_exc.json" ] && [ -f "$WS/runs/lossless_verify/outputs_stock.json" ] && [ -f "$WS/runs/lossless_verify/outputs_nocache.json" ]; }
for try in $(seq 1 300); do
  done_all && { echo "[lv] all legs done"; break; }
  for node in $(held_nodes); do
    jid=$(cat "$POOL/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    exec 200>"$POOL/locks/$node.lock"
    if flock -n 200; then
      echo "[lv] try $try got $node $(date -u +%H:%M:%S)"
      for mode in exc stock nocache; do
        [ -f "$WS/runs/lossless_verify/outputs_$mode.json" ] && continue
        srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$ON" "$mode" || { echo "[lv] $mode leg failed"; break; }
      done
      flock -u 200; exec 200>&-
      done_all && break 2
      echo "[lv] not all legs done on $node; retry"; break
    fi
    exec 200>&-
  done
  sleep 45
done
# compare
python3 - "$WS/runs/lossless_verify" <<'PY'
import json,sys,os
d=sys.argv[1]
def load(m):
    p=f"{d}/outputs_{m}.json"; return json.load(open(p)) if os.path.exists(p) else None
exc,stock,noc=load("exc"),load("stock"),load("nocache")
def cmp(a,b): return sum(1 for x,y in zip(a,b) if x==y), len(a)
res={}
if exc and stock: res["exc_r2_vs_stock_r2 (MY MECHANISM vs stock cache)"]=cmp(exc["r2"],stock["r2"])
if stock: res["stock_r1_vs_r2 (is fresh-vs-cache mismatch INHERENT?)"]=cmp(stock["r1"],stock["r2"])
if exc: res["exc_r1_vs_r2"]=cmp(exc["r1"],exc["r2"])
if exc and noc: res["exc_r2_vs_nocache_r1 (exclusive cache-hit vs fresh no-cache)"]=cmp(exc["r2"],noc["r1"])
if stock and noc: res["stock_r2_vs_nocache_r1"]=cmp(stock["r2"],noc["r1"])
for k,(m,n) in res.items(): print(f"  {m}/{n}  {k}")
json.dump(res, open(f"{d}/verify_result.json","w"), indent=2, default=list)
PY
