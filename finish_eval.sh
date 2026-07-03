#!/usr/bin/env bash
# finish_eval.sh <version> <tag: config|mechanism> — self-audit + log one eval to the W&B curve.
# Runs after eval.sh writes runs/<version>/. Does the mandatory checks, prints key metrics vs the
# official baseline, and logs to sgl-evolve/onyx-7q2.
set -uo pipefail
NAME=onyx-7q2
ROOT=/home/junyanch_google_com/autoresearch
PROG=$ROOT/programs/sgl/researcher
WORK=$ROOT/workspace/sgl/researchers/$NAME
VENV=$WORK/.venv/bin/python
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
export WANDB_API_KEY
VER="${1:?usage: finish_eval.sh <version> <config|mechanism>}"
TAG="${2:?usage: finish_eval.sh <version> <config|mechanism>}"
OUT="$WORK/runs/$VER"
COMMIT=$(git -C "$WORK" rev-parse --short HEAD)

echo "==================== SELF-AUDIT: $VER (commit $COMMIT) ===================="
echo "--- resolved_args.json (must be ctx 262144, mem-frac 0.85, hicache_size 96, tp 8) ---"
cat "$OUT/resolved_args.json" 2>/dev/null || { echo "NO resolved_args.json — eval did not complete"; exit 1; }
echo ""
echo "--- SILENT FALLBACK / BUDGET check ---"
grep -iE "SILENT FALLBACK|BUDGET_MISMATCH|ABORT" "$WORK/eval-$VER.log" 2>/dev/null && echo "!!! FALLBACK/ABORT FOUND — investigate" || echo "OK: none"
echo ""
echo "--- key metrics vs v0_official (mean TTFT 87615 / median 1224 / p90 243466 / out 146.9 / hit .816 / L3 .254) ---"
"$VENV" - "$OUT/summary.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1])); m=s.get("mix",s)
base=dict(ttft_mean_ms=87615.4,ttft_median_ms=1224.53,ttft_p90_ms=243466.0,ttft_p99_ms=270798.52,out_tok_s=146.89,hit_rate=0.8163,hicache_hit_storage_frac=0.2538,tpot_mean_ms=241.29)
for k in ["ttft_mean_ms","ttft_median_ms","ttft_p90_ms","ttft_p99_ms","tpot_mean_ms","out_tok_s","req_throughput","hit_rate","hicache_hit_storage_frac","hicache_load_back_mean_ms"]:
    v=m.get(k); b=base.get(k)
    d=""
    if v is not None and b:
        pct=(v-b)/b*100
        d=f"  ({pct:+.1f}% vs base)"
    print(f"  {k:32s} {v}{d}")
PY
echo ""
echo "==================== LOGGING to W&B ($TAG) ===================="
"$VENV" "$PROG/.claude/skills/report-sop/scripts/log_wandb.py" "$NAME" "$OUT/summary.json" "$VER" "$COMMIT" "$TAG" 2>&1 | tail -4
