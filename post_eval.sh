#!/usr/bin/env bash
# post_eval.sh <version> <commit> <config|mechanism>
# Waits for runs/<version>/summary.json, runs the required self-audit (resolved_args match
# contract, no SILENT FALLBACK, eval exited ok), and logs to W&B ONLY if the audit passes.
# Writes a human-readable verdict to POST_EVAL_STATUS_<version>.txt for me to review + email.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
NAME=quill-7m3
WORK=$ROOT/workspace/sgl/researchers/$NAME
VER="${1:?ver}"; COMMIT="${2:-2dda8247a}"; TAG="${3:-mechanism}"
RUN=$WORK/runs/$VER; SUM=$RUN/summary.json
LOG=$ROOT/programs/sgl/researcher/.claude/skills/report-sop/scripts/log_wandb.py
ST=$WORK/POST_EVAL_STATUS_$VER.txt
set -a; source "$ROOT/.env"; set +a
PY=$WORK/.venv/bin/python

echo "post_eval watching for $SUM ..." > "$ST"; date >> "$ST"
# wait up to ~10h for summary.json
for i in $(seq 1 36000); do [ -f "$SUM" ] && break; sleep 1; done
[ -f "$SUM" ] || { echo "TIMEOUT: no summary.json after 10h" >> "$ST"; exit 1; }

echo "=== summary.json found @ $(date) ===" >> "$ST"
# --- self-audit ---
RA=$RUN/resolved_args.json
audit_ok=1
if [ -f "$RA" ]; then
  for kv in '"context_length": "262144"' '"mem_fraction_static": "0.85"' '"hicache_size": "96"' '"tp_size": "8"'; do
    grep -q "$kv" "$RA" || { echo "AUDIT FAIL: missing $kv in resolved_args" >> "$ST"; audit_ok=0; }
  done
else
  echo "AUDIT FAIL: no resolved_args.json" >> "$ST"; audit_ok=0
fi
if grep -riq "SILENT FALLBACK" "$WORK"/eval-*.log "$RUN"/*.log 2>/dev/null; then
  echo "AUDIT WARN: SILENT FALLBACK present" >> "$ST"; audit_ok=0
fi
echo "--- key metrics vs baselines (tuned TTFT=108824, official=87615) ---" >> "$ST"
$PY - "$SUM" >> "$ST" 2>&1 <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1])); p=d.get("panel",d.get("mix",{}))
for k in ["overall/ttft_mean_ms","mix_latency/ttft_p90_ms","mix_latency/ttft_p99_ms","overall/out_tok_s","overall/hit_rate","overall/l3_hit_frac","mix_latency/tpot_mean_ms"]:
    if k in p: print(f"  {k} = {p[k]}")
t=p.get("overall/ttft_mean_ms")
if t is not None:
    print(f"  vs v0_tuned(108824): {'BETTER' if t<108824 else 'worse'} ({108824-t:+.0f} ms)")
    print(f"  vs v0_official(87615): {'BETTER' if t<87615 else 'worse'} ({87615-t:+.0f} ms)")
PYEOF

if [ "$audit_ok" = 1 ]; then
  echo "=== AUDIT PASSED -> logging to W&B ($TAG) ===" >> "$ST"
  $PY "$LOG" "$NAME" "$SUM" "$VER" "$COMMIT" "$TAG" >> "$ST" 2>&1 && echo "WANDB LOGGED OK" >> "$ST" || echo "WANDB LOG FAILED" >> "$ST"
  # Email if this beats v0_tuned (new best). Factual report; send-and-continue.
  $PY - "$SUM" "$VER" "$COMMIT" "$ST" <<'PYEOF' >> "$ST" 2>&1
import json,sys,os,smtplib
from email.mime.text import MIMEText
sumf,ver,commit,stf=sys.argv[1:5]
d=json.load(open(sumf)); p=d.get("panel",d.get("mix",{}))
t=p.get("overall/ttft_mean_ms") or p.get("mix_overview/ttft_mean_ms")
TUNED=108824.27; OFFICIAL=87615.4
if t is None: print("no ttft; skip email"); sys.exit()
lines=[f"{k} = {p[k]}" for k in ["overall/ttft_mean_ms","mix_latency/ttft_p90_ms","mix_latency/ttft_p99_ms","overall/out_tok_s","overall/hit_rate","overall/l3_hit_frac","mix_latency/tpot_mean_ms"] if k in p]
if t < TUNED:
    pw=os.environ["APP_PASSWORD"].replace(" ",""); snd="chengjunyan2@gmail.com"; to="chengjunyan0@gmail.com"
    body=(f"NEW BEST from quill-7m3 (beats v0_tuned).\n\nversion={ver}  commit={commit}  tag=mechanism\n"
          f"mechanism: parallel L3 file-backend I/O (ThreadPool over per-page reads/writes in HiCacheFile).\n\n"
          f"TTFT mean = {t:.0f} ms  (v0_tuned {TUNED:.0f}, {TUNED-t:+.0f}; v0_official {OFFICIAL:.0f}, {OFFICIAL-t:+.0f})\n"
          + "\n".join("  "+l for l in lines)
          + f"\n\nW&B: https://wandb.ai/acjy777-dartmouth/sgl-evolve/runs/quill-7m3\n"
            f"Branch evolve/quill-7m3 @ {commit}. Lossless (identical bytes/host slots; only I/O concurrency changes).\n- quill-7m3\n")
    m=MIMEText(body); m["Subject"]=f"[AutoEvolve][quill-7m3] NEW BEST {ver}: TTFT {t:.0f}ms (beats v0_tuned)"
    m["From"]=snd; m["To"]=to
    with smtplib.SMTP_SSL("smtp.gmail.com",465) as s: s.login(snd,pw); s.sendmail(snd,[to],m.as_string())
    print("NEW-BEST EMAIL SENT")
else:
    print(f"not a new best (ttft {t:.0f} >= tuned {TUNED:.0f}); no email")
PYEOF
else
  echo "=== AUDIT FAILED -> NOT logging; review needed ===" >> "$ST"
fi
echo "post_eval done @ $(date)" >> "$ST"
