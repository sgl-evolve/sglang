#!/usr/bin/env bash
# Watcher: exit when the RPB A/B completes (v-rpb25 summary.json appears) or campaign ends, max ~4h.
WORK=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/floyd
for i in $(seq 1 120); do   # 120 * 120s = 4h
  if [ -f "$WORK/runs/v-rpb25/summary.json" ]; then echo "RPB A/B DONE: v-rpb25/summary.json exists"; exit 0; fi
  if grep -q "campaign done\|\[campaign\] DONE v-rpb25\|WARN v-rpb25" "$WORK/runs/campaign_rpb.log" 2>/dev/null; then
    echo "campaign finished (see campaign_rpb.log)"; exit 0; fi
  sleep 120
done
echo "watcher timed out after 4h"; exit 0
