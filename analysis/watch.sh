#!/usr/bin/env bash
# Boundary-surviving watcher: logs each eval's completion + key metrics to STATUS.txt
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/valiant
S=STATUS.txt
declare -A seen
while true; do
  for d in runs/*/; do
    v=$(basename "$d")
    if [ -f "$d/summary.json" ] && [ -z "${seen[$v]:-}" ]; then
      seen[$v]=1
      line=$(python3 -c "import json,sys
try:
  d=json.load(open('$d/summary.json'));p=d.get('panel',{})
  print('good@SLO=%.2f peak_tok=%.0f best_hit=%.3f | '%(p.get('overall/goodput_reqs_at_SLO',0),p.get('overall/peak_out_tok_s',0),p.get('overall/hit_rate',0))+' '.join('%s=%s'%(k.split('/')[-1],round(v,1) if isinstance(v,float) else v) for k,v in p.items() if k.startswith('curve/')))
except Exception as e: print('parse-err',e)" 2>/dev/null)
      echo "$(date -u +%H:%M:%S) DONE $v :: $line" >> $S
    fi
  done
  sleep 120
done
