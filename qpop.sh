#!/usr/bin/env bash
Q=/home/junyanch_google_com/autoresearch/workspace/sgl/researchers/onyx-7q2/experiment_queue.txt
exec 201>"$Q.lock"; flock 201
line=$(grep -vE '^\s*$' "$Q" 2>/dev/null | head -1)
[ -n "$line" ] && { grep -vFx "$line" "$Q" > "$Q.tmp" 2>/dev/null; mv "$Q.tmp" "$Q"; }
flock -u 201; exec 201>&-
printf '%s' "$line"
