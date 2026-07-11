#!/usr/bin/env bash
# Launch VMR@1553 on a given free held node. Enables VMR via sentinel (robust) + env.
set -uo pipefail
node="${1:?node}"; jid="${2:?jid}"
POOL="/home/junyanch_google_com/autoresearch/workspace/sgl/v0.3/_pool"
W=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.3/research/researchers/lamport
DIAG="$W/tools/diag_pressure.sh"
echo "vmr" > "$W/.lamport_mech"   # enable VMR
nohup bash -c "
exec 200>'$POOL/locks/$node.lock'
flock -w 30 200 || { echo lockfail; exit 1; }
echo '[vmr1553] on $node'
LAMPORT_MECH=vmr srun --jobid='$jid' --overlap -N1 -w '$node' --gres=gpu:8 bash '$DIAG' diag1553vmr 1553 '5'
flock -u 200; echo '[vmr1553] done rc='\$?
" > "$W/diag-1553-vmr.log" 2>&1 &
echo "vmr1553 launched PID $! on $node"
