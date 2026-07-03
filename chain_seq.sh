#!/usr/bin/env bash
# Wait for the v1 launcher to exit (v1 eval done + node released), then start the sequence launcher.
while ps -p 1705832 >/dev/null 2>&1; do sleep 30; done
sleep 20
cd /home/junyanch_google_com/autoresearch/workspace/sgl/researchers/quill-7m3
nohup bash launch_seq.sh > seq.log 2>&1 & disown
echo "chain: started launch_seq.sh @ $(date)"
