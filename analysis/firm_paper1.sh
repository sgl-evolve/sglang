#!/usr/bin/env bash
# Firm Paper 1: run the two remaining evals back-to-back on the held node, each with
# a DRAM-wait guard + a blocking wait for the previous to finish (summary.json).
#   v2b_flat2 : crater replicate (n=2) — SGLANG_TURING_ADMIT=flat gate_hits=2
#   v3_wb     : write_back control     — SGLANG_TURING_ADMIT=writeback
# Prereq: acquire_node.sh has set /tmp/turing_node.txt + /tmp/turing_hold_jid.txt.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing/runs
wait_done(){ local v="$1"; echo "[firm] waiting for $v ..."; for i in $(seq 1 260); do [ -f "$RUNS/$v/summary.json" ] && { echo "[firm] $v done"; return 0; }; sleep 60; done; echo "[firm] $v TIMEOUT"; }

bash "$HERE/run_next_eval.sh" v2b_flat2 flat 2   ; wait_done v2b_flat2
bash "$HERE/run_next_eval.sh" v3_wb     writeback ; wait_done v3_wb
echo "[firm] both firming evals complete. Log to W&B + update paper.html."
