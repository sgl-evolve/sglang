#!/usr/bin/env bash
# Follow-up experiments after main ablation batch completes.
# Tests split-tier host eviction and other novel mechanisms.
# Usage: bash ablation_followup.sh (submits sbatch job)
set -uo pipefail
ROOT="${AUTORESEARCH_ROOT:-/home/junyanch_google_com/autoresearch}"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
export HF_TOKEN="${HF_API_KEY:-}"
SGL_HOME="/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/sgl_mech"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
NAME="sgl_mech"
WS="/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_mech/researchers/sgl_mech"
EXCL=$(bash "$SGL_HOME/researcher/.claude/skills/submit-gpu-job/scripts/certified-nodes.sh" --exclude 2>/dev/null || true)
EXCL_FLAG=()
[ -n "$EXCL" ] && EXCL_FLAG=(--exclude="$EXCL")
LOGDIR="$WS/logs"
mkdir -p "$LOGDIR"

WRAPPER="$WS/followup_wrapper.sh"
cat > "$WRAPPER" << 'WRAPPER_EOF'
#!/usr/bin/env bash
set -uo pipefail
ROOT="${AUTORESEARCH_ROOT:-/home/junyanch_google_com/autoresearch}"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
export HF_TOKEN="${HF_API_KEY:-}"
SGL_HOME="/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/sgl_mech"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
NAME="sgl_mech"
WS="/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_mech/researchers/sgl_mech"

clean_env() {
    unset SGLANG_EVICTION_POLICY_OVERRIDE SGLANG_ENABLE_COST_AWARE_EVICTION \
          SGLANG_COST_AWARE_EVICT_THRESHOLD SGLANG_SCHEDULE_OVERRIDE \
          SGLANG_WRITE_ADMISSION_MIN_COST SGLANG_CONTINUOUS_COST_ALPHA \
          SGLANG_COST_FREQ_THRESHOLD SGLANG_FREQ_DECAY_RATE \
          SGLANG_SIZE_AWARE_THRESHOLD SGLANG_LOADBACK_MIN_COST \
          SGLANG_LOADBACK_VALUE_GATE SGLANG_WRITE_THROUGH_THRESHOLD \
          SGLANG_FREQ_BOOST_WEIGHT SGLANG_COST_AWARE_REUSE_MIN \
          SGLANG_COST_AWARE_COST_MODE SGLANG_COST_AWARE_EVICT_THRESHOLD2 \
          SGLANG_HOST_EVICTION_POLICY 2>/dev/null || true
}

run_one() {
    local ver="$1"; shift
    local sf="$WS/runs/$ver/summary.json"
    if [ -f "$sf" ]; then
        echo "========== $(date) ========== SKIP: $ver (has summary.json) =========="
        return 0
    fi
    echo "========== $(date) ========== START: $ver =========="
    env | grep -E "SGLANG_(EVICTION|ENABLE_COST|SCHEDULE|WRITE_ADMISSION|LOADBACK|WRITE_THROUGH|CONTINUOUS_COST|FREQ_BOOST|COST_AWARE|SIZE_AWARE|FREQ_DECAY|HOST_EVICTION)" || echo "(no overrides)"
    bash "$EVAL" "$NAME" "$ver" "$@"
    local rc=$?
    echo "========== $(date) ========== DONE: $ver rc=$rc =========="
    return $rc
}

echo "========== $(date) ========== FOLLOW-UP BATCH START on $(hostname) =========="

# v36-splittier-lru: CostAware on device, LRU on host
# Hypothesis: host eviction benefits from pure recency (evicts dead conversations faster)
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=1; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048; export SGLANG_HOST_EVICTION_POLICY=lru
run_one v36-splittier-lru || true

# v37-splittier-gdsf: CostAware on device, GDSF on host
# Hypothesis: GDSF combines freq+cost+size for host; may better predict host-tier reuse
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=1; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048; export SGLANG_HOST_EVICTION_POLICY=gdsf
run_one v37-splittier-gdsf || true

# v38-lru-splittier-cost: stock LRU on device, CostAware on host
# Inverse: cheap nodes evicted from device (LRU, random), expensive nodes protected on host
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=0; export SGLANG_HOST_EVICTION_POLICY=cost_aware
run_one v38-lru-splittier-cost || true

# v0-ctl5: another control (statistical power)
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=0
run_one v0-ctl5 || true

# v39-t2048-ctl: another CostAware reference replicate
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=1; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
run_one v39-t2048-ctl || true

clean_env
echo "========== $(date) ========== ALL FOLLOW-UP EXPERIMENTS COMPLETE =========="
WRAPPER_EOF
chmod +x "$WRAPPER"

JOB_LOG="$LOGDIR/followup-batch-%j.out"
JID=$(sbatch --parsable \
    -p a3 \
    -N1 --exclusive --gres=gpu:8 \
    "${EXCL_FLAG[@]}" \
    -t 12:00:00 \
    -J "sgl_mech-followup" \
    --dependency=afterany:18687 \
    -o "$JOB_LOG" \
    --wrap "bash $WRAPPER" \
    2>&1) || { echo "sbatch failed: $JID" >&2; exit 1; }

echo "Submitted follow-up job $JID (depends on 18687)"
echo "Log: ${JOB_LOG/\%j/$JID}"
echo "Monitor: squeue -j $JID; tail -f ${JOB_LOG/\%j/$JID}"
