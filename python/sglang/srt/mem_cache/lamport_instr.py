"""lamport instrumentation — LOSSLESS counters to characterize the hybrid
dual-cache (Full attn KV + Mamba SSM state) reuse frontier and host churn.

Zero behavior change: pure counting, gated by LAMPORT_INSTR=1. Used to produce
the trace-driven motivation for reuse-frontier co-management. Not a mechanism.
"""
from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

def _enabled() -> bool:
    if os.environ.get("LAMPORT_INSTR", "0") == "1":
        return True
    # Sentinel-file fallback (robust when env does not propagate through srun):
    # clone root = 6 dirs up from this file (.../<clone>/python/sglang/srt/mem_cache/).
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 6))
        return os.path.exists(os.path.join(root, ".lamport_instr"))
    except Exception:
        return False


ENABLED = _enabled()
_LOG_EVERY = int(os.environ.get("LAMPORT_INSTR_EVERY", "3000"))
_lock = threading.Lock()

S = {
    "matches": 0,
    "matches_hit": 0,  # matches with any full frontier > 0
    "matches_aux_limited": 0,  # best_match_tok < full_frontier_tok (mamba strands full KV)
    "full_frontier_tok": 0,  # cumulative full-KV reuse frontier (device|host), tokens
    "best_match_tok": 0,  # cumulative all-component-valid frontier, tokens
    "stranded_tok": 0,  # cumulative (full_frontier - best_match); full KV present but un-continuable
    "mamba_host_evict_nodes": 0,
    "full_host_evict_nodes": 0,
    "full_host_evict_tok": 0,
    "mamba_dev_evict_nodes": 0,
    "full_dev_evict_tok": 0,
}


def record_match(full_frontier_tok: int, best_match_tok: int) -> None:
    if not ENABLED:
        return
    with _lock:
        S["matches"] += 1
        if full_frontier_tok > 0:
            S["matches_hit"] += 1
        S["full_frontier_tok"] += full_frontier_tok
        S["best_match_tok"] += best_match_tok
        gap = full_frontier_tok - best_match_tok
        if gap > 0:
            S["matches_aux_limited"] += 1
            S["stranded_tok"] += gap
        if S["matches"] % _LOG_EVERY == 0:
            _emit()


def add(key: str, n: int = 1) -> None:
    if not ENABLED:
        return
    with _lock:
        S[key] = S.get(key, 0) + n


def _emit() -> None:
    m = max(S["matches"], 1)
    ff = max(S["full_frontier_tok"], 1)
    logger.info(
        "[LAMPORT_INSTR] matches=%d hit=%d aux_limited=%d(%.1f%% of hits) "
        "full_front_tok=%d best_tok=%d STRANDED_tok=%d(%.2f%% of full frontier) "
        "| mamba_host_evict_nodes=%d full_host_evict=%d/%dtok "
        "mamba_dev_evict_nodes=%d full_dev_evict_tok=%d",
        S["matches"],
        S["matches_hit"],
        S["matches_aux_limited"],
        100.0 * S["matches_aux_limited"] / max(S["matches_hit"], 1),
        S["full_frontier_tok"],
        S["best_match_tok"],
        S["stranded_tok"],
        100.0 * S["stranded_tok"] / ff,
        S["mamba_host_evict_nodes"],
        S["full_host_evict_nodes"],
        S["full_host_evict_tok"],
        S["mamba_dev_evict_nodes"],
        S["full_dev_evict_tok"],
    )


def log_final() -> None:
    if ENABLED:
        with _lock:
            _emit()
