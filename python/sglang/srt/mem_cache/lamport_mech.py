"""lamport mechanism config — Value-Density Mamba Retention (VMR).

A hybrid dual-cache co-management primitive: in the scarce Mamba-state pool
(constant-size ~18 MB SSM states), protect checkpoints whose reuse VALUE — the
attention-KV prefix they uniquely unlock — exceeds their own memory cost, instead
of evicting them blind to that value (stock LRU). Rationale: a single Mamba
checkpoint gates continuation over its whole prefix; losing a deep (long-context)
checkpoint forces a full-history recompute (the P99 TTFT tail), while its 18 MB
slot is trivially reclaimable from a shallow checkpoint that unlocks little.

Parameter-free crossover: a checkpoint is worth keeping when the KV it unlocks
(prefix_tokens * KV_BYTES_PER_TOKEN) exceeds the checkpoint's own bytes
(MAMBA_BYTES_PER_STATE) -> prefix_tokens > MAMBA_BYTES_PER_STATE/KV_BYTES_PER_TOKEN.

Gated by env LAMPORT_MECH (off by default) so A/B is on identical code. Lossless:
only changes which checkpoints are *retained*; reuse remains exact.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _mode() -> str:
    m = _env("LAMPORT_MECH", "").strip().lower()
    if m:
        return m
    # Sentinel-file fallback (robust when env does not propagate through srun):
    # walk up from this file for a `.lamport_mech` marker whose contents = mode.
    try:
        d = os.path.dirname(os.path.abspath(__file__))
        for _ in range(8):
            p = os.path.join(d, ".lamport_mech")
            if os.path.exists(p):
                with open(p) as f:
                    return f.read().strip().lower() or "0"
            nd = os.path.dirname(d)
            if nd == d:
                break
            d = nd
    except Exception:
        pass
    return "0"


# "0"/"off" disabled; "vmr" (or "1") = value-density retention.
MODE = _mode()
ENABLED = MODE not in ("0", "", "off", "false")

# Principled crossover threshold in tokens: protect a Mamba checkpoint iff the
# attention-KV prefix it unlocks is larger (in bytes) than the checkpoint itself.
# Defaults derived from the measured Qwen3.5-122B pool sizes (per rank):
#   KV: 2.35e6 tok -> 26.86 GB  => ~11.4 KB/token
#   Mamba: 1351 states -> 24.19 GB => ~17.9 MB/state
# crossover ~= 17.9e6 / 11.4e3 ~= 1570 tokens. Overridable for ablation.
KV_BYTES_PER_TOKEN = float(_env("LAMPORT_KV_BPT", "11400"))
MAMBA_BYTES_PER_STATE = float(_env("LAMPORT_MAMBA_BPS", "17900000"))
_min_tok_env = _env("LAMPORT_MIN_TOK", "")
if _min_tok_env:
    PROTECT_MIN_TOKENS = int(_min_tok_env)
else:
    PROTECT_MIN_TOKENS = int(MAMBA_BYTES_PER_STATE / max(KV_BYTES_PER_TOKEN, 1.0))

VAL_KEY = "lamport_val"  # metadata key: prefix-token depth of a Mamba checkpoint

if ENABLED:
    logger.info(
        "[LAMPORT_MECH] ENABLED mode=%s PROTECT_MIN_TOKENS=%d (KV_BPT=%.0f MAMBA_BPS=%.0f)",
        MODE,
        PROTECT_MIN_TOKENS,
        KV_BYTES_PER_TOKEN,
        MAMBA_BYTES_PER_STATE,
    )


def is_protected(value_tokens) -> bool:
    """A checkpoint unlocking > its own memory cost worth of KV is protected."""
    return ENABLED and value_tokens is not None and value_tokens >= PROTECT_MIN_TOKENS
