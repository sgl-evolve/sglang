#!/usr/bin/env python3
"""Log one point on a researcher's W&B evolution curve — the Report SOP's deterministic logger.

Why a script: every researcher must log the SAME curated keys to ONE run so the curves are
comparable. Hand-rolled wandb code drifts; this bundles the exact, repeated action.

Project `sgl-evolve`, ONE run named <name> (id=<name>, resume="allow") so successive calls
accumulate as the evolution curve. Logs the curated scalar keys from a summary.json (or the shared
baseline.json), records version/commit/resolved_args in the run config, and attaches the raw run
dir as an artifact.

Usage:
  log_wandb.py <name> <summary.json> <version> [commit] [kind]     # kind = config|mechanism (default: -)
The reference point (once, first):
  log_wandb.py <name> $SGL_HOME/researcher/baseline.json       v0_official baseline
Retract a flagged version (fairness-guard warning — see references/wandb_recall.md):
  log_wandb.py --retract <name> <version> "<reason>"

Requires: WANDB_API_KEY in the env and `wandb` installed in your venv (`uv pip install wandb`).
Exit codes: 0 ok; 2 usage; 3 nothing to log.
"""
import os
import sys
import json


def retract(name, version, reason):
    """Mark a version RETRACTED on the run (per-point delete is unsupported; tag + config note)."""
    import wandb
    api = wandb.Api()
    run = api.run(f"{os.environ.get('WANDB_PROJECT','sgl-evolve')}/{os.environ.get('SGLFREE_RUN_ID', name)}")
    rv = set(run.config.get("retracted_versions", []) or []); rv.add(version)
    run.config["retracted_versions"] = sorted(rv)
    run.config["retraction_note"] = f"{version}: {reason}"
    if "flagged" not in (run.tags or []):
        run.tags = list(run.tags or []) + ["flagged"]
    run.update()
    print(f"retracted '{version}' on sgl-evolve/{name}: {reason}")


def flatten(d, prefix=""):
    """Flatten nested dicts to slash-keys, keeping only numeric (non-bool) leaves."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "/"))
        elif isinstance(v, bool):
            continue
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def code_churn(work):
    """Cumulative engine-code churn vs the pristine base: files/added/deleted lines under python/.
    A quantitative proxy for exploration depth — major mechanism work vs config-only tweaks."""
    import subprocess

    def g(*a):
        return subprocess.run(["git", "-C", work, *a], capture_output=True, text=True).stdout

    base = (g("merge-base", "HEAD", "main").strip() or "main")
    files = added = deleted = 0
    for line in g("diff", "--numstat", base, "HEAD", "--", "python/").splitlines():
        p = line.split("\t")
        if len(p) >= 3:
            files += 1
            added += int(p[0]) if p[0].isdigit() else 0
            deleted += int(p[1]) if p[1].isdigit() else 0
    return {"files": files, "added": added, "deleted": deleted, "net": added + deleted}


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--retract":
        if len(sys.argv) < 5:
            sys.exit("usage: log_wandb.py --retract <name> <version> \"<reason>\"")
        retract(sys.argv[2], sys.argv[3], sys.argv[4]); return
    if len(sys.argv) < 4:
        sys.exit("usage: log_wandb.py <name> <summary_or_baseline.json> <version> [commit] [kind]")
    name, summ, version = sys.argv[1], sys.argv[2], sys.argv[3]
    commit = sys.argv[4] if len(sys.argv) > 4 else "?"
    kind = sys.argv[5] if len(sys.argv) > 5 else "-"   # config | mechanism | baseline | -

    if not os.path.exists(summ):
        sys.exit(f"no such file: {summ} (baseline.json is produced by the supervisor's golden run)")
    data = json.load(open(summ))
    # Log ONLY the curated `panel` — the comparable evolution curves. Its keys are section-prefixed
    # ("overall/…", "mix_hicache/…", …) so W&B groups them into collapsible sections automatically;
    # we log them verbatim. The raw per-rank/per-bucket Prometheus dumps stay in the artifact, never here.
    # Key off panel PRESENCE, not emptiness: a current-format summary always has "panel", so an
    # all-null panel (failed eval) must NOT fall back to flattening the detail dict onto the curve.
    if "panel" in data:
        metrics = {k: v for k, v in (data["panel"] or {}).items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
    else:  # pre-panel summaries -> flatten the whole thing (backward compat)
        metrics = flatten(data)
    if not metrics:
        sys.exit(f"no numeric metrics in {summ} — nothing to log")

    import wandb  # imported late so a missing wandb gives a clear error only when actually logging

    run_dir = os.path.dirname(os.path.abspath(summ))
    is_run_dir = any(
        os.path.exists(os.path.join(run_dir, f))
        for f in ("server.log", "mix_result.json", "resolved_args.json")
    )
    # Quantitative code-change (exploration depth): cumulative engine-code (python/) churn vs the pristine
    # base. Logged as OBSERVED provenance for the manager's exploration-strength accounting — not a target.
    churn = None
    if is_run_dir:
        work = os.path.dirname(os.path.dirname(run_dir))  # runs/<ver> -> the researcher's clone (WORK)
        if os.path.isdir(os.path.join(work, ".git")):
            churn = code_churn(work)
            metrics.update({"churn/py_files": churn["files"], "churn/py_added": churn["added"],
                            "churn/py_deleted": churn["deleted"], "churn/py_net": churn["net"]})
    cfg = {"version": version, "commit": commit}
    # resolved_args: prefer the block embedded in the summary/baseline JSON (always present, incl. the
    # shared baseline.json whose dir has no sibling file); fall back to a sibling resolved_args.json.
    if isinstance(data.get("resolved_args"), dict):
        cfg["resolved_args"] = data["resolved_args"]
    else:
        ra = os.path.join(run_dir, "resolved_args.json")
        if os.path.exists(ra):
            cfg["resolved_args"] = json.load(open(ra))

    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "sgl-evolve"),
                     group=os.environ.get("WANDB_RUN_GROUP", "v0.25-ablation"),
                     name=name, id=os.environ.get("SGLFREE_RUN_ID", name), resume="allow", config=cfg)
    # record this version's novelty class (config vs mechanism) for honest novelty accounting
    kinds = dict(run.config.get("version_kind", {}) or {}); kinds[version] = kind
    run.config.update({"version_kind": kinds}, allow_val_change=True)
    if churn is not None:  # per-version engine-code churn, for exploration-strength accounting
        cc = dict(run.config.get("code_churn", {}) or {}); cc[version] = churn
        run.config.update({"code_churn": cc}, allow_val_change=True)
    run.log(metrics)

    # Attach the raw run dir as an artifact — but not for the shared baseline.json, which lives in
    # $SGL_HOME/researcher/ and has no per-run dir of its own.
    if is_run_dir:
        art = wandb.Artifact(f"{name}-{version}".replace("/", "-"), type="eval-run")
        art.add_dir(run_dir)
        run.log_artifact(art)

    run.finish()
    print(f"logged '{version}' [{kind}] to sgl-evolve/{name}: {len(metrics)} metrics"
          + (f" + churn(py net {churn['net']})" if churn else "")
          + (" + artifact" if is_run_dir else ""))


if __name__ == "__main__":
    main()
