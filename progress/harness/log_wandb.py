#!/usr/bin/env python3
"""Log one version's summary.json to the single persistent W&B run (one point per version).
Usage: log_wandb.py <summary.json> <version_idx> [notes]
Resumes run id 'kv-onyx-mf76' in project autoevolve-sglang. x-axis = 'version'.
"""
import json, os, sys

import wandb


def flatten(prefix, d, out):
    for k, v in d.items():
        if k == "cache_metrics":
            # log a compact count + keep raw under config-ish keys
            if isinstance(v, dict):
                out[f"{prefix}/n_cache_metrics"] = len(v)
                for ck, cv in v.items():
                    if isinstance(cv, (int, float)):
                        safe = ck.replace(" ", "").replace("{", "_").replace("}", "").replace("\"", "").replace(",", "_").replace("=", "")
                        out[f"{prefix}/cm/{safe}"] = cv
            continue
        if isinstance(v, (int, float)) and v is not None:
            out[f"{prefix}/{k}"] = v


def main():
    summary_path = sys.argv[1]
    version_idx = int(sys.argv[2])
    notes = sys.argv[3] if len(sys.argv) > 3 else ""
    with open(summary_path) as f:
        s = json.load(f)

    metrics = {"version": version_idx}
    flatten("loogle", s.get("loogle", {}), metrics)
    flatten("sharegpt", s.get("sharegpt", {}), metrics)

    os.environ.setdefault("WANDB_SILENT", "true")
    run = wandb.init(project="autoevolve-sglang", id="kv-onyx-mf76", name="kv-onyx-mf76", resume="allow")
    # store per-version meta in a table-friendly way
    run.summary[f"v{version_idx}_label"] = s.get("version")
    run.summary[f"v{version_idx}_commit"] = s.get("commit")
    if notes:
        run.summary[f"v{version_idx}_notes"] = notes
    wandb.log(metrics)
    print("LOGGED version", version_idx, s.get("version"), "commit", s.get("commit"))
    print("  loogle.ttft_mean_ms =", s.get("loogle", {}).get("ttft_mean_ms"))
    print("  sharegpt.ttft_mean_ms =", s.get("sharegpt", {}).get("ttft_mean_ms"))
    run.finish()


if __name__ == "__main__":
    main()
