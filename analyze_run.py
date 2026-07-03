#!/usr/bin/env python3
"""Diagnose the throughput bottleneck of an eval run from its server.log.
Parses sglang's periodic 'Prefill batch' / 'Decode batch' lines + retract warnings.
Usage: analyze_run.py <run_dir>   (e.g. runs/v1-timeout)
"""
import sys, re, json, statistics as st, os

run = sys.argv[1]
log = os.path.join(run, "server.log")
if not os.path.exists(log):
    print("no server.log at", log); sys.exit(1)
txt = open(log, errors="ignore").read()

def nums(pat, s):
    return [float(x) for x in re.findall(pat, s)]

# Decode batch lines: "#running-req: N", "token usage: X", "gen throughput (token/s): T", "#queue-req: Q"
dec = re.findall(r"Decode batch.*?#running-req:\s*(\d+).*?token usage:\s*([\d.]+).*?gen throughput \(token/s\):\s*([\d.]+).*?#queue-req:\s*(\d+)", txt)
pre = re.findall(r"Prefill batch.*?#running-req:\s*(\d+).*?#queue-req:\s*(\d+)", txt)
retr = re.findall(r"Retract requests\. #retracted_reqs:\s*(\d+)", txt)

def summ(name, xs):
    if not xs: print(f"  {name}: (none)"); return
    xs = sorted(xs)
    p = lambda q: xs[min(len(xs)-1, int(q*len(xs)))]
    print(f"  {name}: n={len(xs)} min={xs[0]:.1f} p50={p(.5):.1f} mean={st.mean(xs):.1f} p90={p(.9):.1f} max={xs[-1]:.1f}")

print(f"=== {run} bottleneck diagnosis ===")
print(f"decode-batch log lines: {len(dec)} | prefill-batch lines: {len(pre)} | retract events: {len(retr)}")
if dec:
    run_reqs = [float(d[0]) for d in dec]
    tok_use  = [float(d[1]) for d in dec]
    gen_tps  = [float(d[2]) for d in dec]
    q_reqs   = [float(d[3]) for d in dec]
    print("DECODE steps:")
    summ("#running-req", run_reqs)
    summ("token usage ", tok_use)
    summ("gen tok/s   ", gen_tps)
    summ("#queue-req  ", q_reqs)
if pre:
    summ("PREFILL #running-req", [float(p[0]) for p in pre])
    summ("PREFILL #queue-req  ", [float(p[1]) for p in pre])
if retr:
    tot = sum(int(r) for r in retr)
    print(f"RETRACT: {len(retr)} events, {tot} reqs retracted total")
else:
    print("RETRACT: none (no device-KV-pool-full retraction)")

# merge summary.json headline if present
sj = os.path.join(run, "summary.json")
if os.path.exists(sj):
    s = json.load(open(sj)).get("mix", {})
    print("HEADLINE:", {k: s.get(k) for k in ["ttft_mean_ms","ttft_p99_ms","out_tok_s","req_throughput","hit_rate","hicache_hit_storage_frac"]})
