#!/usr/bin/env python3
"""Paper 7: settle the capacity-ceiling attribution (prefill-compute vs decode-memory-bandwidth)
from sglang profile_by_stage chrome traces.

Usage:
  analyze_profile.py PREFILL_TRACE.json[.gz] DECODE_TRACE.json[.gz]

profile_by_stage=true writes stage-prefixed chrome traces (*.trace.json.gz). This script sums GPU
KERNEL time per stage and classifies kernels into memory-bound (attention/paged/KV-read/copy) vs
compute-bound (gemm/matmul/MoE) vs other (norm/rope/act). The verdict:
  - at saturation, if DECODE-stage GPU time dominates AND is attention/KV-read-bound => C is
    decode-memory-bandwidth-bound (the runaway theory's prediction);
  - if PREFILL-stage GPU time dominates AND is gemm/MoE-bound => C is prefill-compute-bound.

NOTE: kernel-name regexes below are first-guess; refine against the real trace's actual kernel
names (print --list to dump the top kernels by total time) before quoting numbers in the paper.
"""
import sys, json, gzip, re
from collections import defaultdict

# classification regexes (TUNE against real kernel names via --list)
MEMORY = re.compile(r"atten|paged|flash|kv|rope_?cache|copy|memcpy|cat_|gather|index|reduce|elementwise|norm", re.I)
COMPUTE = re.compile(r"gemm|matmul|moe|grouped|cutlass|wgmma|mma|fp8|conv|linear|mlp", re.I)

def load_trace(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    return d.get("traceEvents", d if isinstance(d, list) else [])

def kernel_events(events):
    # torch-profiler GPU kernels: complete events (ph="X") with cat "kernel"/"gpu_op" and a duration
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = str(e.get("cat", "")).lower()
        if "kernel" in cat or "gpu" in cat:
            yield e.get("name", "?"), float(e.get("dur", 0.0))

def classify(name):
    if COMPUTE.search(name): return "compute"
    if MEMORY.search(name):  return "memory"
    return "other"

def analyze(path):
    ev = load_trace(path)
    by_kernel = defaultdict(float); by_cat = defaultdict(float); tot = 0.0
    for name, dur in kernel_events(ev):
        by_kernel[name] += dur; by_cat[classify(name)] += dur; tot += dur
    return tot, by_cat, by_kernel

def report(tag, path):
    tot, by_cat, by_kernel = analyze(path)
    print(f"\n=== {tag}: {path} ===")
    print(f"  total GPU kernel time: {tot/1e6:.3f} s  ({tot:.0f} us)")
    for c in ("compute", "memory", "other"):
        v = by_cat.get(c, 0.0)
        print(f"    {c:8}: {v/1e6:8.3f} s  ({100*v/tot if tot else 0:5.1f}%)")
    print("  top kernels by time:")
    for name, dur in sorted(by_kernel.items(), key=lambda x: -x[1])[:10]:
        print(f"    {dur/1e6:7.3f}s  [{classify(name):7}]  {name[:70]}")
    return tot, by_cat

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    pt, pc = report("PREFILL stage", sys.argv[1])
    dt, dc = report("DECODE stage", sys.argv[2])
    print("\n=== VERDICT ===")
    if pt and dt:
        print(f"stage GPU-time share: prefill {100*pt/(pt+dt):.0f}%  decode {100*dt/(pt+dt):.0f}%")
        dmem = dc.get("memory",0)/dt*100 if dt else 0
        pcmp = pc.get("compute",0)/pt*100 if pt else 0
        print(f"decode stage: {dmem:.0f}% memory-bound kernels | prefill stage: {pcmp:.0f}% compute-bound kernels")
        print("=> C is decode-memory-BW-bound if decode dominates wall-clock AND is memory-kernel-heavy;")
        print("   C is prefill-compute-bound if prefill dominates AND is gemm/MoE-heavy. (Refine regexes first.)")
