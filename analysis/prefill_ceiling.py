#!/usr/bin/env python3
"""Paper 7 nucleus: WHAT sets the capacity ceiling C, and is it a prefill-COMPUTE wall or a
memory-bandwidth-bound-decode wall?

Finding (from existing bench_r10.json, saturation point λ=10):
  C (req/s) is set by the EFFECTIVE PREFILL TOKEN THROUGHPUT (input_throughput). Same workload
  (total_input_tokens identical), so req_throughput ∝ input_throughput. Acceleration raises
  input_throughput +36% => raises C +36%. The gain is NOT more prefill FLOPs (accel does the
  same prefill tokens) but a larger share of GPU wall-clock going to prefill because decode
  (memory-bandwidth-bound) is cleared faster (lower tpot).

Compute-boundedness check: compare effective prefill FLOP rate to the 8xH100 FP8 peak.
"""
import json, os
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")

# 8x H100 SXM, FP8 dense (no sparsity) ~= 1979 TFLOP/s each
PEAK_FP8 = 1979e12 * 8            # ~1.58e16 FLOP/s
ACTIVE_PARAMS = 10e9              # Qwen3.5-122B-A10B => ~10B activated per token
FLOP_PER_TOK_LINEAR = 2 * ACTIVE_PARAMS   # 2N for the matmuls (MoE + attn projections); excludes attention score FLOPs

def row(ver):
    d = json.load(open(os.path.join(RUNS, ver, "bench_r10.json")))
    return dict(req=d["request_throughput"], pin=d["input_throughput"], pout=d["output_throughput"],
                tpot=d["mean_tpot_ms"], conc=d["concurrency"], dur=d["duration"],
                tin=d["total_input_tokens"], tout=d["total_output_tokens"])

def show(tag, r):
    frate = r["pin"] * FLOP_PER_TOK_LINEAR                    # effective prefill FLOP/s (linear only)
    mfu = frate / PEAK_FP8
    print(f"{tag:12} req/s={r['req']:.2f}  prefill_tok/s={r['pin']:,.0f}  out_tok/s={r['pout']:.0f}  "
          f"tpot={r['tpot']:.0f}ms  conc={r['conc']:.0f}")
    print(f"{'':12} effective prefill FLOP/s(linear)={frate:.2e}  => MFU={mfu*100:.1f}% of 8xH100 FP8 peak")
    return r

print("=== Saturation point (offered λ=10): what sets C? ===")
a = show("accel(f=2)", row("v13_accelfull"))
s = show("stock",      row("v1_stock"))

print()
print("--- ROBUST (assumption-free) findings ---")
print(f"C ratio (req/s):            accel/stock = {a['req']/s['req']:.3f}  (+{(a['req']/s['req']-1)*100:.0f}%)")
print(f"prefill-throughput ratio:   accel/stock = {a['pin']/s['pin']:.3f}  (+{(a['pin']/s['pin']-1)*100:.0f}%)")
print(f"  => C is set by EFFECTIVE PREFILL THROUGHPUT (identical total_input_tokens "
      f"{a['tin']:,} vs {s['tin']:,}); the two ratios match to 3 sig figs.")
print(f"decode tpot:                accel {a['tpot']:.0f} vs stock {s['tpot']:.0f} ms "
      f"(-{(1-a['tpot']/s['tpot'])*100:.0f}%) at ~equal concurrency ({a['conc']:.0f} vs {s['conc']:.0f})")
print(f"  => acceleration reclaims decode's wall-clock share (faster decode steps at equal batch)")
print(f"     for prefill => +{(a['pin']/s['pin']-1)*100:.0f}% effective prefill throughput => +{(a['req']/s['req']-1)*100:.0f}% C.")
print()
print("--- Compute-vs-memory attribution (NEEDS a profiler; NOT claimed here) ---")
print("Linear-only prefill MFU is ~7-10%, but attention-score FLOPs for the heavy tail (docs to")
print("191K tok; last chunk attends to the whole doc) are large and workload-dependent, so a")
print("back-of-envelope CANNOT settle whether the residual wall is prefill-compute or decode-")
print("memory-bandwidth. A torch/nsys profile at saturation (prefill-kernel vs decode-kernel")
print("wall-clock share) is the clean test => Paper 7. The runaway theory predicts decode-BW.")
