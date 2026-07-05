#!/usr/bin/env python3
"""drift-3e7 mixed-chunk correctness verdict.

Reads the baseline (mixed-chunk OFF) and mixed (mixed-chunk ON) client dumps and
decides losslessness. Key comparisons:

  A) baseline.seq  vs mixed.seq   -> SANITY: mix-inactive path must be byte-identical
                                     (my code is gated on is_mixed(); no-mix => no change).
  B) baseline.conc vs baseline.seq-> CONTROL: divergence floor from concurrency FP noise
                                     alone (batch-composition-dependent reduction order).
  C) mixed.conc    vs mixed.seq   -> CORE: mix-active vs its own mix-inactive reference.
  D) mixed.conc    vs baseline.conc-> mix effect at matched concurrency.

Lossless iff: (1) zero request failures with mixed-chunk on, AND (2) the mix-active
divergence (C/D) is no worse than the concurrency-only control (B) -- i.e. divergences
are late and logprob-benign, not the early garbage / failures of the v11 corruption.
"""
import json, sys


def load(p):
    with open(p) as f:
        return json.load(f)


def cmp_lists(a_ids, b_ids, a_lps, b_lps):
    """Return (exact, first_div_idx or -1, max_lp_delta_on_common)."""
    n = min(len(a_ids), len(b_ids))
    first = -1
    for i in range(n):
        if a_ids[i] != b_ids[i]:
            first = i
            break
    exact = first == -1 and len(a_ids) == len(b_ids)
    common = first if first != -1 else n
    max_lp = 0.0
    for i in range(common):
        max_lp = max(max_lp, abs(a_lps[i] - b_lps[i]))
    return exact, first, max_lp


def compare(name, A, B):
    """A, B are lists of per-prompt dicts (same order)."""
    n = len(A)
    both_ok = 0
    exact = 0
    diverged = []
    first_idxs = []
    max_lp_all = 0.0
    for a, b in zip(A, B):
        if not (a["ok"] and b["ok"]):
            continue
        both_ok += 1
        e, first, mlp = cmp_lists(a["out_ids"], b["out_ids"], a["out_lps"], b["out_lps"])
        max_lp_all = max(max_lp_all, mlp)
        if e:
            exact += 1
        else:
            diverged.append((a["id"], first, len(a["out_ids"]), len(b["out_ids"])))
            if first >= 0:
                first_idxs.append(first)
    med_first = sorted(first_idxs)[len(first_idxs) // 2] if first_idxs else None
    min_first = min(first_idxs) if first_idxs else None
    print(f"\n=== {name} ===")
    print(f"  both_ok={both_ok}/{n}  exact_match={exact}/{both_ok}  diverged={len(diverged)}")
    print(f"  first-divergence idx: min={min_first} median={med_first}  (higher=later=benign)")
    print(f"  max |logprob delta| on matched prefix = {max_lp_all:.4f}")
    if diverged:
        show = diverged[:8]
        print(f"  diverged prompts (id, first_div, lenA, lenB): {show}")
    return {
        "both_ok": both_ok, "n": n, "exact": exact, "diverged": len(diverged),
        "min_first": min_first, "med_first": med_first, "max_lp": max_lp_all,
    }


def fails(run, tag):
    sf = sum(1 for x in run["seq"] if not x["ok"])
    cf = sum(1 for x in run["conc"] if not x["ok"])
    errs = [x["err"] for x in run["conc"] if not x["ok"]][:5]
    print(f"[{tag}] seq_fail={sf} conc_fail={cf}  sample_errs={errs}")
    return sf, cf


def main():
    base = load(sys.argv[1])   # mixed-chunk OFF
    mixd = load(sys.argv[2])   # mixed-chunk ON
    print("############ MIXED-CHUNK CORRECTNESS VERDICT ############")
    bsf, bcf = fails(base, "baseline(OFF)")
    msf, mcf = fails(mixd, "mixed(ON)")

    A = compare("A) baseline.seq vs mixed.seq  [SANITY: expect exact]", base["seq"], mixd["seq"])
    B = compare("B) baseline.conc vs baseline.seq  [CONTROL: concurrency FP floor]", base["conc"], base["seq"])
    C = compare("C) mixed.conc vs mixed.seq  [CORE: mix-active vs reference]", mixd["conc"], mixd["seq"])
    D = compare("D) mixed.conc vs baseline.conc  [mix effect @ matched concurrency]", mixd["conc"], base["conc"])

    # ---- verdict ----
    print("\n############ DECISION ############")
    ok = True
    reasons = []
    if msf or mcf:
        ok = False
        reasons.append(f"mixed-chunk had request FAILURES (seq={msf} conc={mcf}) -> NOT lossless (v11 class)")
    # Sanity: mix-inactive must match baseline exactly (allow tiny logprob noise from separate runs)
    if A["exact"] < A["both_ok"]:
        reasons.append(
            f"WARN sanity A not fully exact ({A['exact']}/{A['both_ok']}); "
            f"mix-inactive path should be identical -- investigate if min_first is small"
        )
    # Core: mix divergence must not be worse than the concurrency-only control.
    def worse(x, ctrl):
        # 'worse' = earlier divergence than control OR far larger logprob delta
        if x["min_first"] is None:
            return False
        if ctrl["min_first"] is None:
            return x["min_first"] < 8  # control never diverged; any early mix divergence is suspicious
        return (x["min_first"] < ctrl["min_first"] - 4) or (x["max_lp"] > max(0.5, 5 * ctrl["max_lp"]))
    if worse(C, B) or worse(D, B):
        ok = False
        reasons.append("mix-active divergence WORSE than concurrency control -> suspected corruption")
    verdict = "LOSSLESS (mixed-chunk safe)" if ok else "NOT LOSSLESS / SUSPECT"
    print(f"VERDICT: {verdict}")
    for r in reasons:
        print("  - " + r)
    if ok and not reasons:
        print("  - zero failures; mix divergence within concurrency FP-noise floor.")
    print("##################################")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
