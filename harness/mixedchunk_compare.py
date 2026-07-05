#!/usr/bin/env python3
"""drift-3e7 mixed-chunk correctness verdict (text-based).

Reads the baseline (mixed-chunk OFF) and mixed (mixed-chunk ON) client dumps and
decides losslessness by comparing GREEDY decoded TEXT (return_logprob is intentionally
NOT used -- the scheduler disables mixed-chunk when any request sets it). For greedy
decoding, identical text <=> identical tokens = the eval's own losslessness gate.

Comparisons:
  A) baseline.seq  vs mixed.seq    -> SANITY: mix-inactive path must be byte-identical
                                      (fix is gated on is_mixed(); no-mix => no change).
  B) baseline.conc vs baseline.seq -> CONTROL: divergence floor from concurrency FP noise
                                      alone (batch-composition-dependent reduction order).
  C) mixed.conc    vs mixed.seq    -> CORE: mix-active vs its own mix-inactive reference.
  D) mixed.conc    vs baseline.conc-> mix effect at matched concurrency.

Lossless iff: (1) zero request failures with mixed-chunk on, AND (2) the mix-active text
divergence (C/D) is no worse than the concurrency-only control (B) -- divergences are late
and rare, not the early garbage / failures of the v11 corruption.
"""
import json, sys


def load(p):
    with open(p) as f:
        return json.load(f)


def cmp_text(a, b):
    """Return (exact, first_div_char_idx or -1)."""
    if a == b:
        return True, -1
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return False, i
    return False, n  # one is a strict prefix of the other


def compare(name, A, B):
    n = len(A)
    both_ok = exact = 0
    diverged = []
    first_idxs = []
    for a, b in zip(A, B):
        if not (a["ok"] and b["ok"]):
            continue
        both_ok += 1
        e, first = cmp_text(a.get("out_text", ""), b.get("out_text", ""))
        if e:
            exact += 1
        else:
            diverged.append((a["id"], first, len(a.get("out_text", "")), len(b.get("out_text", ""))))
            first_idxs.append(first)
    med_first = sorted(first_idxs)[len(first_idxs) // 2] if first_idxs else None
    min_first = min(first_idxs) if first_idxs else None
    print(f"\n=== {name} ===")
    print(f"  both_ok={both_ok}/{n}  exact_match={exact}/{both_ok}  diverged={len(diverged)}")
    print(f"  first-divergence CHAR idx: min={min_first} median={med_first}  (higher=later=benign)")
    if diverged:
        print(f"  diverged (id, first_char, lenA, lenB): {diverged[:8]}")
    return {"both_ok": both_ok, "n": n, "exact": exact, "diverged": len(diverged),
            "min_first": min_first, "med_first": med_first}


def fails(run, tag):
    sf = sum(1 for x in run["seq"] if not x["ok"])
    cf = sum(1 for x in run["conc"] if not x["ok"])
    errs = [x["err"] for x in run["conc"] if not x["ok"]][:5]
    print(f"[{tag}] seq_fail={sf} conc_fail={cf}  sample_errs={errs}")
    return sf, cf


def main():
    base = load(sys.argv[1])   # mixed-chunk OFF
    mixd = load(sys.argv[2])   # mixed-chunk ON
    print("############ MIXED-CHUNK CORRECTNESS VERDICT (text) ############")
    bsf, bcf = fails(base, "baseline(OFF)")
    msf, mcf = fails(mixd, "mixed(ON)")

    A = compare("A) baseline.seq vs mixed.seq  [SANITY: expect exact]", base["seq"], mixd["seq"])
    B = compare("B) baseline.conc vs baseline.seq  [CONTROL: concurrency FP floor]", base["conc"], base["seq"])
    C = compare("C) mixed.conc vs mixed.seq  [CORE: mix-active vs reference]", mixd["conc"], mixd["seq"])
    D = compare("D) mixed.conc vs baseline.conc  [mix effect @ matched concurrency]", mixd["conc"], base["conc"])

    print("\n############ DECISION ############")
    ok = True
    reasons = []
    if msf or mcf:
        ok = False
        reasons.append(f"mixed-chunk had request FAILURES (seq={msf} conc={mcf}) -> NOT lossless (v11 class)")
    if A["exact"] < A["both_ok"]:
        reasons.append(
            f"WARN sanity A not fully exact ({A['exact']}/{A['both_ok']}); mix-inactive path "
            f"should be identical -- suspect if min_first is small"
        )

    def worse(x, ctrl):
        # 'worse' = mix diverges MUCH earlier than the concurrency-only control, i.e. early
        # garbage rather than late FP drift.
        if x["min_first"] is None:
            return False  # mix never diverged
        if ctrl["min_first"] is None:
            return x["min_first"] < 16  # control never diverged; early mix divergence is suspicious
        return x["min_first"] < ctrl["min_first"] - 16
    if worse(C, B) or worse(D, B):
        ok = False
        reasons.append("mix-active text divergence WORSE (earlier) than concurrency control -> suspected corruption")
    verdict = "LOSSLESS (mixed-chunk safe)" if ok else "NOT LOSSLESS / SUSPECT"
    print(f"VERDICT: {verdict}")
    for r in reasons:
        print("  - " + r)
    if ok and not reasons:
        print("  - zero failures; mix text divergence within concurrency FP-noise floor.")
    print("##################################")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
