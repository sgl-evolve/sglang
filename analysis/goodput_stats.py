#!/usr/bin/env python3
"""floyd: Fisher-exact significance of the SRPF goodput@SLO win (n=3 firming, same node 0-3).

goodput@SLO is a per-rate pass/fail on p99 TTFT <= 8s. We test SRPF vs stock by pass-count.
Measured (same node 0-3, SLO 8000ms):
  SRPF  λ3: {5892, 6039, 7387}  -> 3/3 pass ;  λ5: {5850, 6576, 7457} -> 3/3 pass
  stock λ3: {6327,6391,7765,8124,11786} -> 3/5 pass (coin-flip) ; λ5: 5 runs all 17-24s -> 0/5 pass
Requires scipy (use the cu129 .venv). Reproduces the p-values cited in P3 §5 / P1 §4.
"""
try:
    from scipy.stats import fisher_exact
except Exception:
    raise SystemExit("needs scipy — run inside the .venv (source .venv/bin/activate)")

def report(name, srpf_pass, srpf_fail, stock_pass, stock_fail):
    _, p = fisher_exact([[srpf_pass, srpf_fail], [stock_pass, stock_fail]], alternative='greater')
    print(f"{name}: SRPF {srpf_pass}/{srpf_pass+srpf_fail} pass vs stock "
          f"{stock_pass}/{stock_pass+stock_fail} pass -> Fisher exact (one-tailed) p={p:.4f}")
    return p

if __name__ == "__main__":
    report("lambda5 (decisive: stock stable-fail)", 3, 0, 0, 5)   # p=0.0179
    report("both rates combined",                    6, 0, 3, 7)   # p=0.0105
    report("lambda3 alone (both pass often)",        3, 0, 3, 2)   # p=0.357 (NOT separable by count)
    print("\nInterpretation: the lambda5 win is statistically significant (p=0.018) — SRPF passes where")
    print("stock stably fails. lambda3 is NOT separable by pass-count (both pass often); there the SRPF")
    print("effect is DISTRIBUTIONAL (tighter, lower tail: srpf 5.9-7.4s all<8s vs stock 6.3-11.8s, 2 fail),")
    print("not a pass-rate effect. Report lambda5 as the significant separation; lambda3 as coin-flip relief.")
