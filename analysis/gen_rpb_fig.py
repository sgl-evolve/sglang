#!/usr/bin/env python3
"""floyd: P4 (rpb-chunking) figure — the rate-dependent flip, as a self-contained inline SVG.

Grouped bars per rate (λ=3, λ=5): baseline SRPF (grey), fixed RPB (red), adaptive RPB (green), showing
p99 TTFT vs the 8 s SLO line. Fixed RPB spikes over the SLO at λ=5 (the negative); adaptive sits at the
baseline (neutral). DATA-DRIVEN: reads the mean p99 over all same-node replicate runs of each config, and
draws a min–max whisker when n≥2. Reproducible.

configs (each = list of run dirs, averaged):
  baseline = v-srpf-ctl4, v-srpf-ctl5, ...      (--schedule-policy srpf)
  fixed    = v-rpb25, v-rpb25-r2, ...            (+ --enable-rpb-chunking --rpb-reserve-frac 0.25)
  adaptive = v-rpbA25, v-rpbA25-r2, ...          (+ --rpb-adaptive)
Usage: python3 analysis/gen_rpb_fig.py > /tmp/rpb_fig.svg   (run from the floyd dir)
"""
import csv, os, glob, statistics as st

RUNS = {
    "baseline SRPF": (["v-srpf-ctl4", "v-srpf-ctl5", "v-srpf-ctl6"], "#888"),
    "fixed RPB":     (["v-rpb25", "v-rpb25-r2", "v-rpb25-r3"], "#c0392b"),
    "adaptive RPB":  (["v-rpbA25", "v-rpbA25-r2", "v-rpbA25-r3"], "#1a7f37"),
}
RATES = [3, 5]
SLO = 8000.0
W, H = 620, 340
L, R, T, B = 62, 18, 26, 54
Y1 = 10500.0

def p99_at(ver, rate):
    p = f"runs/{ver}/curve.csv"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                if abs(float(row["rate"]) - rate) < 1e-6:
                    v = float(row["ttft_p99_ms"])
                    # guard against crashed/partial runs (req/s far below offered, or hit missing)
                    if float(row.get("req_throughput", 0)) < rate * 0.6:
                        return None
                    return v
            except (KeyError, ValueError):
                continue
    return None

def collect(vers, rate):
    xs = [v for ver in vers if (v := p99_at(ver, rate)) is not None]
    return xs

def ymap(v):
    return (T) + (1 - min(v, Y1) / Y1) * (H - B - T)

def xpos(ri, ci, ncfg):
    # rate group ri, config ci
    gw = (W - L - R) / len(RATES)
    gx = L + ri * gw
    bw = gw * 0.62 / ncfg
    x0 = gx + gw * 0.19
    return x0 + ci * bw, bw

def main():
    cfgs = list(RUNS.items())
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Georgia,serif" font-size="12.5">']
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>')
    s.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#333"/>')
    s.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#333"/>')
    for ms in range(0, 10001, 2000):
        y = ymap(ms)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="#eee"/>')
        s.append(f'<text x="{L-6}" y="{y+4:.1f}" text-anchor="end" fill="#555">{ms//1000}s</text>')
    # SLO line
    ys = ymap(SLO)
    s.append(f'<line x1="{L}" y1="{ys:.1f}" x2="{W-R}" y2="{ys:.1f}" stroke="#c00" stroke-width="1.5" stroke-dasharray="6,4"/>')
    s.append(f'<text x="{W-R-2}" y="{ys-5:.1f}" text-anchor="end" fill="#c00">8 s SLO</text>')
    ncfg = len(cfgs)
    for ri, rate in enumerate(RATES):
        gw = (W - L - R) / len(RATES)
        s.append(f'<text x="{L + ri*gw + gw/2:.0f}" y="{H-B+18}" text-anchor="middle" fill="#333">λ = {rate}</text>')
        for ci, (name, (vers, color)) in enumerate(cfgs):
            xs = collect(vers, rate)
            if not xs:
                continue
            mean = st.mean(xs)
            x, bw = xpos(ri, ci, ncfg)
            y = ymap(mean)
            over = mean > SLO
            s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.86:.1f}" height="{H-B-y:.1f}" '
                     f'fill="{color}" opacity="{0.95 if over else 0.8}"/>')
            s.append(f'<text x="{x+bw*0.43:.1f}" y="{y-4:.1f}" text-anchor="middle" fill="#333" font-size="11">{mean:.0f}{"*" if over else ""}</text>')
            if len(xs) >= 2:  # min-max whisker (error bar)
                ylo, yhi = ymap(min(xs)), ymap(max(xs))
                cx = x + bw*0.43
                s.append(f'<line x1="{cx:.1f}" y1="{ylo:.1f}" x2="{cx:.1f}" y2="{yhi:.1f}" stroke="#222" stroke-width="1.2"/>')
                for yy in (ylo, yhi):
                    s.append(f'<line x1="{cx-3:.1f}" y1="{yy:.1f}" x2="{cx+3:.1f}" y2="{yy:.1f}" stroke="#222" stroke-width="1.2"/>')
    # legend
    lx, ly = L + 8, T + 4
    for ci, (name, (vers, color)) in enumerate(cfgs):
        n = len(collect(vers, 3))
        s.append(f'<rect x="{lx}" y="{ly+ci*16:.0f}" width="11" height="11" fill="{color}" opacity="0.85"/>')
        s.append(f'<text x="{lx+16}" y="{ly+ci*16+10:.0f}" fill="#333" font-size="11.5">{name} (n={n})</text>')
    s.append(f'<text x="14" y="{(T+H-B)/2:.0f}" text-anchor="middle" fill="#333" transform="rotate(-90 14 {(T+H-B)/2:.0f})">p99 TTFT</text>')
    s.append('</svg>')
    print("\n".join(s))
    import sys
    for rate in RATES:
        for name, (vers, _) in cfgs:
            xs = collect(vers, rate)
            print(f"<!-- λ{rate} {name}: n={len(xs)} vals={[round(v) for v in xs]} -->", file=sys.stderr)

if __name__ == "__main__":
    main()
