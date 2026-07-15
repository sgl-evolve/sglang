#!/usr/bin/env python3
"""floyd: generate the goodput rate-sweep figure (p99 TTFT vs lambda, stock vs SRPF, 8s SLO line)
as a self-contained inline SVG for the papers — DATA-DRIVEN from the run curve.csv files (reproducible).

stock  = runs/v0-stock/curve.csv        (reference full sweep, lambda 3/5/7/10)
srpf   = runs/v-srpf-full/curve.csv      (full contract sweep) if present, else runs/v-srpf-r1 (lambda 3,5)
The lambda=3 stock coin-flip range (n=5 {6327..11786}) is drawn as a grey bar. Same-node stock control
(lambda3 7.6s, lambda5 22.7s; runs/v-stock-srpfctl) is cited in the caption, not the plot, to keep one
stock source. Usage: python3 analysis/gen_curve_svg.py > /tmp/curve.svg
"""
import csv, os

W, H = 680, 380
L, R, T, B = 72, 18, 28, 46
PX0, PX1 = L, W - R
PY0, PY1 = T, H - B
LAM0, LAM1 = 3, 10
Y0, Y1 = 0, 45000
SLO = 8000
STOCK_L3_LO, STOCK_L3_HI = 6327, 11786   # n=5 coin-flip range

def xmap(lam): return PX0 + (lam - LAM0) / (LAM1 - LAM0) * (PX1 - PX0)
def ymap(ms):  return PY1 - (min(ms, Y1) - Y0) / (Y1 - Y0) * (PY1 - PY0)

def read_curve(path):
    pts = {}
    if not os.path.exists(path): return pts
    with open(path) as f:
        for row in csv.DictReader(f):
            try: pts[float(row["rate"])] = float(row["ttft_p99_ms"])
            except (KeyError, ValueError): continue
    return pts

def poly(pts, color, dash=""):
    d = " ".join(f"{xmap(l):.1f},{ymap(v):.1f}" for l, v in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline fill="none" stroke="{color}" stroke-width="2.5"{da} points="{d}"/>'

def dots(pts, color):
    return "".join(f'<circle cx="{xmap(l):.1f}" cy="{ymap(v):.1f}" r="4.5" fill="{color}"/>' for l, v in pts)

def main():
    stock_d = read_curve("runs/v0-stock/curve.csv")
    srpf_d = read_curve("runs/v-srpf-full/curve.csv") or read_curve("runs/v-srpf-r1/curve.csv")
    stock = sorted(stock_d.items()); srpf = sorted(srpf_d.items())
    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Georgia,serif" font-size="13">')
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>')
    s.append(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="#333"/>')
    s.append(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="#333"/>')
    for ms in range(0, 45001, 10000):
        y = ymap(ms)
        s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#eee"/>')
        s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{ms//1000}s</text>')
    for lam in (3, 5, 7, 10):
        s.append(f'<text x="{xmap(lam):.1f}" y="{PY1+18}" text-anchor="middle" fill="#555">λ={lam}</text>')
    s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-8}" text-anchor="middle" fill="#333">request rate λ (req/s, Poisson)</text>')
    s.append(f'<text x="16" y="{(PY0+PY1)/2:.0f}" text-anchor="middle" fill="#333" transform="rotate(-90 16 {(PY0+PY1)/2:.0f})">p99 TTFT</text>')
    yslo = ymap(SLO)
    s.append(f'<line x1="{PX0}" y1="{yslo:.1f}" x2="{PX1}" y2="{yslo:.1f}" stroke="#c00" stroke-width="1.5" stroke-dasharray="6,4"/>')
    s.append(f'<text x="{PX1-4}" y="{yslo-6:.1f}" text-anchor="end" fill="#c00">8 s SLO</text>')
    xb = xmap(3)
    s.append(f'<line x1="{xb:.1f}" y1="{ymap(STOCK_L3_LO):.1f}" x2="{xb:.1f}" y2="{ymap(STOCK_L3_HI):.1f}" stroke="#999" stroke-width="6" opacity="0.5"/>')
    s.append(poly(stock, "#888")); s.append(dots(stock, "#888"))
    s.append(poly(srpf, "#1a7f37")); s.append(dots(srpf, "#1a7f37"))
    s.append(f'<rect x="{PX1-168}" y="{PY0+6}" width="150" height="52" fill="white" stroke="#ddd"/>')
    s.append(f'<line x1="{PX1-158}" y1="{PY0+22}" x2="{PX1-138}" y2="{PY0+22}" stroke="#888" stroke-width="2.5"/><text x="{PX1-132}" y="{PY0+26}" fill="#333">stock (FCFS)</text>')
    s.append(f'<line x1="{PX1-158}" y1="{PY0+42}" x2="{PX1-138}" y2="{PY0+42}" stroke="#1a7f37" stroke-width="2.5"/><text x="{PX1-132}" y="{PY0+46}" fill="#333">SRPF (reorder)</text>')
    s.append('</svg>')
    print("\n".join(s))
    import sys
    print(f"<!-- stock={stock} srpf={srpf} -->", file=sys.stderr)

if __name__ == "__main__":
    main()
