#!/usr/bin/env python3
"""floyd: generate the goodput rate-sweep figure (p99 TTFT vs lambda, stock vs SRPF, SLO line)
as a self-contained inline SVG for the papers. Data from runs/v0-stock and runs/v-srpf-r1."""

W, H = 680, 380
L, R, T, B = 72, 18, 28, 46          # margins
PX0, PX1 = L, W - R
PY0, PY1 = T, H - B
LAM0, LAM1 = 3, 10
Y0, Y1 = 0, 45000                    # p99 TTFT ms axis
SLO = 8000

def xmap(lam): return PX0 + (lam - LAM0) / (LAM1 - LAM0) * (PX1 - PX0)
def ymap(ms):  return PY1 - (ms - Y0) / (Y1 - Y0) * (PY1 - PY0)

stock = [(3, 7612), (5, 22746), (7, 33961), (10, 41334)]     # same-node (0-3) λ3,5; v0-stock λ7,10 (overload)
srpf  = [(3, 5892), (5, 5850)]                               # v-srpf-r1 (same node 0-3)
stock_l3_lo, stock_l3_hi = 6327, 11786                       # n=5 coin-flip range

def poly(pts, color, dash=""):
    d = " ".join(f"{xmap(l):.1f},{ymap(v):.1f}" for l, v in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline fill="none" stroke="{color}" stroke-width="2.5"{da} points="{d}"/>'

def dots(pts, color):
    return "".join(f'<circle cx="{xmap(l):.1f}" cy="{ymap(v):.1f}" r="4.5" fill="{color}"/>' for l, v in pts)

s = []
s.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Georgia,serif" font-size="13">')
s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>')
# axes
s.append(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="#333"/>')
s.append(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="#333"/>')
# y gridlines + labels
for ms in range(0, 45001, 10000):
    y = ymap(ms)
    s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#eee"/>')
    s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{ms//1000}s</text>')
# x labels
for lam in (3, 5, 7, 10):
    s.append(f'<text x="{xmap(lam):.1f}" y="{PY1+18}" text-anchor="middle" fill="#555">λ={lam}</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-8}" text-anchor="middle" fill="#333">request rate λ (req/s, Poisson)</text>')
s.append(f'<text x="16" y="{(PY0+PY1)/2:.0f}" text-anchor="middle" fill="#333" transform="rotate(-90 16 {(PY0+PY1)/2:.0f})">p99 TTFT</text>')
# SLO line
yslo = ymap(SLO)
s.append(f'<line x1="{PX0}" y1="{yslo:.1f}" x2="{PX1}" y2="{yslo:.1f}" stroke="#c00" stroke-width="1.5" stroke-dasharray="6,4"/>')
s.append(f'<text x="{PX1-4}" y="{yslo-6:.1f}" text-anchor="end" fill="#c00">8 s SLO</text>')
# stock coin-flip error bar at lambda=3
xb = xmap(3)
s.append(f'<line x1="{xb:.1f}" y1="{ymap(stock_l3_lo):.1f}" x2="{xb:.1f}" y2="{ymap(stock_l3_hi):.1f}" stroke="#999" stroke-width="6" opacity="0.5"/>')
# curves
s.append(poly(stock, "#888"))
s.append(dots(stock, "#888"))
s.append(poly(srpf, "#1a7f37"))
s.append(dots(srpf, "#1a7f37"))
# legend
s.append(f'<rect x="{PX1-168}" y="{PY0+6}" width="150" height="52" fill="white" stroke="#ddd"/>')
s.append(f'<line x1="{PX1-158}" y1="{PY0+22}" x2="{PX1-138}" y2="{PY0+22}" stroke="#888" stroke-width="2.5"/><text x="{PX1-132}" y="{PY0+26}" fill="#333">stock (FCFS)</text>')
s.append(f'<line x1="{PX1-158}" y1="{PY0+42}" x2="{PX1-138}" y2="{PY0+42}" stroke="#1a7f37" stroke-width="2.5"/><text x="{PX1-132}" y="{PY0+46}" fill="#333">SRPF (reorder)</text>')
s.append('</svg>')
out = "\n".join(s)
print(out)

if __name__ == "__main__":
    pass
