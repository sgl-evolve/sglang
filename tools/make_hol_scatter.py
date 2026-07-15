#!/usr/bin/env python3
# kleinrock: generate an inline SVG scatter of per-request (prompt_len vs TTFT) for Paper 1 §5.6 — the direct
# HOL evidence. Tiny prompts (left) stranded at high TTFT = head-of-line victims; heavy docs (right) sit low.
# Reads stock (reserve0) per-request dumps; deterministic subsample (no RNG) for legibility. Writes SVG to stdout.
import sys, csv, glob, math
files=[]
for a in sys.argv[1:]: files+=glob.glob(a)
pts=[]
for f in files:
    for r in csv.DictReader(open(f)):
        try:
            pl=float(r['prompt_len']); tt=float(r['ttft_ms'])/1000.0
            if pl>=1: pts.append((pl,tt))
        except: pass
pts.sort()
# deterministic subsample to ~700 points (every k-th), but keep ALL points with ttft>8s (the tail we care about)
tail=[p for p in pts if p[1]>8.0]
body=[p for p in pts if p[1]<=8.0]
body_s=body[::max(1,len(body)//400)]
tail_s=tail[::max(1,len(tail)//400)]
show=body_s+tail_s
W,H=660,400; ml,mr,mt,mb=70,20,30,55
pw,ph=W-ml-mr,H-mt-mb
xmin,xmax=math.log10(10),math.log10(200000)   # log prompt_len
ymin,ymax=0,45
def X(pl): return ml+(math.log10(max(10,pl))-xmin)/(xmax-xmin)*pw
def Y(tt): return mt+(1-(min(ymax,tt)-ymin)/(ymax-ymin))*ph
def col(pl): return "#c0392b" if pl<1000 else ("#7f8c8d" if pl<50000 else "#2471a3")
out=[]
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Georgia,serif" font-size="12">')
out.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>')
# axes
out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333"/>')
out.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>')
# x ticks (log): 10,100,1K,10K,100K
for v,lab in [(10,"10"),(100,"100"),(1000,"1K"),(10000,"10K"),(100000,"100K")]:
    x=X(v); out.append(f'<line x1="{x:.1f}" y1="{mt+ph}" x2="{x:.1f}" y2="{mt+ph+4}" stroke="#333"/>')
    out.append(f'<text x="{x:.1f}" y="{mt+ph+18}" text-anchor="middle" fill="#333">{lab}</text>')
# y ticks
for v in [0,8,15,30,45]:
    y=Y(v); out.append(f'<line x1="{ml-4}" y1="{y:.1f}" x2="{ml}" y2="{y:.1f}" stroke="#333"/>')
    out.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" fill="#333">{v}</text>')
# SLO line at 8s
ys=Y(8); out.append(f'<line x1="{ml}" y1="{ys:.1f}" x2="{ml+pw}" y2="{ys:.1f}" stroke="#c0392b" stroke-width="1.3" stroke-dasharray="6 3"/>')
out.append(f'<text x="{ml+pw-4}" y="{ys-5:.1f}" text-anchor="end" fill="#c0392b">8&#160;s SLO</text>')
# points
for pl,tt in show:
    out.append(f'<circle cx="{X(pl):.1f}" cy="{Y(tt):.1f}" r="1.7" fill="{col(pl)}" fill-opacity="0.55"/>')
# annotations
out.append(f'<text x="{ml+8}" y="{mt+14}" fill="#c0392b" font-weight="bold">head-of-line victims: tiny prompts, TTFT up to ~40&#160;s</text>')
out.append(f'<text x="{ml+pw}" y="{mt+ph-8}" text-anchor="end" fill="#2471a3">heavy docs (&#8805;50K): mostly low TTFT</text>')
# axis titles
out.append(f'<text x="{ml+pw/2}" y="{H-8}" text-anchor="middle" fill="#333">prompt length (tokens, log scale)</text>')
out.append(f'<text x="16" y="{mt+ph/2}" text-anchor="middle" fill="#333" transform="rotate(-90 16 {mt+ph/2})">TTFT (s)</text>')
out.append('</svg>')
sys.stdout.write("\n".join(out))
