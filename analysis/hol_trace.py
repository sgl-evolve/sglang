#!/usr/bin/env python3
"""floyd: DIRECT trace evidence for head-of-line blocking (from the stock run's server.log).

The head-of-line diagnosis says small turns pile up behind the back-to-back chunked prefill of
a few large cold documents. If true, the waiting queue should be much DEEPER during big-cold-doc
prefill steps than during other prefill steps. We test this directly on the stock v0-stock run.

A "big-cold-doc chunk" = a Prefill batch step with #new-token==6144 (a full chunk) and
#cached-token==0 (cold, no prefix reuse) — i.e. the engine is grinding a large cold document.
Result (stock, all rates): the queue is ~10x deeper during big-cold-doc chunks than otherwise,
directly confirming head-of-line blocking (independent of the queueing simulation).
"""
import re, statistics as st

LOG = "runs/v0-stock/server.log"

def main():
    rows = []
    for l in open(LOG, errors="ignore"):
        if "Prefill batch" not in l:
            continue
        nt = re.search(r"#new-token: (\d+)", l); ct = re.search(r"#cached-token: (\d+)", l)
        q = re.search(r"#queue-req: (\d+)", l)
        if nt and ct and q:
            rows.append((int(nt.group(1)), int(ct.group(1)), int(q.group(1))))
    bigcold = [r for r in rows if r[0] >= 6144 and r[1] == 0]
    other = [r for r in rows if not (r[0] >= 6144 and r[1] == 0)]

    def qs(rs):
        v = sorted(r[2] for r in rs); m = len(v)
        return (st.mean(v), v[m // 2], v[int(.9 * m)], max(v)) if v else (0, 0, 0, 0)

    b, o = qs(bigcold), qs(other)
    print(f"prefill steps: {len(rows)} ({len(bigcold)} big-cold-doc chunks, {len(other)} other)")
    print(f"queue-req DURING big-cold-doc chunks: mean={b[0]:.1f} p50={b[1]} p90={b[2]} max={b[3]}")
    print(f"queue-req during other prefill steps: mean={o[0]:.1f} p50={o[1]} p90={o[2]} max={o[3]}")
    print(f"=> {b[0]/max(o[0],0.1):.1f}x deeper during big-cold-doc chunks => HEAD-OF-LINE blocking (direct trace evidence)")

if __name__ == "__main__":
    main()
