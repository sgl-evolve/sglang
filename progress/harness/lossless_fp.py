#!/usr/bin/env python3
"""Capture a deterministic output fingerprint from a running sglang server, to gate losslessness.
Sends a fixed set of prompts (greedy, temp=0, ignore_eos) TWICE each (cold then warm) so the
prefix/HiCache path is exercised; records generated text. Any lossless change must reproduce the
unmodified reference fingerprint bit-for-bit.

Usage: lossless_fp.py --port 30000 --out fp.json [--max-new 48]
"""
import argparse, json, sys, urllib.request


def build_prompts():
    # Deterministic prompts. A long shared prefix (>> page_size*threshold) triggers prefix caching;
    # distinct suffixes produce distinct continuations. No randomness.
    base = (
        "The following is a technical passage about distributed key-value cache systems. "
        "Hierarchical caching tiers store attention keys and values across GPU memory, host RAM, "
        "and disk to serve long contexts under memory pressure. Prefetching, eviction, admission, "
        "and write policies all interact with the scheduler. "
    ) * 24  # ~ long shared prefix
    qs = [
        "Summarize the passage in one sentence.",
        "List three components mentioned.",
        "What problem does hierarchical caching solve?",
        "Define prefetching in this context.",
        "Continue the passage with one more sentence.",
        "Name the three storage tiers.",
        "What interacts with the scheduler?",
        "Give a synonym for eviction.",
    ]
    prompts = []
    for i, q in enumerate(qs):
        prompts.append(f"{base}\n\nDocument id {i}. Question: {q}\nAnswer:")
    # also a few short prompts
    prompts += [
        "Q: What is 2+2? A:",
        "Q: Name a primary color. A:",
        "Translate to French: good morning. ->",
    ]
    return prompts


def gen(port, prompt, max_new):
    payload = {
        "text": prompt,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True},
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://localhost:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    if isinstance(d, list):
        d = d[0]
    return d.get("text", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    a = ap.parse_args()
    prompts = build_prompts()
    fp = {}
    for i, p in enumerate(prompts):
        t1 = gen(a.port, p, a.max_new)   # cold
        t2 = gen(a.port, p, a.max_new)   # warm (cache hit)
        fp[str(i)] = {"p1": t1, "p2": t2}
    with open(a.out, "w") as f:
        json.dump(fp, f, indent=2)
    # warm must equal cold within a run (cache is lossless), else the server itself is broken.
    self_consistent = all(v["p1"] == v["p2"] for v in fp.values())
    print(f"FP captured: {len(fp)} prompts -> {a.out}  self_consistent(cold==warm)={self_consistent}")
    if not self_consistent:
        for k, v in fp.items():
            if v["p1"] != v["p2"]:
                print(f"  MISMATCH prompt {k}: cold!=warm")
        sys.exit(3)


if __name__ == "__main__":
    main()
