#!/usr/bin/env python3
"""Compare a version's lossless fingerprint to the reference. Exit 0 if identical (LOSSLESS),
1 if any generated text differs (REGRESSION). Usage: compare_fp.py <reference.json> <version.json>
"""
import json, sys


def main():
    ref = json.load(open(sys.argv[1]))
    ver = json.load(open(sys.argv[2]))
    keys = sorted(set(ref) | set(ver), key=lambda x: int(x) if x.isdigit() else x)
    mismatches = 0
    for k in keys:
        r = ref.get(k, {})
        v = ver.get(k, {})
        for pass_key in ("p1", "p2"):
            if r.get(pass_key) != v.get(pass_key):
                mismatches += 1
                print(f"MISMATCH prompt {k} [{pass_key}]:")
                print(f"  ref: {repr((r.get(pass_key) or '')[:120])}")
                print(f"  ver: {repr((v.get(pass_key) or '')[:120])}")
    if mismatches == 0:
        print(f"LOSSLESS ✓ — all {len(keys)} prompts (cold+warm) match reference exactly")
        sys.exit(0)
    print(f"REGRESSION ✗ — {mismatches} mismatches")
    sys.exit(1)


if __name__ == "__main__":
    main()
