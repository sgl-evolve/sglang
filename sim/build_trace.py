#!/usr/bin/env python3
"""Tokenize mooncake_mix_v1 into a per-conversation turn-length trace (faithful to the
loogle multiturn loader), cached to conv_trace.json for offline cache simulation.

Per conversation: list of turns [(new_input_tokens, output_tokens), ...].
  turn 0     : "Input: <doc> Question: <Q0>"  -> big
  turn i>0   : "<Qi>"                          -> small
  output i   : len(tokenize(Ai))  (ignore_eos -> server generates ~this many)
Cache-relevant: at turn i the reusable prefix = sum_{k<i}(in_k+out_k); new work if resident = in_i;
                new work if evicted = whole prefix + in_i.
"""
import json, os, sys, time

DS = "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
TOK = "/rmeng_data/junyanch-data/hf_cache/hub/models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9"
OUT = os.path.join(os.path.dirname(__file__), "conv_trace.json")

os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer

def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
    print(f"tokenizer loaded in {time.time()-t0:.1f}s", flush=True)
    convs = []
    n = 0
    with open(DS) as f:
        for line in f:
            data = json.loads(line)
            turns = []
            qa = data.get("qa_pairs", "none")
            if qa == "none" or not qa or len(qa) == 0:
                prompt = "Input: " + data["input"] + " Question: Please summarize the input"
                out = data["input"][:1024]
                turns.append((len(tok.encode(prompt)), len(tok.encode(out))))
            else:
                pairs = eval(qa)
                for i, p in enumerate(pairs):
                    if i == 0:
                        prompt = "Input: " + data["input"] + " Question: " + p["Q"]
                    else:
                        prompt = p["Q"]
                    ilen = len(tok.encode(prompt))
                    olen = len(tok.encode(p["A"]))
                    if ilen < 4:  # loader prunes prompt<4 tok
                        continue
                    turns.append((ilen, olen))
            if turns:
                convs.append(turns)
            n += 1
            if n % 200 == 0:
                print(f"  {n} convs, {time.time()-t0:.1f}s", flush=True)
    json.dump(convs, open(OUT, "w"))
    # summary
    nturns = [len(c) for c in convs]
    ws = sum(sum(i + o for i, o in c) for c in convs)   # working set = all KV tokens
    turn0 = [c[0][0] for c in convs]
    print(f"\nconvs={len(convs)}  total_turns={sum(nturns)}  avg_turns={sum(nturns)/len(convs):.2f}"
          f"  max_turns={max(nturns)}")
    print(f"working_set_tokens={ws:,}  (L1+L2 ~10.7M => oversub {ws/10.7e6:.2f}x)")
    turn0.sort()
    print(f"turn0 input tokens: p50={turn0[len(turn0)//2]:,} p90={turn0[int(len(turn0)*0.9)]:,} "
          f"max={turn0[-1]:,} min={turn0[0]:,}")
    # per-conv total KV distribution
    convkv = sorted(sum(i + o for i, o in c) for c in convs)
    print(f"per-conv total KV: p50={convkv[len(convkv)//2]:,} p90={convkv[int(len(convkv)*0.9)]:,} "
          f"max={convkv[-1]:,}")
    print(f"wrote {OUT} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
