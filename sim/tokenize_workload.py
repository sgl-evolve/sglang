#!/usr/bin/env python3
"""Tokenize the fixed mooncake_mix_v1 workload into per-conversation turn chains,
replicating benchmark/hicache/data_processing.py::sample_loogle_requests exactly
(disable_shuffle=True as in the eval, enable_multiturn=True, fixed_output_len=None).

Output: sim/workload_tokens.json
  { "conversations": [ [ {"new_prompt_tok": int, "output_tok": int}, ... turns ], ... ],
    "meta": {...} }

new_prompt_tok = tokens of the NEW user message that turn (turn 0 = doc+Q0; turns 1+ = Qi).
output_tok     = tokens of that turn's reference answer (= max_tokens the bench requests).
Template/role-marker overhead is ignored (constant, tiny vs the document) — this preserves
the RELATIVE prefix-reuse structure, which is what the cache simulator compares.
"""
import json, os, sys, time
os.environ.setdefault("HF_HOME", "/rmeng_data/junyanch-data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"
DATA = "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
OUT = os.path.join(os.path.dirname(__file__), "workload_tokens.json")

def build_chats():
    """Mirror sample_loogle_requests (enable_multiturn=True, enable_shared_prefix=False)."""
    dataset = []
    with open(DATA) as f:
        for line in f:
            dataset.append(json.loads(line))
    new_dataset = []
    for data in dataset:
        chat = []
        qa = data.get("qa_pairs", "none")
        if "qa_pairs" not in data or qa == "none" or len(qa) == 0:
            chat.append(("Input: " + data["input"] + " Question: Please summarize the input",
                         data["input"][:1024]))
            new_dataset.append(chat)
        else:
            qa_pairs = eval(qa) if isinstance(qa, str) else qa
            for i, p in enumerate(qa_pairs):
                if i == 0:
                    chat.append(("Input: " + data["input"] + " Question: " + p["Q"], p["A"]))
                else:
                    chat.append((p["Q"], p["A"]))
            new_dataset.append(chat)
    return new_dataset  # disable_shuffle=True in eval -> no shuffle

def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    chats = build_chats()
    convs = []
    tot_in = tot_out = tot_turns = 0
    for ci, chat in enumerate(chats):
        turns = []
        for (prompt, completion) in chat:
            p_len = len(tok.encode(prompt))
            o_len = len(tok.encode(completion))
            if o_len < 1:
                o_len = 1
            turns.append({"new_prompt_tok": p_len, "output_tok": o_len})
            tot_in += p_len; tot_out += o_len; tot_turns += 1
        convs.append(turns)
        if ci % 200 == 0:
            print(f"  conv {ci}/{len(chats)}  ({time.time()-t0:.0f}s)", flush=True)
    meta = {"n_conversations": len(convs), "n_turns_total": tot_turns,
            "input_tokens": tot_in, "output_tokens": tot_out,
            "avg_turns": tot_turns/len(convs)}
    json.dump({"conversations": convs, "meta": meta}, open(OUT, "w"))
    print("META:", json.dumps(meta, indent=2))
    print("wrote", OUT, f"({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
