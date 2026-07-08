#!/usr/bin/env python3
"""Extract the EXACT loogle-multiturn trace (per-conversation per-turn token lengths)
that eval.sh feeds bench_serving, so we can drive an offline cache simulator.

Mirrors data_processing.sample_loogle_requests + common_filter_chat with the
frozen eval args: --dataset-name loogle --enable-multiturn --disable-shuffle
--num-prompts 1553  (enable_shared_prefix=False, fixed_output_len=None, min_prompt_len=4).

Output: sim/trace.json = list of conversations; each conversation is a list of
turns [ {prompt_len, output_len} ], in the exact order eval serves them.
"""
import json, sys, os
os.environ.setdefault("HF_HOME", "/rmeng_data/junyanch-data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer

DATASET = "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"
NUM_REQUESTS = 1553
MIN_PROMPT_LEN = 4
OUT = os.path.join(os.path.dirname(__file__), "trace.json")

def main():
    print("loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # ---- sample_loogle_requests (multiturn, no shared prefix, no shuffle) ----
    dataset = []
    with open(DATASET) as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    print(f"loaded {len(dataset)} docs", flush=True)

    new_dataset = []  # each = list of (prompt_text, completion_text)
    for data in dataset:
        chat = []
        if ("qa_pairs" not in data or data["qa_pairs"] == "none"
                or len(data["qa_pairs"]) == 0):
            chat.append(("Input: " + data["input"] + " Question: Please summarize the input",
                         data["input"][:1024]))
            new_dataset.append(chat)
        else:
            qa_pairs = eval(data["qa_pairs"])
            for i, qa in enumerate(qa_pairs):
                if i == 0:  # enable_shared_prefix=False -> only i==0 gets the doc
                    chat.append(("Input: " + data["input"] + " Question: " + qa["Q"], qa["A"]))
                else:  # enable_multiturn=True
                    chat.append((qa["Q"], qa["A"]))
            new_dataset.append(chat)
    # disable_shuffle=True -> no shuffle

    # ---- common_filter_chat (num_requests=1553, min_prompt_len=4) ----
    out = []
    l = 0
    input_tokens = 0
    output_tokens = 0
    done = False
    while l < NUM_REQUESTS and not done:
        made_progress = False
        for i in range(len(new_dataset)):
            if l == NUM_REQUESTS:
                break
            processed = []
            for (prompt, completion) in new_dataset[i]:
                plen = len(tok.encode(prompt))
                olen = len(tok.encode(completion))
                if plen < MIN_PROMPT_LEN:
                    continue
                input_tokens += plen
                output_tokens += olen
                processed.append({"prompt_len": plen, "output_len": olen})
            if processed:
                out.append(processed)
                l += 1
                made_progress = True
        if not made_progress:
            done = True  # avoid infinite loop if nothing passes filter
        if len(new_dataset) >= NUM_REQUESTS:
            break  # one pass suffices when we have enough distinct convs
        if l % 200 == 0:
            print(f"  ...{l} convs", flush=True)

    json.dump(out, open(OUT, "w"))
    n_turns = sum(len(c) for c in out)
    print(f"WROTE {OUT}: {len(out)} convs, {n_turns} turns, "
          f"input_tokens={input_tokens}, output_tokens={output_tokens}", flush=True)
    # quick structural stats
    import statistics as st
    turns_per = [len(c) for c in out]
    doc0 = [c[0]["prompt_len"] for c in out]
    print(f"turns/conv: mean={st.mean(turns_per):.2f} med={st.median(turns_per)} max={max(turns_per)}")
    print(f"turn0 prompt_len (doc): mean={st.mean(doc0):.0f} med={st.median(doc0):.0f} max={max(doc0)}")

if __name__ == "__main__":
    main()
