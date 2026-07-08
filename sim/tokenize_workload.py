#!/usr/bin/env python3
"""Tokenize mooncake_mix_v1.jsonl into per-conversation turn token structure,
exactly mirroring bench_serving's loogle loader (data_processing.sample_loogle_requests
+ common_filter_chat, min_prompt_len=4, no shuffle, multiturn).

Output (cached): sim/workload_tokens.json
  convs: list of conversations, each a list of turns; each turn = [prompt_len, output_len]
  where prompt_len = tokens of the ISOLATED turn text (turn0 = "Input: "+doc+" Question: "+Q0;
  turn i>=1 = Qi), output_len = tokens of the answer Ai.
The cumulative prefix at turn i (reuse target) = sum over k<i of (prompt_len_k + output_len_k),
plus per-message chat-template overhead (added in the simulator).
"""
import json, ast, os, sys, time

DATASET = "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
TOK_DIR = "/rmeng_data/junyanch-data/hf_cache/hub/models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9"
OUT = os.path.join(os.path.dirname(__file__), "workload_tokens.json")

def main():
    from transformers import AutoTokenizer
    os.environ["HF_HUB_OFFLINE"] = "1"
    tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True)
    dataset = []
    with open(DATASET) as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    convs = []
    t0 = time.time()
    for di, data in enumerate(dataset):
        chat = []  # list of (prompt_text, answer_text)
        qp = data.get("qa_pairs", "")
        has_qa = not (qp == "none" or qp == "" or (isinstance(qp, list) and len(qp) == 0))
        if not has_qa:
            chat.append(("Input: " + data["input"] + " Question: Please summarize the input",
                         data["input"][:1024]))
        else:
            try:
                qa_pairs = ast.literal_eval(qp) if isinstance(qp, str) else qp
            except Exception:
                qa_pairs = []
            for i, qa in enumerate(qa_pairs):
                if i == 0:
                    chat.append(("Input: " + data["input"] + " Question: " + qa["Q"], qa["A"]))
                else:
                    chat.append((qa["Q"], qa["A"]))
        # tokenize turns; apply min_prompt_len=4 filter (skip turns with prompt_len<4)
        turns = []
        for (p, a) in chat:
            pl = len(tok.encode(p))
            ol = len(tok.encode(a))
            if pl < 4:
                continue
            turns.append([pl, ol])
        if turns:
            convs.append(turns)
        if di % 200 == 0:
            print(f"  {di}/{len(dataset)} convs, {time.time()-t0:.1f}s", file=sys.stderr)
    # stats
    n_turns = sum(len(c) for c in convs)
    total_prompt = sum(t[0] for c in convs for t in c)
    total_out = sum(t[1] for c in convs for t in c)
    print(f"convs={len(convs)} turns={n_turns} sum_isolated_prompt_tok={total_prompt} sum_out_tok={total_out}")
    json.dump({"convs": convs}, open(OUT, "w"))
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
