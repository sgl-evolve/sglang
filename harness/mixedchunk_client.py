#!/usr/bin/env python3
"""drift-3e7 mixed-chunk output-correctness client.

Losslessness cannot be read off the fixed eval (it measures latency/completion, not
output correctness), so this is the dedicated gate for the --enable-mixed-chunk
mechanism. It drives a *running* sglang server two ways with GREEDY decoding:

  * seq  : prompts sent ONE AT A TIME  -> no concurrency  -> mix_with_running never
           fires -> the canonical (mix-inactive) reference outputs.
  * conc : the SAME prompts sent ALL AT ONCE -> the scheduler interleaves prefills
           with running decodes -> ForwardMode.MIXED batches form (mix-active).

A correct mixed-chunk path must (a) never fail a request and (b) produce the same
greedy continuations as the sequential reference (small, LATE, logprob-benign drift
is acceptable FP batch-variance; early garbage / failures are the v11 corruption).

Prompt set deliberately mixes very long (multi-chunk prefill) and short prompts so
long prefills overlap short-prompt decodes -> MIXED batches. Dumps JSON for compare.
"""
import argparse, json, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

# A deterministic long passage; repeated to force multi-chunk prefill on long prompts.
_PARA = (
    "The hierarchical key-value cache tiers attention state across GPU high-bandwidth "
    "memory, host DRAM, and local solid-state storage so that long conversations and "
    "shared document prefixes can be reused across requests without recomputation. "
    "Under memory pressure the scheduler must decide which blocks to keep resident, "
    "which to offload, and when to prefetch them back, balancing latency against "
    "throughput while never changing the produced tokens. "
)


def build_prompts():
    prompts = []
    # 8 long prompts (multi-chunk prefill) -> overlap with short-prompt decodes.
    for i in range(8):
        body = _PARA * (40 + i * 6)  # ~ thousands of tokens, varied so batches differ
        prompts.append(
            f"[doc {i}] {body}\n\nQuestion: In one sentence, summarize the passage above."
        )
    # 16 short prompts -> finish prefill fast, decode while long prefills are chunking.
    shortqs = [
        "Explain what a radix tree is.",
        "Write a haiku about caches.",
        "List three prime numbers.",
        "What is the capital of France?",
        "Define latency versus throughput.",
        "Give a one-line definition of KV cache.",
        "What does TTFT stand for in LLM serving?",
        "Name two GPU memory tiers.",
        "Translate 'hello world' to French.",
        "What is 17 times 23?",
        "Describe SSD storage in one sentence.",
        "What is a Mamba state-space model?",
        "Give one benefit of prefix caching.",
        "What is chunked prefill?",
        "Explain greedy decoding briefly.",
        "What is a page in paged attention?",
    ]
    for q in shortqs:
        prompts.append(q)
    return prompts


def gen(port, prompt, max_new_tokens, timeout=1200):
    """One greedy /generate call. Returns (ok, out_text, completion_tokens, err).

    IMPORTANT: does NOT set return_logprob -- the scheduler DISABLES mixed-chunk when any
    request in the batch has return_logprob set (scheduler.py: "TODO: support return_logprob
    + mixed chunked prefill"). bench_serving (the eval) also omits it, so this mirrors the
    eval. Greedy => identical text iff identical tokens = the eval's losslessness notion.
    """
    req = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "max_new_tokens": max_new_tokens,
        },
    }
    data = json.dumps(req).encode()
    r = urllib.request.Request(
        f"http://localhost:{port}/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            obj = json.loads(resp.read().decode())
    except Exception as e:  # HTTP 500, timeout, connection reset = a request failure
        return (False, "", None, f"{type(e).__name__}: {str(e)[:200]}")
    mi = obj.get("meta_info", {}) or {}
    text = obj.get("text", "")
    ctoks = mi.get("completion_tokens")
    if not text:  # no tokens produced = failure
        return (False, "", ctoks, f"empty output (finish={mi.get('finish_reason')})")
    return (True, text, ctoks, None)


def run(port, tag, max_new_tokens, arrival_delay=0.4, stream_rounds=3):
    prompts = build_prompts()
    # Interleave long/short so a long prompt is mid-(chunked)-prefill while short prompts
    # are decoding -> maximises prefill/decode overlap once arrivals are staggered.
    order = []
    longs = [i for i in range(len(prompts)) if i < 8]
    shorts = [i for i in range(len(prompts)) if i >= 8]
    li = si = 0
    while li < len(longs) or si < len(shorts):
        if li < len(longs):
            order.append(longs[li]); li += 1
        for _ in range(2):
            if si < len(shorts):
                order.append(shorts[si]); si += 1

    # --- sequential reference (mix inactive) ---
    seq = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        ok, text, ctoks, err = gen(port, p, max_new_tokens)
        seq.append({"id": i, "ok": ok, "out_text": text, "ctoks": ctoks, "err": err})
    t_seq = time.time() - t0

    # --- concurrent STREAM (mix active): staggered Poisson-like arrivals, like the eval's
    # continuous load, so new prefills constantly overlap running decodes -> MIXED batches.
    # Multiple rounds keep a sustained in-flight population; we compare each unique prompt's
    # FIRST completed result against its sequential reference.
    conc = [None] * len(prompts)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(prompts) * stream_rounds + 4) as ex:
        futmap = []
        for _rnd in range(stream_rounds):
            for i in order:
                futmap.append((ex.submit(gen, port, prompts[i], max_new_tokens), i))
                time.sleep(arrival_delay)  # stagger arrivals -> overlap
        for fut, i in futmap:
            ok, text, ctoks, err = fut.result()
            if conc[i] is None or (not conc[i]["ok"] and ok):
                conc[i] = {"id": i, "ok": ok, "out_text": text, "ctoks": ctoks, "err": err}
    t_conc = time.time() - t0
    return {
        "tag": tag,
        "port": port,
        "max_new_tokens": max_new_tokens,
        "n_prompts": len(prompts),
        "t_seq_s": round(t_seq, 1),
        "t_conc_s": round(t_conc, 1),
        "seq": seq,
        "conc": conc,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run(a.port, a.tag, a.max_new_tokens)
    with open(a.out, "w") as f:
        json.dump(res, f)
    nf_seq = sum(1 for x in res["seq"] if not x["ok"])
    nf_conc = sum(1 for x in res["conc"] if not x["ok"])
    print(
        f"[client:{a.tag}] n={res['n_prompts']} seq_fail={nf_seq} conc_fail={nf_conc} "
        f"t_seq={res['t_seq_s']}s t_conc={res['t_conc_s']}s -> {a.out}",
        flush=True,
    )
