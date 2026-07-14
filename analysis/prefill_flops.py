#!/usr/bin/env python3
"""floyd: prefill FLOP breakdown for the Qwen3.5-122B-A10B hybrid model.

Question: how much of a long document's prefill compute is O(n^2) full-attention
(the part context-parallelism most obviously accelerates) vs O(n) per-token work
(projections, GatedDeltaNet linear-attention, MoE FFN)?

Architecture (from HF config text_config): 48 layers = 36 linear_attention
(GatedDeltaNet) + 12 full_attention (full_attention_interval=4); hidden 3072;
32 attn heads x 256 head_dim (GQA kv_heads=2); MoE 256 experts, 8/tok,
moe_intermediate 1024 + shared 1024.

IMPORTANT on CP: context parallelism splits the SEQUENCE across ranks, so ALL
per-token work (FFN/MoE, projections, linear-attn in/out) parallelizes; only
token-mixing ops need cross-rank comms -- attention (ring attention) and the
GatedDeltaNet recurrence (a chunked cross-device state scan, O(cp) serial
combine). So CP is NOT Amdahl-capped by the O(n) fraction; prefill parallelizes
near-linearly. The O(n^2) attention fraction below is reported only to show that
the biggest documents (where the p99 floor lives) are increasingly
attention-heavy, i.e. exactly where parallelizing prefill helps most.
"""
d=3072; L=48; L_full=12; L_lin=36
n_heads=32; head_dim=256; kv_heads=2
qdim=n_heads*head_dim; kvdim=kv_heads*head_dim
moe_inter=1024; experts_per_tok=8; shared_inter=1024
lin_k_heads=16; lin_k_dim=128; lin_v_heads=64; lin_v_dim=128; conv_k=4

def breakdown(n):
    qkv = 2*n*d*(qdim+2*kvdim); oproj = 2*n*qdim*d
    attn_n2 = 2*(2*n*n*qdim)/2                       # QK^T + AV, causal
    inproj = 2*n*d*(lin_k_heads*lin_k_dim*2 + lin_v_heads*lin_v_dim)
    conv   = 2*n*(lin_k_heads*lin_k_dim*2+lin_v_heads*lin_v_dim)*conv_k
    recur  = 2*n*lin_v_heads*lin_v_dim*lin_k_dim
    outp   = 2*n*(lin_v_heads*lin_v_dim)*d
    moe    = 2*n*d*moe_inter*3*experts_per_tok + 2*n*d*shared_inter*3
    attn_total  = L_full*attn_n2                                    # O(n^2), token-mixing
    perTok      = L_full*(qkv+oproj) + L_lin*(inproj+conv+recur+outp) + L*moe   # O(n)
    return attn_total, perTok, attn_total+perTok

if __name__ == "__main__":
    print(f"{'n(tok)':>8} {'attn O(n^2)':>12} {'perTok O(n)':>12} {'attn%':>7}")
    for n in [1000,5000,12800,18700,33200,67200,190900]:
        a,p,t=breakdown(n)
        print(f"{n:>8} {a/1e12:>10.2f}T {p/1e12:>10.2f}T {a/t*100:>6.1f}%")
    print("\nAttention share of prefill FLOPs: ~21% @ median doc (18.7k), ~49% @ p99 (67k),")
    print("~73% @ max (190.9k). Biggest cold docs (the p99 TTFT floor) are attention-dominated,")
    print("so parallelizing their prefill (CP) would cut the tail most. CP splits the sequence,")
    print("so the O(n) fraction parallelizes too; the GatedDeltaNet recurrence needs a chunked")
    print("cross-device scan (O(cp) serial). => CP would help; it is simply NOT IMPLEMENTED for")
    print("this hybrid model in sglang v0.31 (attn_cp_size stays 1; gated to DSA/MHA).")
