#!/usr/bin/env python3
"""Paper 2 §3 deepening: WHY does the token-linear capacity law C=K/(1-h) hold despite
the model's O(D^2) full-attention? Because prefill cost is MoE-FFN-dominated (linear in
tokens) up to a crossover doc length D*, above which the 12 full-attention layers' O(D^2)
term takes over. We compute D* from the real Qwen3.5-122B-A10B dims and compare to the
document-size distribution: the bulk of token mass is below D* (linear regime => K invariant);
only the extreme tail (p99.9+) is quadratic (and that is also where the feasibility margin
shrinks, Paper 2 §4). This is a HYBRID-architecture effect: 36 linear-attn layers keep the
aggregate linear; a dense model (all 48 full-attn) would cross over ~4x sooner."""
import json, ast

# --- exact dims (text_config, Qwen3.5-122B-A10B-FP8) ---
d_model=3072; L=48; L_full=12; L_lin=36
n_qh=32; head_dim=256; n_kvh=2
q_dim=n_qh*head_dim            # 8192
kv_dim=n_kvh*head_dim          # 512
moe_interm=1024; n_active=8; n_shared=1  # experts active per token (+1 shared)
lin_kh=16; lin_kd=128; lin_vh=64; lin_vd=128
MAC=2

# --- LINEAR (token-proportional) prefill FLOPs/token, independent of position D ---
ffn = L * (n_active+n_shared) * 3 * MAC * d_model * moe_interm      # SwiGLU: gate,up,down
attn_proj_full = L_full * MAC * (d_model*(q_dim+2*kv_dim) + q_dim*d_model)  # QKV + O
lin_q=d_model*(lin_kh*lin_kd); lin_k=d_model*(lin_kh*lin_kd); lin_v=d_model*(lin_vh*lin_vd)
lin_o=(lin_vh*lin_vd)*d_model
attn_proj_lin = L_lin * MAC * (lin_q+lin_k+lin_v+lin_o)
LIN = ffn + attn_proj_full + attn_proj_lin
# cross-check vs A10B active params (2 FLOP/param/token)
LIN_active = 2*10e9

# --- QUADRATIC full-attention FLOPs/token at position D (QK^T + AV over D keys) ---
# per full-attn layer per query token: 2(QK,AV) * MAC * n_qh * head_dim * D
QUAD_per_D = L_full * 2 * MAC * n_qh * head_dim   # * D

print(f"LINEAR prefill FLOPs/token (dim-derived) = {LIN:.3e}")
print(f"   (FFN {ffn:.2e} + full-attn-proj {attn_proj_full:.2e} + linear-attn {attn_proj_lin:.2e})")
print(f"LINEAR prefill FLOPs/token (A10B active-param) = {LIN_active:.3e}")
print(f"QUADRATIC coeff = {QUAD_per_D:.3e} * D  FLOPs/token")
for name,lin in [("dim-derived",LIN),("A10B-active",LIN_active)]:
    Dstar = lin/QUAD_per_D
    print(f"  crossover D* ({name}) = {Dstar:,.0f} tokens")
# dense-model counterfactual: all 48 layers full-attn -> quad coeff x4, D* /~ (48/12)*(cost ratio)
QUAD_dense = L * 2 * MAC * n_qh * head_dim
print(f"  DENSE counterfactual (all {L} full-attn): D*_dense = {LIN/QUAD_dense:,.0f} tok (hybrid extends linear regime {QUAD_dense/QUAD_per_D:.0f}x)")

# --- document uncached-work distribution (stock hit) vs D* ---
MIX="/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"; CH=4.0
U=[]
for line in open(MIX):
    r=json.loads(line); qp=r.get("qa_pairs",""); doc=r.get("input","")
    if qp=="none" or (isinstance(qp,(list,str)) and len(qp)==0): turns=["Input: "+doc+" Q: sum"]
    else:
        try: qa=ast.literal_eval(qp) if isinstance(qp,str) else qp
        except Exception: continue
        turns=[("Input: "+doc+" Q: "+str(q.get("Q",""))) if i==0 else str(q.get("Q","")) for i,q in enumerate(qa)]
    committed=0
    for ti,p in enumerate(turns):
        newq=max(1,int(len(p)/CH)); U.append(newq if ti==0 else committed*0+newq); committed+=newq
Us=sorted(U); n=len(Us)
print("\nDoc uncached-work U vs crossover D* (~35-51K):")
for q,lab in [(.5,'p50'),(.9,'p90'),(.99,'p99'),(.999,'p99.9'),(1.0,'max')]:
    u=Us[min(n-1,int(q*n)) if q<1 else n-1]
    reg = "LINEAR (FFN-dominated)" if u<35400 else ("crossover" if u<50900 else "QUADRATIC (attn-dominated)")
    print(f"  {lab:6} U={u:7d} -> {reg}")
# fraction of token mass below crossover
tot=sum(Us)
for Dstar in (35400,50900):
    below=sum(u for u in Us if u<Dstar)
    print(f"  token mass below D*={Dstar}: {100*below/tot:.1f}%  (=> aggregate cost ~linear => K invariant)")
