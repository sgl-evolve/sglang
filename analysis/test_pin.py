import types
# Minimal mock of the valiant pin logic extracted (mirror the methods)
class MockNode:
    def __init__(self,i): self.id=i
class MockTree:
    def __init__(self,root): self.root_node=root; self.inc=0; self.dec=0
    def inc_host_lock_ref(self, node):
        self.inc+=1
        class R:
            def to_dec_params(self): return ("p",node.id)
        return R()
    def dec_host_lock_ref(self, node, params=None):
        self.dec+=1
class MockReq:
    def __init__(self,pref_len,host_hit,node): 
        import torch
        self.prefix_indices=list(range(pref_len)); self.host_hit_length=host_hit; self.last_host_node=node
# import the real Scheduler methods by binding them to a bare object
import sys
sys.argv=["x"]
from sglang.srt.managers import scheduler as S
class Fake:
    pass
f=Fake()
# bind methods
for m in ["_valiant_pin_budget_tokens","_valiant_pin_pending_prefix","_valiant_release_pin","_valiant_sweep_pins"]:
    setattr(f, m, types.MethodType(getattr(S.Scheduler, m), f))
root=MockNode(0); tree=MockTree(root)
f.tree_cache=tree; f.valiant_pin_fraction=0.5
f._valiant_pin_budget=1000   # small budget for test
f._valiant_pin_tokens=0; f._valiant_pinned_reqs=set(); f.waiting_queue=[]
# len(prefix_indices)+host_hit_length tokens each
r1=MockReq(300,100,MockNode(1))   # 400 tok
r2=MockReq(300,100,MockNode(2))   # 400 tok
r3=MockReq(300,100,MockNode(3))   # 400 tok -> would exceed 1000 (800+400>1000) -> skip
for r in (r1,r2,r3): f.waiting_queue.append(r); f._valiant_pin_pending_prefix(r)
print(f"after 3 enqueues: pinned_reqs={len(f._valiant_pinned_reqs)} tokens={f._valiant_pin_tokens} inc={tree.inc}  (expect 2 pinned, 800 tok, inc=2)")
assert len(f._valiant_pinned_reqs)==2 and f._valiant_pin_tokens==800 and tree.inc==2
# admit r1: remove from waiting, sweep should release it
f.waiting_queue=[r2,r3]
f._valiant_sweep_pins()
print(f"after admitting r1: pinned={len(f._valiant_pinned_reqs)} tokens={f._valiant_pin_tokens} dec={tree.dec}  (expect 1 pinned,400,dec=1)")
assert len(f._valiant_pinned_reqs)==1 and f._valiant_pin_tokens==400 and tree.dec==1
# now r3 fits (400<=1000-400? budget 1000, tokens 400, +400=800<=1000) -> pin r3
f._valiant_pin_pending_prefix(r3)
print(f"after pinning r3: pinned={len(f._valiant_pinned_reqs)} tokens={f._valiant_pin_tokens}  (expect 2,800)")
assert len(f._valiant_pinned_reqs)==2 and f._valiant_pin_tokens==800
# double-pin r3 (idempotent guard)
f._valiant_pin_pending_prefix(r3)
assert f._valiant_pin_tokens==800, "double-pin should be no-op"
# drain queue -> sweep releases all
f.waiting_queue=[]; f._valiant_sweep_pins()
print(f"after drain: pinned={len(f._valiant_pinned_reqs)} tokens={f._valiant_pin_tokens} dec={tree.dec}  (expect 0,0,dec=3)")
assert len(f._valiant_pinned_reqs)==0 and f._valiant_pin_tokens==0 and tree.dec==3
print("ALL PIN BOOKKEEPING TESTS PASSED")
