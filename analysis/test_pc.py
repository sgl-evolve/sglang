import types, time
class MockNode:
    def __init__(self,i): self.id=i
class MockTree:
    def __init__(self,root): self.root_node=root; self.inc=0; self.dec=0
    def inc_host_lock_ref(self, node):
        self.inc+=1
        class R:
            def to_dec_params(self): return ("p",node.id)
        return R()
    def dec_host_lock_ref(self, node, params=None): self.dec+=1
class MockReq:
    def __init__(self,pref,hh,node): self.prefix_indices=list(range(pref)); self.host_hit_length=hh; self.last_host_node=node
import sys; sys.argv=["x"]
from sglang.srt.managers import scheduler as S
f=types.SimpleNamespace()
for m in ["_valiant_pin_budget_tokens","_valiant_pin_pending_prefix","_valiant_release_pin","_valiant_sweep_pins"]:
    setattr(f, m, types.MethodType(getattr(S.Scheduler, m), f))
root=MockNode(0); tree=MockTree(root)
f.tree_cache=tree; f.valiant_pin_fraction=0.5; f._valiant_pin_budget=100000
f._valiant_pin_tokens=0; f._valiant_pinned_reqs=set(); f.waiting_queue=[]
f.valiant_pc_enable=True; f.valiant_pc_horizon_s=0.4   # short horizon for test
r1=MockReq(300,100,MockNode(1))
f.waiting_queue.append(r1); f._valiant_pin_pending_prefix(r1)
assert len(f._valiant_pinned_reqs)==1 and tree.inc==1, "pin at enqueue"
# admit r1 (remove from waiting). pc: should NOT release yet (set expiry)
f.waiting_queue=[]
f._valiant_sweep_pins()
assert len(f._valiant_pinned_reqs)==1 and tree.dec==0, f"pc: keep pin after admit (got dec={tree.dec})"
assert getattr(r1,"_valiant_pc_expiry",None) is not None, "expiry set"
# sweep again immediately: still within horizon -> keep
f._valiant_sweep_pins()
assert tree.dec==0, "still within horizon"
# wait past horizon -> release
time.sleep(0.5)
f._valiant_sweep_pins()
assert len(f._valiant_pinned_reqs)==0 and tree.dec==1, f"pc: release after horizon (dec={tree.dec}, pinned={len(f._valiant_pinned_reqs)})"
assert f._valiant_pin_tokens==0
print("PC (post-completion) horizon test PASSED (inc=%d dec=%d)"%(tree.inc,tree.dec))
# Now test pc DISABLED = immediate release (v1 behavior)
tree2=MockTree(MockNode(0)); g=types.SimpleNamespace()
for m in ["_valiant_pin_budget_tokens","_valiant_pin_pending_prefix","_valiant_release_pin","_valiant_sweep_pins"]:
    setattr(g, m, types.MethodType(getattr(S.Scheduler, m), g))
g.tree_cache=tree2; g.valiant_pin_fraction=0.5; g._valiant_pin_budget=100000
g._valiant_pin_tokens=0; g._valiant_pinned_reqs=set(); g.waiting_queue=[]
g.valiant_pc_enable=False; g.valiant_pc_horizon_s=20
r2=MockReq(300,100,MockNode(2)); g.waiting_queue.append(r2); g._valiant_pin_pending_prefix(r2)
g.waiting_queue=[]; g._valiant_sweep_pins()
assert len(g._valiant_pinned_reqs)==0 and tree2.dec==1, "pc OFF -> immediate release (v1 behavior)"
print("PC-disabled = v1 immediate-release behavior PASSED")
print("ALL PC TESTS PASSED")
