#!/usr/bin/env bash
# Parse [LAMPORT_INSTR] lines from a run's server.log -> latest cumulative snapshot.
# Usage: parse_instr.sh <run_dir_or_serverlog>
set -uo pipefail
p="${1:?usage: parse_instr.sh <run_dir|server.log>}"
[ -d "$p" ] && p="$p/server.log"
echo "== last 3 LAMPORT_INSTR snapshots in $p =="
grep -h "LAMPORT_INSTR" "$p" 2>/dev/null | tail -3
echo ""
echo "== decision hints (from last snapshot) =="
grep -h "LAMPORT_INSTR" "$p" 2>/dev/null | tail -1 | python3 -c '
import sys,re
line=sys.stdin.read()
if not line.strip():
    print("  (no LAMPORT_INSTR lines yet)"); sys.exit()
g=lambda k: (re.search(k+r"=([0-9.]+)",line) or [None,None])[1]
stranded_pct=re.search(r"STRANDED_tok=[0-9]+\(([0-9.]+)% of full",line)
aux_pct=re.search(r"aux_limited=[0-9]+\(([0-9.]+)% of hits",line)
mhe=g("mamba_host_evict_nodes"); fhe=re.search(r"full_host_evict=([0-9]+)",line)
mde=g("mamba_dev_evict_nodes")
sp=float(stranded_pct.group(1)) if stranded_pct else 0.0
print(f"  stranded_tok = {sp:.2f}% of full frontier | aux_limited hits = {aux_pct.group(1) if aux_pct else '?'}%")
print(f"  mamba_host_evict_nodes={mhe} mamba_dev_evict_nodes={mde} full_host_evict={fhe.group(1) if fhe else '?'}")
print()
if sp >= 5 and mhe and int(float(mhe))>0:
    print("  => HIGH stranding + mamba host eviction: EVICTION-DRIVEN stranding -> frontier-coupled retention mechanism")
elif sp >= 5:
    print("  => HIGH stranding, low mamba host evict: SPARSE-CHECKPOINT stranding -> denser/branch-point checkpointing")
elif sp >= 1:
    print("  => MODEST stranding -> co-management may give p99 tail win; verify tail composition")
else:
    print("  => LOW stranding -> PIVOT (elastic host pool sharing / latency-hiding / scheduling)")
'
