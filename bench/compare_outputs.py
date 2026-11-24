#!/usr/bin/env python3
"""compare_outputs.py A.json B.json : compare two sanity.py runs (or two bench_inkling.py runs) prompt by prompt.
Reports identical outputs and, for differing ones, the first differing character offset."""
import json, sys
def load(p):
    d = json.load(open(p))
    if isinstance(d, list):
        return {i: (x["reasoning"] + "\n<content>\n" + x["content"]) for i, x in enumerate(d)}
    out = {}
    for cat, e in d.get("types", {}).items():
        for it in e["c1"]["items"]:
            out[it["id"]] = it["output"]
    return out
A, B = load(sys.argv[1]), load(sys.argv[2])
same = 0
for k in A:
    if k not in B: continue
    a, b = A[k], B[k]
    if a == b: same += 1; continue
    i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    print(f"  {k}: differ at char {i} of {len(a)}/{len(b)}: A {a[max(0,i-30):i+40]!r}\n{'':10}B {b[max(0,i-30):i+40]!r}")
print(f"identical: {same}/{len([k for k in A if k in B])}")
