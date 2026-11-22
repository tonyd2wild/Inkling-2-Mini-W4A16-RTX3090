#!/usr/bin/env python3
"""Single-stream decode benchmark for the Inkling 3090 lane.

Same 6 prompts every run, temperature 0, reasoning_effort none (as the 69 eval),
streaming. Decode tok/s = (completion_tokens - 1) / (t_last - t_first).
Spec-decode acceptance from the server's Prometheus /metrics deltas.
usage: bench_decode.py BASE_URL MODEL [max_tokens] [label]
"""
import json, re, sys, time, urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 512
LABEL = sys.argv[4] if len(sys.argv) > 4 else ""

PROMPTS = [
    "Write a Python function that parses an ISO-8601 date string and returns the weekday name, with a docstring and three unit tests.",
    "Explain how a refrigerator works to a 12-year-old, in about 300 words.",
    "List 15 practical tips for keeping a small retail sneaker store organized, one sentence each.",
    "A train leaves at 3:40 PM going 72 mph; a second leaves the same station at 4:10 PM going 90 mph on the same track. When does the second catch the first? Show the steps.",
    "Write a short product description (about 200 words) for a waterproof hiking boot, then give 5 bullet-point specs.",
    "Translate into Spanish and French: 'The store opens at ten tomorrow, and the new shoes arrive on Friday morning.' Then explain two grammar differences between the translations.",
]


def metrics():
    try:
        txt = urllib.request.urlopen(BASE.replace("/v1", "") + "/metrics", timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for name in ("spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
                 "spec_decode_num_accepted_tokens_total"):
        tot = 0.0
        for m in re.finditer(r"^vllm:%s\{[^}]*\} ([0-9.e+]+)$" % name, txt, re.M):
            tot += float(m.group(1))
        out[name] = tot
    return out


def run(prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "max_tokens": MAXTOK,
        "chat_template_kwargs": {"reasoning_effort": "none"},
    }
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); t_first = t_last = None; usage = None; text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece:
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    text.append(d.get("content") or "")
    n = usage["completion_tokens"] if usage else 0
    dec = (n - 1) / (t_last - t_first) if t_first and t_last and t_last > t_first else 0.0
    return dict(tokens=n, ttft=(t_first - t0) if t_first else None, decode=dec, text="".join(text))


def main():
    m0 = metrics()
    rows = []
    for i, p in enumerate(PROMPTS):
        r = run(p)
        rows.append(r)
        print(f"  p{i+1}: {r['tokens']:4d} tok  ttft {r['ttft']*1000:6.0f} ms  decode {r['decode']:6.1f} tok/s  | {r['text'][:70]!r}")
    m1 = metrics()
    toks = sum(r["tokens"] for r in rows)
    span = sum((r["tokens"] - 1) / r["decode"] for r in rows if r["decode"])
    decs = sorted(r["decode"] for r in rows)
    print(f"{LABEL} aggregate decode {toks / span:.1f} tok/s, median {decs[len(decs)//2]:.1f}, total {toks} tok")
    if m1 and m0 and m1.get("spec_decode_num_drafts_total", 0) > m0.get("spec_decode_num_drafts_total", 0):
        drafts = m1["spec_decode_num_drafts_total"] - m0["spec_decode_num_drafts_total"]
        dt = m1["spec_decode_num_draft_tokens_total"] - m0["spec_decode_num_draft_tokens_total"]
        acc = m1["spec_decode_num_accepted_tokens_total"] - m0["spec_decode_num_accepted_tokens_total"]
        print(f"{LABEL} spec: drafts {drafts:.0f}, draft tokens {dt:.0f}, accepted {acc:.0f} -> "
              f"acceptance {acc/dt*100:.1f}%, mean accepted/draft {acc/drafts:.2f} "
              f"(tokens per target step {1+acc/drafts:.2f})")
    json.dump(rows, open(f"/tmp/bench_{LABEL or 'run'}.json", "w"))


main()
