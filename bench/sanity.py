#!/usr/bin/env python3
"""sanity.py BASE MODEL OUT.json : 8 fixed prompts, temp 0, reasoning_effort none, max 800 tokens, non-stream.
Saves reasoning + content + usage so two configs can be diffed token for token (compare_sanity.py)."""
import json, sys, time, urllib.request
BASE, MODEL, OUT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
PROMPTS = [
 "Write a Python function roman_to_int(s) converting a Roman numeral string to an integer. Return only one ```python code block.",
 "Write a Python function merge_intervals(intervals) that merges overlapping [start, end] intervals and returns the merged list sorted by start. Return only one ```python code block.",
 "What is the remainder when 7^100 is divided by 5? Give the final answer on the last line as ANSWER: <n>",
 "A store sells notebooks at $3.40 each. Maya buys 7 notebooks and pays with a $50 bill. Then she returns 2 notebooks for a full refund. How much money does she have left from the $50? Give the final answer on the last line as ANSWER: <number>",
 "Two trains 300 km apart travel toward each other at 70 km/h and 80 km/h. A bird flying at 120 km/h goes back and forth between them until they meet. How many km does the bird fly? Final line: ANSWER: <n>",
 "Explain how a refrigerator works to a 12-year-old, in about 150 words.",
 "What is the capital of Australia? Answer in one word.",
 "List the numbers from 1 to 60 separated by commas. Output only the numbers.",
]
res = []
for p in PROMPTS:
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 800,
            "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "none"}}
    t = time.time(); d = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    m = d["choices"][0]["message"]
    res.append({"prompt": p, "reasoning": m.get("reasoning") or m.get("reasoning_content") or "", "content": m.get("content") or "",
                "usage": d["usage"], "finish": d["choices"][0]["finish_reason"], "wall_s": round(time.time() - t, 2)})
    print(f"{d['usage']['completion_tokens']:4} tok  {res[-1]['finish']:6} {res[-1]['wall_s']:5.1f}s  content: {res[-1]['content'][-90:]!r}")
json.dump(res, open(OUT, "w"), indent=1)
