#!/usr/bin/env python3
"""Long-context needle check for the Inkling 3090 lane.

Builds varied synthetic filler (log-style sentences) to TARGET prompt tokens
(measured with the server's /tokenize), hides a passphrase at DEPTH, asks for
it at the end. Streams the answer: reports prompt tokens, TTFT (prefill time),
prefill tok/s, decode tok/s after the long context, and PASS/FAIL.
usage: needle.py BASE_URL MODEL TARGET_TOKENS [depth=0.5]
"""
import json, random, sys, time, urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
TARGET = int(sys.argv[3])
DEPTH = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
ROOT = BASE[:-3] if BASE.endswith("/v1") else BASE

CITIES = ["Dothan", "Decatur", "Lawrenceville", "Atlanta", "Savannah", "Macon", "Athens", "Augusta",
          "Columbus", "Valdosta", "Albany", "Marietta", "Rome", "Tifton", "Brunswick", "Dalton"]
ITEMS = ["boots", "sandals", "runners", "loafers", "slides", "cleats", "sneakers", "mules", "clogs", "trainers"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
VERBS = ["shipped", "received", "returned", "counted", "restocked", "priced", "photographed", "tagged"]


def sentence(rng):
    return (f"Log {rng.randint(10000, 99999)}: the {rng.choice(CITIES)} store {rng.choice(VERBS)} "
            f"{rng.randint(2, 480)} pairs of {rng.choice(ITEMS)} (lot {rng.randint(100, 9999)}) on "
            f"{rng.choice(DAYS)} at {rng.randint(1, 12)}:{rng.randint(0, 59):02d}, bin {rng.choice('ABCDEFGH')}{rng.randint(1, 60)}.")


def post(path, payload, timeout=60):
    req = urllib.request.Request(ROOT + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def count_tokens(text):
    r = json.loads(post("/tokenize", {"model": MODEL, "prompt": text}, 300).read())
    return r["count"]


def main():
    rng = random.Random(1121)
    sample = " ".join(sentence(rng) for _ in range(400))
    per_sent = count_tokens(sample) / 400.0
    n = int((TARGET - 200) / per_sent)
    rng = random.Random(7)
    sents = [sentence(rng) for _ in range(n)]
    secret = "violet-otter-4729"
    needle = f"IMPORTANT NOTE: the secret passphrase for the vault is {secret}. Remember it."
    sents.insert(int(len(sents) * DEPTH), needle)
    question = ("\n\nAbove is a long inventory log. Somewhere in it is an important note with a secret passphrase. "
                "What is the secret passphrase? Answer with the passphrase only.")
    prompt = "Inventory log:\n" + " ".join(sents) + question
    payload = {
        "model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": True,
        "stream_options": {"include_usage": True}, "temperature": 0.0, "max_tokens": 256,
        "chat_template_kwargs": {"reasoning_effort": "none"},
    }
    t0 = time.time(); t_first = t_last = None; usage = None; content = []; reasoning = []
    with post("/v1/chat/completions", payload, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:].strip())
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                c = d.get("content") or ""; rs = d.get("reasoning_content") or d.get("reasoning") or ""
                if c or rs:
                    now = time.time(); t_first = t_first or now; t_last = now
                content.append(c); reasoning.append(rs)
    ans = "".join(content).strip()
    pt = usage["prompt_tokens"]; ct = usage["completion_tokens"]
    ttft = t_first - t0
    dec = (ct - 1) / (t_last - t_first) if t_last and t_last > t_first else 0.0
    ok = secret in ans
    print(json.dumps({"target": TARGET, "depth": DEPTH, "prompt_tokens": pt, "ttft_s": round(ttft, 1),
                      "prefill_tok_s": round(pt / ttft), "completion_tokens": ct, "decode_tok_s": round(dec, 1),
                      "answer": ans[:120], "reasoning_chars": len("".join(reasoning)),
                      "result": "PASS" if ok else "FAIL"}))


main()
