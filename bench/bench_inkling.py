#!/usr/bin/env python3
"""bench_inkling.py <base_url> <served_model> <label> [--parts ttft,prefill,types] [--levels 1,2,3,4,5,6] [--out DIR]

Inkling DSpark vs DFlash2 bench, adapted from the house toolkit (glm53-exl3-repo/tools: bench_categories.py prompts +
graders, bench_ttft_fresh.py fresh-prompt TTFT, bench_prefill_len.py, bench_sweep.py counting prompt).
Everything: thinking off (chat_template_kwargs enable_thinking=false + reasoning_effort none), temperature 0, streaming.
Inkling still writes a short reasoning block at effort 0, so TTFT and decode tok/s count the first streamed token of
either kind; ttft_content_s (first answer token) is recorded too.

  ttft     streamed time to first token (reasoning or content) at c1..c6, fresh ~1.6K-token prompts (no prefix-cache hits), 8-token answers
  prefill  cold prefill tok/s at 1K / 8K / 32K prompt tokens (unique nonce first, so no prefix-cache hit), c1
  types    8 prompt types x 5 prompts: coding, math, json, html, prose, narrative, summary, counting
           c1: every prompt sequentially -> streamed TTFT, decode tok/s, auto-grade, spec acceptance per type
           c2..c6: per type, c concurrent streams (prompts of that type, cycled) -> aggregate tok/s, per-stream decode
           tok/s, acceptance per (type, c)
Spec metrics come from /metrics deltas (vllm:spec_decode_*). Every part also checks vllm:request_success_total so a
foreign request during the run is flagged ("foreign_requests").
"""
import argparse, concurrent.futures, json, os, random, re, statistics, subprocess, sys, tempfile, time, urllib.request
from html.parser import HTMLParser

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("label")
ap.add_argument("--parts", default="ttft,prefill,types")
ap.add_argument("--levels", default="1,2,3,4,5,6")
ap.add_argument("--types", default="")
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--prefill-sizes", default="1000,8000,32000")
ap.add_argument("--out", default="results")
a = ap.parse_args()
BASE = a.base.rstrip("/"); URL = BASE + "/v1/chat/completions"; MET = BASE + "/metrics"
LEVELS = [int(x) for x in a.levels.split(",")]
os.makedirs(a.out, exist_ok=True)
KW = {"enable_thinking": False, "reasoning_effort": "none"}

# ------------------------------------------------------------------ metrics
def metrics():
    try:
        txt = urllib.request.urlopen(MET, timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for name in ("spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
                 "spec_decode_num_accepted_tokens_total", "request_success_total"):
        out[name] = sum(float(m.group(1)) for m in re.finditer(r"^vllm:%s\{[^}]*\} ([0-9.e+]+)$" % name, txt, re.M))
    pos = {}
    for m in re.finditer(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\} ([0-9.e+]+)$', txt, re.M):
        pos[int(m.group(1))] = pos.get(int(m.group(1)), 0.0) + float(m.group(2))
    out["per_pos"] = pos
    return out

def spec_delta(m0, m1):
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in ("spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
                                                  "spec_decode_num_accepted_tokens_total", "request_success_total")}
    drafts, dtoks, acc = d["spec_decode_num_drafts_total"], d["spec_decode_num_draft_tokens_total"], d["spec_decode_num_accepted_tokens_total"]
    res = {"drafts": int(drafts), "accepted": int(acc), "requests": int(d["request_success_total"])}
    if drafts > 0:
        res["acceptance_length"] = round(1 + acc / drafts, 3)          # tokens per target step (bonus + accepted)
        res["draft_acceptance_rate"] = round(acc / dtoks, 4) if dtoks else None
        p0, p1 = m0.get("per_pos", {}), m1.get("per_pos", {})
        res["per_pos_rate"] = [round((p1.get(i, 0) - p0.get(i, 0)) / drafts, 3) for i in sorted(p1)]
    return res

# ------------------------------------------------------------------ prompts (from bench_categories.py, plus math + counting + own summary texts)
P = []
P += [("code1", "coding", "Write a Python function merge_intervals(intervals) that merges overlapping [start, end] intervals and returns the merged list sorted by start. Return only one ```python code block.",
       {"tests": "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]\nassert merge_intervals([[5,6],[1,2]])==[[1,2],[5,6]]"}),
      ("code2", "coding", "Write a Python function parse_duration(s) that converts strings like '1h30m15s', '45s', '2h', '3m', '1h1s' into total seconds (int). Units may be omitted but always appear in h, m, s order. Return only one ```python code block.",
       {"tests": "assert parse_duration('1h30m15s')==5415\nassert parse_duration('45s')==45\nassert parse_duration('2h')==7200\nassert parse_duration('3m')==180\nassert parse_duration('1h1s')==3601"}),
      ("code3", "coding", "Write a Python function top_k_frequent(words, k) returning the k most frequent words, most frequent first; ties broken alphabetically. Return only one ```python code block.",
       {"tests": "assert top_k_frequent(['b','a','c','a','b','a'],2)==['a','b']\nassert top_k_frequent(['x','y'],1)==['x']\nassert top_k_frequent(['z','z','y','y','x'],2)==['y','z']\nassert top_k_frequent([],3)==[]"}),
      ("code4", "coding", "This binary search has a bug that can make it loop forever. Return the corrected function only, as one ```python code block, same name and signature.\n\ndef binary_search(a, target):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] == target:\n            return mid\n        if a[mid] < target:\n            lo = mid\n        else:\n            hi = mid\n    return -1",
       {"tests": "assert binary_search([1,3,5,7,9],7)==3\nassert binary_search([1,3,5,7,9],2)==-1\nassert binary_search([],1)==-1\nassert binary_search([1],1)==0\nassert binary_search([1,2],2)==1\nassert binary_search([1,3,5,7,9],9)==4"}),
      ("code5", "coding", "Write a Python function roman_to_int(s) converting a Roman numeral string to an integer. Return only one ```python code block.",
       {"tests": "assert roman_to_int('XIV')==14\nassert roman_to_int('MCMXCIV')==1994\nassert roman_to_int('IX')==9\nassert roman_to_int('LVIII')==58\nassert roman_to_int('III')==3"})]
SUF = "\n\nShow your work step by step, then give your final answer on the last line exactly as: ANSWER: <answer>"
P += [("math1", "math", "Ana is twice as old as Ben. In 6 years Ana will be 1.5 times Ben's age. How old is Ben now?" + SUF, {"answer": "6"}),
      ("math2", "math", "A fair coin is flipped 4 times. What is the probability of exactly 2 heads? Give a fraction in lowest terms." + SUF, {"answer": "3/8", "alt": ["0.375"]}),
      ("math3", "math", "A store sells notebooks at $3.40 each. Maya buys 7 notebooks and pays with a $50 bill. Then she returns 2 notebooks for a full refund. How much money does she have left from the $50, in dollars?" + SUF, {"answer": "33"}),
      ("math4", "math", "Two trains 300 km apart travel toward each other at 70 km/h and 80 km/h. A bird flying at 120 km/h goes back and forth between them until they meet. How many km does the bird fly?" + SUF, {"answer": "240"}),
      ("math5", "math", "What is the remainder when 7^100 is divided by 5?" + SUF, {"answer": "1"})]
JS = "\n\nOutput only the JSON, no prose, no code fences."
P += [("json1", "json", "Extract into a JSON object with keys invoice_id (integer), customer (string), issued (ISO date string), amount (number), due_days (integer):\n\n\"Invoice 4471 was issued to Marta Ruiz on 2026-03-14 for $1,250.50, due in 30 days.\"" + JS,
       {"json": {"invoice_id": 4471, "customer": "Marta Ruiz", "issued": "2026-03-14", "amount": 1250.5, "due_days": 30}}),
      ("json2", "json", "Convert this CSV to a JSON array of objects with numeric qty and price:\n\nname,qty,price\napple,3,0.5\npear,2,0.75" + JS,
       {"json": [{"name": "apple", "qty": 3, "price": 0.5}, {"name": "pear", "qty": 2, "price": 0.75}]}),
      ("json3", "json", "Fix this invalid JSON and output only the corrected JSON:\n\n{'user': 'sam', 'tags': ['a', 'b',], 'active': True}" + JS,
       {"json": {"user": "sam", "tags": ["a", "b"], "active": True}}),
      ("json4", "json", "Return a JSON object mapping each city to the arithmetic mean of its temperatures:\n\n{\"Oslo\": [2, 4, 6], \"Cairo\": [30, 32], \"Lima\": [18, 19, 20, 21]}" + JS,
       {"json": {"Oslo": 4, "Cairo": 31, "Lima": 19.5}}),
      ("json5", "json", "Produce a JSON object with exactly these keys and values: id = integer 7, name = string \"Widget\", price = number 19.99, in_stock = boolean true, tags = array of strings [\"tools\", \"home\"]." + JS,
       {"json": {"id": 7, "name": "Widget", "price": 19.99, "in_stock": True, "tags": ["tools", "home"]}})]
HS = "\n\nOutput only the HTML."
P += [("html1", "html", "Write a complete HTML5 page: a <header> containing an <h1> that reads Spark Bench and a <nav> with three links (Home, Results, Method), then a <main> containing a <section id=\"summary\"> with one paragraph." + HS,
       {"html": {"doctype": True, "tags": {"header": 1, "h1": 1, "nav": 1, "main": 1, "section": 1, "p": 1}, "min_tags": {"a": 3}, "ids": ["summary"], "text": ["Spark Bench", "Home", "Results", "Method"]}}),
      ("html2", "html", "Write an HTML <table> with a <thead> row of Lane, Decode tok/s, TTFT s and a <tbody> with two rows: NVFP4, 64.0, 1.54 and EXL3, 61.5, 0.52." + HS,
       {"html": {"tags": {"table": 1, "thead": 1, "tbody": 1}, "min_tags": {"th": 3, "td": 6}, "text": ["NVFP4", "64.0", "1.54", "EXL3", "61.5", "0.52"]}}),
      ("html3", "html", "Write an HTML <form> with: a required email input with id=\"email\", a <select id=\"lane\"> with the options NVFP4 and EXL3, a <textarea id=\"notes\">, and a submit <button>." + HS,
       {"html": {"tags": {"form": 1, "select": 1, "textarea": 1, "button": 1}, "min_tags": {"option": 2}, "ids": ["email", "lane", "notes"], "attrs": [("input", "type", "email"), ("input", "required", None)]}}),
      ("html4", "html", "Write an accessible site navigation: a skip link <a href=\"#main\" class=\"skip\">Skip to content</a>, then <nav aria-label=\"Main\"> containing a <ul> of four <li> items each with an <a>." + HS,
       {"html": {"tags": {"nav": 1, "ul": 1}, "min_tags": {"li": 4, "a": 5}, "attrs": [("nav", "aria-label", "Main"), ("a", "class", "skip"), ("a", "href", "#main")]}}),
      ("html5", "html", "Write an HTML card component with an inline <style> block: a <div class=\"card\"> containing an <img> with an alt attribute, an <h2>, a <p>, and an <a class=\"btn\">." + HS,
       {"html": {"tags": {"style": 1, "img": 1, "h2": 1, "p": 1}, "min_tags": {"div": 1, "a": 1}, "attrs": [("div", "class", "card"), ("a", "class", "btn"), ("img", "alt", None)]}})]
P += [("prose1", "prose", "Explain what tensor parallelism is to a smart high-school student, in 120 to 180 words.", {"words": (120, 180)}),
      ("prose2", "prose", "Write a product description of 80 to 120 words for a compact AI workstation with 128 GB of unified memory. Do not use the words revolutionary, game-changing, or unleash.", {"words": (80, 120), "banned": ["revolutionary", "game-changing", "unleash"]}),
      ("prose3", "prose", "Write a professional email under 150 words politely declining a meeting request and proposing two specific alternative times next week.", {"words": (30, 150)}),
      ("prose4", "prose", "In 150 to 200 words, argue for or against benchmarking language models with synthetic prompts. Take one side clearly.", {"words": (150, 200)}),
      ("prose5", "prose", "In 100 to 150 words, explain in plain language why time to first token matters more than tokens per second for a chat assistant. Use exactly one concrete analogy.", {"words": (100, 150)})]
P += [("story1", "narrative", "Write a 150 to 250 word short story in first person that includes the words lantern, ledger and thunder, contains at least two lines of dialogue, and ends with a twist.", {"words": (150, 250), "must": ["lantern", "ledger", "thunder"], "quotes": 2}),
      ("story2", "narrative", "Write a 120 to 200 word fable with a talking fox. State the moral in the final sentence, starting with the word Moral:", {"words": (120, 200), "must": ["fox", "Moral:"]}),
      ("story3", "narrative", "Write a 200 to 300 word scene: two engineers arguing at 2 a.m. about a benchmark result. Mostly dialogue; no narration longer than one sentence at a time.", {"words": (200, 300), "quotes": 6}),
      ("story4", "narrative", "Write a 100 to 150 word micro-story told entirely in second person, present tense, set in a data center during a power outage.", {"words": (100, 150), "must": ["you"]}),
      ("story5", "narrative", "Write a 150 to 250 word story for children aged 6 to 8 about a robot learning to wait. Exactly three paragraphs.", {"words": (150, 250), "paragraphs": 3})]
DOC_A = """Speculative decoding speeds up text generation without changing what the model writes. A small, fast draft model guesses several upcoming tokens at once, and the large target model then checks all of those guesses in a single forward pass. Every guess that matches what the target would have produced on its own is kept, and the first wrong guess is replaced by the target's own token. Because the target verifies every token, the output is the same as normal decoding when sampling is greedy, and it keeps the same distribution when sampling is random.
The speed-up depends on two numbers: how many guessed tokens are accepted on average per verification step, and how much a verification step with several tokens costs compared with a normal one-token step. On a memory-bound GPU, checking eight tokens costs only a little more than checking one, so an acceptance length of three or four can more than double decode speed. The draft model itself must be cheap; if drafting takes as long as a target step, the gain disappears.
Acceptance is not the same for every kind of text. Code, structured data and lists are highly predictable, so drafts are accepted often. Open-ended prose and stories are harder to predict, so fewer guesses survive. Math explanations sit in between: the wording is formulaic but the numbers are not. Batch size matters too. With many requests in flight, the GPU is busier, verification is no longer almost free, and the benefit of speculation shrinks.
Block-diffusion drafters predict a whole block of future tokens in one pass instead of one token at a time. Some keep several candidate tokens per position and use a small selector to pick one consistent path through them, which raises the number of accepted tokens per step."""
DOC_B = """The downtown store receives three deliveries a week: Monday, Wednesday and Friday. Each delivery is checked against the purchase order before anything goes on the shelf. Staff count every box, scan every pair, and flag any size that is missing or extra. Damaged boxes are photographed and set aside for a return to the vendor within 48 hours.
New releases follow a separate path. Pairs for a release are tagged when they arrive and stay in the back room until release morning. On release day the store opens at 10 a.m., online sales open at the same time, and any pairs left after the first week move to the regular sales floor.
Inventory is reconciled every Sunday night. The system count and the shelf count are compared style by style, and any difference larger than two pairs triggers a recount on Monday morning before the first delivery. The reconciliation report goes to the buying team, who use it to plan reorders and markdowns. Last quarter, recounts dropped by a third after the store moved to scanning every pair at receiving instead of counting boxes."""
P += [("sum1", "summary", "Summarize the following text in exactly 3 bullet points, each one sentence.\n\n---\n" + DOC_A, {"bullets": 3}),
      ("sum2", "summary", "Summarize the following text in one paragraph of at most 80 words.\n\n---\n" + DOC_A, {"words": (20, 80), "paragraphs": 1}),
      ("sum3", "summary", "In exactly two sentences, summarize what the following text says about acceptance.\n\n---\n" + DOC_A, {"sentences": 2}),
      ("sum4", "summary", "Summarize the following store procedure in 3 bullets labeled Receiving, Releases, Reconciliation.\n\n---\n" + DOC_B, {"bullets": 3, "must": ["Receiving", "Releases", "Reconciliation"]}),
      ("sum5", "summary", "Turn the following into a single tweet of at most 280 characters. Output only the tweet.\n\n---\n" + DOC_B, {"chars": 280})]
P += [("count1", "counting", "List the numbers from 1 to 300 separated by commas. Output only the numbers, nothing else, no commentary.", {"seq": list(range(1, 301))}),
      ("count2", "counting", "Count down from 150 to 1, one number per line. Output only the numbers.", {"seq": list(range(150, 0, -1))}),
      ("count3", "counting", "List the even numbers from 2 to 400 separated by spaces. Output only the numbers.", {"seq": list(range(2, 401, 2))}),
      ("count4", "counting", "List the multiples of 7 from 7 to 700 separated by commas. Output only the numbers.", {"seq": list(range(7, 701, 7))}),
      ("count5", "counting", "List the numbers from 101 to 250 separated by commas. Output only the numbers, nothing else.", {"seq": list(range(101, 251))})]
CATS = ["coding", "math", "json", "html", "prose", "narrative", "summary", "counting"]
if a.types: CATS = [c for c in CATS if c in a.types.split(",")]

# ------------------------------------------------------------------ graders (bench_categories.py)
def words(t): return len(re.findall(r"\b[\w'’-]+\b", t))
def strip_fences(t):
    m = re.search(r"```(?:json|html|python)?\s*(.*?)```", t, re.S); return m.group(1) if m else t
def first_json(t):
    t = strip_fences(t).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i = t.find(opener); j = t.rfind(closer)
        if i != -1 and j > i:
            try: return json.loads(t[i:j + 1])
            except Exception: pass
    return None
def jeq(x, y):
    if isinstance(x, bool) or isinstance(y, bool): return x is y
    if isinstance(x, (int, float)) and isinstance(y, (int, float)): return abs(x - y) < 1e-6
    if isinstance(x, dict) and isinstance(y, dict): return x.keys() == y.keys() and all(jeq(x[k], y[k]) for k in x)
    if isinstance(x, list) and isinstance(y, list): return len(x) == len(y) and all(jeq(p, q) for p, q in zip(x, y))
    return isinstance(x, str) and isinstance(y, str) and x.strip() == y.strip()
class HP(HTMLParser):
    def __init__(s): super().__init__(); s.tags = {}; s.attrs = []; s.stack = []; s.bad = 0; s.text = []
    VOID = {"img", "input", "br", "hr", "meta", "link", "source"}
    def handle_starttag(s, tag, attrs):
        s.tags[tag] = s.tags.get(tag, 0) + 1; s.attrs.append((tag, dict(attrs)))
        if tag not in s.VOID: s.stack.append(tag)
    def handle_endtag(s, tag):
        if tag in s.VOID: return
        if s.stack and s.stack[-1] == tag: s.stack.pop()
        elif tag in s.stack:
            while s.stack and s.stack[-1] != tag: s.stack.pop(); s.bad += 1
            s.stack.pop()
        else: s.bad += 1
    def handle_data(s, d): s.text.append(d)
def grade_html(spec, out):
    h = HP(); raw = strip_fences(out); h.feed(raw); checks = []
    if spec.get("doctype"): checks.append(("doctype", "<!doctype html>" in raw.lower()))
    for t, n in spec.get("tags", {}).items(): checks.append((f"{t}x{n}", h.tags.get(t, 0) == n))
    for t, n in spec.get("min_tags", {}).items(): checks.append((f"{t}>={n}", h.tags.get(t, 0) >= n))
    ids = {at.get("id") for _, at in h.attrs}
    for i in spec.get("ids", []): checks.append((f"id={i}", i in ids))
    for tag, k, v in spec.get("attrs", []):
        checks.append((f"{tag}[{k}{'=' + v if v else ''}]", any(t == tag and k in at and (v is None or at.get(k) == v) for t, at in h.attrs)))
    txt = " ".join(h.text)
    for s in spec.get("text", []): checks.append((f"text:{s}", s in txt))
    checks.append(("balanced", h.bad == 0 and not h.stack or all(t in ("html", "body", "head") for t in h.stack)))
    return sum(ok for _, ok in checks) / len(checks), [c for c, ok in checks if not ok]
def grade_code(spec, out):
    code = strip_fences(out)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + "\n".join(f"try:\n    {t}\n    print('PASS')\nexcept Exception as e:\n    print('FAIL', type(e).__name__)" for t in spec["tests"].splitlines())); path = f.name
    try:
        r = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=10); lines = r.stdout.splitlines()
        passed = sum(l == "PASS" for l in lines); total = len(spec["tests"].splitlines())
        return passed / total, ([] if passed == total else [f"{passed}/{total} tests"])
    except subprocess.TimeoutExpired: return 0.0, ["timeout"]
    finally: os.unlink(path)
def grade(cat, spec, out):
    if cat == "coding": return grade_code(spec, out)
    if cat == "math":
        m = re.findall(r"ANSWER:\s*(.+)", out); got = (m[-1] if m else out.strip().splitlines()[-1] if out.strip() else "").strip().strip("*` .$\\")
        got = re.sub(r"\s*(km|kilometers?|years?( old)?|dollars?|\$)\s*$", "", got, flags=re.I).strip().strip("*` .$")
        got = re.sub(r"^\\boxed\{(.*)\}$", r"\1", got).replace("\\$", "").replace("$", "").strip()
        ok = got.replace(" ", "").lower() == spec["answer"].lower() or got in spec.get("alt", [])
        try: ok = ok or abs(float(got.replace(",", "")) - float(spec["answer"])) < 1e-6
        except ValueError: pass
        return (1.0 if ok else 0.0), ([] if ok else [f"got {got[:30]!r}"])
    if cat == "json":
        got = first_json(out); ok = got is not None and jeq(got, spec["json"]); return (1.0 if ok else 0.0), ([] if ok else ["json mismatch"])
    if cat == "html": return grade_html(spec["html"], out)
    if cat == "counting":
        nums = [int(x) for x in re.findall(r"\d+", out)]; ok = nums == spec["seq"]
        return (1.0 if ok else 0.0), ([] if ok else [f"sequence mismatch ({len(nums)} numbers)"])
    checks = []
    if "words" in spec: lo, hi = spec["words"]; w = words(out); checks.append((f"words {w} in {lo}-{hi}", lo <= w <= hi))
    if "chars" in spec: checks.append((f"chars {len(out.strip())}<={spec['chars']}", len(out.strip()) <= spec["chars"]))
    for m_ in spec.get("must", []): checks.append((f"has {m_}", m_.lower() in out.lower()))
    for b in spec.get("banned", []): checks.append((f"no {b}", b.lower() not in out.lower()))
    if "quotes" in spec: checks.append((f"dialogue>={spec['quotes']}", len(re.findall(r"[\"“][^\"”]{2,}[\"”]", out)) >= spec["quotes"]))
    if "paragraphs" in spec: checks.append((f"paragraphs={spec['paragraphs']}", len([p for p in re.split(r"\n\s*\n", out.strip()) if p.strip()]) == spec["paragraphs"]))
    if "bullets" in spec: checks.append((f"bullets={spec['bullets']}", len(re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+", out, re.M)) == spec["bullets"]))
    if "sentences" in spec: checks.append((f"sentences={spec['sentences']}", len(re.findall(r"[^.!?]+[.!?]", out.strip())) == spec["sentences"]))
    if not checks: return None, []
    return sum(ok for _, ok in checks) / len(checks), [c for c, ok in checks if not ok]

# ------------------------------------------------------------------ streaming call
def call(prompt, max_tokens):
    body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens,
                       "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": KW}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; t_content = None; txt = []; usage = {}; fin = None; rc = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            j = json.loads(data)
            if j.get("usage"): usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {}); c = d.get("content"); r_ = d.get("reasoning") or d.get("reasoning_content")
                if r_:
                    rc += len(r_)
                    if t_first is None: t_first = time.time()
                if c:
                    if t_first is None: t_first = time.time()
                    if t_content is None: t_content = time.time()
                    txt.append(c)
                if ch.get("finish_reason"): fin = ch["finish_reason"]
    t_end = time.time(); ct = usage.get("completion_tokens", 0)
    dec = (ct - 1) / (t_end - t_first) if (t_first and t_end > t_first and ct > 1) else None
    return {"ttft_s": round((t_first or t_end) - t0, 3), "ttft_content_s": round((t_content or t_end) - t0, 3), "wall_s": round(t_end - t0, 3), "completion_tokens": ct,
            "prompt_tokens": usage.get("prompt_tokens", 0), "decode_tok_s": round(dec, 1) if dec else None,
            "finish": fin, "reasoning_chars": rc, "output": "".join(txt)}

result = {"label": a.label, "model": a.model, "ts": time.strftime("%Y-%m-%d %H:%M"), "settings": {"temperature": 0, "chat_template_kwargs": KW, "max_tokens": a.max_tokens}}
parts = a.parts.split(",")
def check_foreign(m0, m1, sent, where):
    n = int(m1.get("request_success_total", 0) - m0.get("request_success_total", 0)) - sent
    if n: print(f"  !! {n} foreign request(s) during {where}", flush=True)
    return n

# warm-up (kernel JIT for new shapes)
for _ in range(2): call("Say hello in five words.", 16)
call(P[0][2], 64)

if "ttft" in parts:
    WORDS = ("speculative draft verify accept token block cache prefill decode latency stream batch kernel memory shard tensor "
             "graph layer expert router window anchor mask selector candidate softcap rank conv tap position budget").split()
    cnt = [0]
    def fresh(n_words=1500):
        cnt[0] += 1; r = random.Random(cnt[0] * 7919 + hash(a.label) % 1000)
        return f"Document {cnt[0]}-{r.randint(0, 10**9)}: " + " ".join(r.choice(WORDS) for _ in range(n_words)) + "\n\nReply with the single word OK."
    for _ in range(2): call(fresh(), 8)
    rows = []; m0 = metrics(); sent = 0
    for c in LEVELS:
        ttfts = []; pts = []
        for _ in range(3):
            ps = [fresh() for _ in range(c)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex: rs = list(ex.map(lambda p: call(p, 8), ps))
            sent += c; ttfts += [r["ttft_s"] for r in rs]; pts += [r["prompt_tokens"] for r in rs]
        row = {"c": c, "ttft_med_s": round(statistics.median(ttfts), 3), "ttft_max_s": round(max(ttfts), 3), "prompt_tokens": int(statistics.median(pts))}
        rows.append(row); print(f"  [{a.label}] TTFT c{c}: median {row['ttft_med_s']}s max {row['ttft_max_s']}s ({row['prompt_tokens']} tok)", flush=True)
    result["ttft"] = {"rows": rows, "foreign_requests": check_foreign(m0, metrics(), sent, "ttft")}

if "prefill" in parts:
    PARA = ("Inkling 2 Mini is a looped mixture-of-experts language model: a short prelude of layers, a core block that runs several "
            "passes over the same weights, and a short coda. Speculative decoding drafts several tokens ahead and lets the target "
            "verify them in one step, which is cheap on a memory-bound GPU. ")
    rows = []; m0 = metrics(); sent = 0
    for target in [int(s) for s in a.prefill_sizes.split(",")]:
        reps = max(1, int(target * 4.0 / len(PARA)))
        runs = []
        for i in range(2):
            p = f"Nonce {random.randint(0, 10**12)} run {i}.\nRead the following text and reply with the single word OK.\n\n" + PARA * reps
            r = call(p, 8); sent += 1; runs.append(r)
            print(f"  [{a.label}] prefill {r['prompt_tokens']:>6} tok: TTFT {r['ttft_s']:.2f}s -> {r['prompt_tokens'] / r['ttft_s']:.0f} tok/s", flush=True)
        best = min(runs, key=lambda r: r["ttft_s"])
        rows.append({"target": target, "prompt_tokens": best["prompt_tokens"], "ttft_s": best["ttft_s"], "prefill_tok_s": round(best["prompt_tokens"] / best["ttft_s"]),
                     "all_ttft_s": [r["ttft_s"] for r in runs]})
    result["prefill"] = {"rows": rows, "foreign_requests": check_foreign(m0, metrics(), sent, "prefill")}

if "types" in parts:
    by_cat = {c: [p for p in P if p[1] == c] for c in CATS}
    types = {}
    for cat in CATS:
        items = by_cat[cat]; entry = {"c1": {}, "levels": []}
        # c1: each prompt alone
        m0 = metrics(); recs = []
        for pid, _, prompt, spec in items:
            mi = metrics(); r = call(prompt, a.max_tokens); mo = metrics()
            sc, fails = grade(cat, spec, r["output"]); r.update({"id": pid, "score": sc, "fails": fails, "spec": spec_delta(mi, mo)})
            recs.append(r)
            print(f"  [{a.label}] c1 {pid:7} score={('%.2f' % sc) if sc is not None else ' n/a'} ttft {r['ttft_s']:.2f}s decode {r['decode_tok_s'] or 0:6.1f} tok/s {r['completion_tokens']:5} tok AL {r['spec'].get('acceptance_length', '-')}" + (f" fails: {'; '.join(fails)[:60]}" if fails else ""), flush=True)
        m1 = metrics(); sd = spec_delta(m0, m1)
        scored = [r["score"] for r in recs if r["score"] is not None]
        entry["c1"] = {"decode_med_tok_s": round(statistics.median(r["decode_tok_s"] for r in recs if r["decode_tok_s"]), 1),
                       "ttft_med_s": round(statistics.median(r["ttft_s"] for r in recs), 3), "tokens_med": int(statistics.median(r["completion_tokens"] for r in recs)),
                       "auto_score": round(sum(scored) / len(scored), 3) if scored else None, "spec": sd,
                       "foreign_requests": check_foreign(m0, m1, len(items), f"{cat} c1"), "items": recs}
        print(f"  [{a.label}] {cat:9} c1: decode {entry['c1']['decode_med_tok_s']} tok/s, AL {sd.get('acceptance_length')}, auto {entry['c1']['auto_score']}", flush=True)
        for c in [x for x in LEVELS if x > 1]:
            batch = [items[i % len(items)] for i in range(c)]
            m0 = metrics(); t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex: rs = list(ex.map(lambda it: call(it[2], a.max_tokens), batch))
            wall = time.time() - t0; m1 = metrics(); toks = sum(r["completion_tokens"] for r in rs)
            scores = [grade(cat, it[3], r["output"])[0] for it, r in zip(batch, rs)]
            scores = [s for s in scores if s is not None]
            row = {"c": c, "agg_tok_s": round(toks / wall, 1), "per_stream_decode_med_tok_s": round(statistics.median(r["decode_tok_s"] for r in rs if r["decode_tok_s"]), 1),
                   "ttft_med_s": round(statistics.median(r["ttft_s"] for r in rs), 3), "tokens": toks, "wall_s": round(wall, 2),
                   "auto_score": round(sum(scores) / len(scores), 3) if scores else None, "spec": spec_delta(m0, m1),
                   "foreign_requests": check_foreign(m0, m1, c, f"{cat} c{c}")}
            entry["levels"].append(row)
            print(f"  [{a.label}] {cat:9} c{c}: agg {row['agg_tok_s']:6.1f} tok/s, per-stream {row['per_stream_decode_med_tok_s']:6.1f}, AL {row['spec'].get('acceptance_length')}", flush=True)
        types[cat] = entry
    result["types"] = types

path = os.path.join(a.out, f"bench_{a.label}.json")
json.dump(result, open(path, "w"), indent=1)
print(f"[{a.label}] -> {path}")
