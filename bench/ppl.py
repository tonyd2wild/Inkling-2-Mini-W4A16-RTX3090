"""Mean NLL / perplexity of fixed passages via prompt_logprobs (one request
per passage, sequential). usage: python3 ppl_inkling.py <label> <base_url> <out.json>"""

import json
import math
import sys
import urllib.request

PASSAGES = {
    "prose": (
        "The history of the bicycle is a story of many inventors working over a long period. "
        "Early designs in the nineteenth century had no pedals at all; riders pushed themselves along "
        "with their feet. Later, cranks and pedals were attached directly to the front wheel, which led "
        "to the high-wheeled machines of the 1870s. These were fast but dangerous, because a rider sat "
        "far above the ground and could be thrown forward by a small obstacle. The safety bicycle, with "
        "two wheels of similar size and a chain driving the rear wheel, solved this problem and quickly "
        "became popular. Pneumatic tires, introduced soon after, made riding far more comfortable. By the "
        "end of the century, cycling clubs had formed in many countries, and the bicycle had changed how "
        "ordinary people traveled, worked, and spent their free time. Women in particular gained a new "
        "kind of independence, and fashions changed to suit the new activity. Road surfaces improved as "
        "cyclists campaigned for better routes, which later benefited motorists as well."
    ),
    "code": (
        "def merge_sorted(a, b):\n"
        "    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n"
        "    result = []\n"
        "    i = j = 0\n"
        "    while i < len(a) and j < len(b):\n"
        "        if a[i] <= b[j]:\n"
        "            result.append(a[i])\n"
        "            i += 1\n"
        "        else:\n"
        "            result.append(b[j])\n"
        "            j += 1\n"
        "    result.extend(a[i:])\n"
        "    result.extend(b[j:])\n"
        "    return result\n\n\n"
        "def binary_search(items, target):\n"
        "    lo, hi = 0, len(items) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if items[mid] == target:\n"
        "            return mid\n"
        "        if items[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n\n\n"
        "class LRUCache:\n"
        "    def __init__(self, capacity):\n"
        "        self.capacity = capacity\n"
        "        self.data = {}\n\n"
        "    def get(self, key):\n"
        "        if key not in self.data:\n"
        "            return None\n"
        "        value = self.data.pop(key)\n"
        "        self.data[key] = value\n"
        "        return value\n\n"
        "    def put(self, key, value):\n"
        "        if key in self.data:\n"
        "            self.data.pop(key)\n"
        "        elif len(self.data) >= self.capacity:\n"
        "            self.data.pop(next(iter(self.data)))\n"
        "        self.data[key] = value\n"
    ),
    "math": (
        "Problem: A tank is filled by two pipes. The first pipe fills the tank in 6 hours and the second "
        "in 3 hours. How long does it take to fill the tank with both pipes open?\n"
        "Solution: In one hour the first pipe fills 1/6 of the tank and the second fills 1/3 of the tank. "
        "Together they fill 1/6 + 1/3 = 1/6 + 2/6 = 3/6 = 1/2 of the tank per hour. Therefore both pipes "
        "together fill the tank in 2 hours.\n\n"
        "Problem: The sum of three consecutive even integers is 78. What are the integers?\n"
        "Solution: Let the integers be n, n + 2 and n + 4. Then 3n + 6 = 78, so 3n = 72 and n = 24. "
        "The integers are 24, 26 and 28.\n\n"
        "Problem: A recipe uses 3 cups of flour for every 2 cups of sugar. How much sugar is needed "
        "with 12 cups of flour?\n"
        "Solution: The ratio of sugar to flour is 2 to 3. With 12 cups of flour, sugar = 12 * 2 / 3 = 8 "
        "cups. So 8 cups of sugar are needed."
    ),
}


def main():
    label, base, out = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3]
    res = {}
    for name, text in PASSAGES.items():
        body = {"model": "inkling-2-mini", "prompt": text, "max_tokens": 1,
                "temperature": 0, "prompt_logprobs": 0}
        req = urllib.request.Request(f"{base}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=600).read())
        plp = d["choices"][0].get("prompt_logprobs") or d.get("prompt_logprobs")
        lps = []
        for entry in plp[1:]:
            if not entry:
                continue
            # prompt_logprobs=0: vLLM returns only the actual prompt token per position
            vals = list(entry.values())
            lp = vals[0]["logprob"] if isinstance(vals[0], dict) else vals[0]
            lps.append(lp)
        nll = -sum(lps) / len(lps)
        res[name] = {"tokens": len(lps), "mean_nll": nll, "ppl": math.exp(nll), "logprobs": lps}
        print(f"[{label}] {name:6s} tokens={len(lps):4d} mean_nll={nll:.4f} ppl={math.exp(nll):.3f}", flush=True)
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main()
