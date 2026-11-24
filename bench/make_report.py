#!/usr/bin/env python3
"""make_report.py BASE.json NEW.json OUT_PREFIX "Box title" ["footer"]

Charcoal-and-white chart (house style of glm53-exl3-repo/tools/make_category_report.py) plus a markdown table,
DSpark (gray) vs DFlash2 (white), from two bench_inkling.py result files:
  row 1: TTFT c1..c6 (streamed, fresh prompts) | prefill tok/s at 1K / 8K / 32K
  row 2: aggregate tok/s c1..c6 (mean over prompt types) | per-type decode tok/s at c1
  row 3: per-type acceptance length at c1 | DFlash2 / DSpark aggregate speedup per type x concurrency
Writes OUT_PREFIX.png and OUT_PREFIX.md.
"""
import json, statistics, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
OUT, TITLE = sys.argv[3], sys.argv[4]
FOOT = sys.argv[5] if len(sys.argv) > 5 else ""
NA, NB = "DSpark", "DFlash2"
CATS = [c for c in ["coding", "math", "json", "html", "prose", "narrative", "summary", "counting"] if c in A.get("types", {}) and c in B.get("types", {})]
LEVELS = sorted({r["c"] for r in A["ttft"]["rows"]}) if "ttft" in A else [1, 2, 3, 4, 5, 6]

def lvl(d, cat, c, key):
    e = d["types"][cat]
    if c == 1:
        return {"agg": e["c1"]["decode_med_tok_s"], "per": e["c1"]["decode_med_tok_s"], "al": e["c1"]["spec"].get("acceptance_length")}[key]
    for r in e["levels"]:
        if r["c"] == c:
            return {"agg": r["agg_tok_s"], "per": r["per_stream_decode_med_tok_s"], "al": r["spec"].get("acceptance_length")}[key]
    return None

def mean_agg(d, c):
    v = [lvl(d, cat, c, "agg") for cat in CATS]; v = [x for x in v if x]
    return statistics.mean(v) if v else None

G = "#1B1C1F"; P = "#232428"; INK = "#F2F2F0"; MUT = "#9A9DA3"; RULE = "#3A3C41"; WH = "#F2F2F0"; GY = "#8F9297"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": RULE, "axes.labelcolor": MUT,
                     "xtick.color": MUT, "ytick.color": MUT, "axes.facecolor": P, "figure.facecolor": G, "savefig.facecolor": G,
                     "font.size": 12})
fig, axs = plt.subplots(3, 2, figsize=(14, 17), dpi=140)

def style(ax, title):
    ax.set_title(title, fontsize=14, loc="left", fontweight="bold", color=INK, pad=10)
    ax.grid(axis="y", color=RULE, lw=0.6); ax.spines[["top", "right"]].set_visible(False)

def bars(ax, labels, va, vb, fmt, rot=0):
    x = list(range(len(labels))); w = 0.4
    ax.bar([i - w / 2 for i in x], [v or 0 for v in va], w, color=GY, label=NA)
    ax.bar([i + w / 2 for i in x], [v or 0 for v in vb], w, color=WH, label=NB)
    for i, (p, q) in enumerate(zip(va, vb)):
        if p: ax.text(i - w / 2, p, fmt(p), ha="center", va="bottom", fontsize=9.5, color=INK)
        if q: ax.text(i + w / 2, q, fmt(q), ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center", fontsize=11)
    ax.legend(frameon=False, fontsize=11, labelcolor=INK, loc="upper left")
    top = max([v or 0 for v in va + vb] + [1]); ax.set_ylim(0, top * 1.18)

# 1 TTFT
ax = axs[0][0]
ta = [r["ttft_med_s"] * 1000 for r in A["ttft"]["rows"]]; tb = [r["ttft_med_s"] * 1000 for r in B["ttft"]["rows"]]
bars(ax, [f"c{c}" for c in LEVELS], ta, tb, lambda v: f"{v:.0f}")
style(ax, f"TTFT, ms (median, {A['ttft']['rows'][0]['prompt_tokens']:,}-token fresh prompt)")
# 2 prefill
ax = axs[0][1]
pa = A["prefill"]["rows"]; pb = B["prefill"]["rows"]
bars(ax, [f"{r['prompt_tokens'] / 1000:.0f}K" for r in pa], [r["prefill_tok_s"] for r in pa], [r["prefill_tok_s"] for r in pb], lambda v: f"{v:,.0f}")
style(ax, "Prefill tok/s (cold prompt, c1)")
# 3 aggregate
ax = axs[1][0]
aa = [mean_agg(A, c) for c in LEVELS]; ab = [mean_agg(B, c) for c in LEVELS]
bars(ax, [f"c{c}" for c in LEVELS], aa, ab, lambda v: f"{v:.0f}")
style(ax, "Aggregate tok/s by concurrency (mean of 8 types)")
# 4 per-type decode c1
ax = axs[1][1]
da = [lvl(A, c, 1, "per") for c in CATS]; db = [lvl(B, c, 1, "per") for c in CATS]
bars(ax, CATS, da, db, lambda v: f"{v:.0f}", rot=30)
style(ax, "Decode tok/s per prompt type, one user (c1)")
# 5 acceptance
ax = axs[2][0]
la = [lvl(A, c, 1, "al") for c in CATS]; lb = [lvl(B, c, 1, "al") for c in CATS]
bars(ax, CATS, la, lb, lambda v: f"{v:.2f}", rot=30)
style(ax, "Acceptance length per type, c1 (tokens per target step)")
# 6 heatmap of speedup
ax = axs[2][1]
M = [[(lvl(B, cat, c, "agg") or 0) / (lvl(A, cat, c, "agg") or 1) for c in LEVELS] for cat in CATS]
ax.imshow(M, cmap="Greys_r", vmin=0.6, vmax=max(2.0, max(max(r) for r in M)), aspect="auto")
for i, row in enumerate(M):
    for j, v in enumerate(row):
        ax.text(j, i, f"{v:.2f}x", ha="center", va="center", fontsize=10.5, color=G if v > 1.25 else INK, fontweight="bold")
ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels([f"c{c}" for c in LEVELS], fontsize=11)
ax.set_yticks(range(len(CATS))); ax.set_yticklabels(CATS, fontsize=11)
ax.set_title("DFlash2 / DSpark speedup (aggregate tok/s)", fontsize=14, loc="left", fontweight="bold", color=INK, pad=10)
for s in ax.spines.values(): s.set_visible(False)

fig.suptitle(TITLE, fontsize=19, fontweight="bold", color=INK, x=0.02, ha="left", y=0.995)
fig.text(0.02, 0.968, "Thinking off, temperature 0, streaming. Same scripts, same session, nothing else on the endpoint.", fontsize=12, color=MUT)
if FOOT: fig.text(0.02, 0.006, FOOT, fontsize=10.5, color=MUT)
fig.tight_layout(rect=(0, 0.015, 1, 0.962), h_pad=2.6, w_pad=2.2)
fig.savefig(OUT + ".png"); print("->", OUT + ".png")

# markdown
def f(v, nd=1): return "-" if v is None else f"{v:.{nd}f}"
md = [f"# {TITLE}", "", "## TTFT (streamed, fresh prompts) and prefill", "", "| c | TTFT DSpark (s) | TTFT DFlash2 (s) |", "|---|---|---|"]
for ra, rb in zip(A["ttft"]["rows"], B["ttft"]["rows"]): md.append(f"| c{ra['c']} | {ra['ttft_med_s']:.3f} | {rb['ttft_med_s']:.3f} |")
md += ["", "| prompt tokens | prefill DSpark (tok/s) | prefill DFlash2 (tok/s) |", "|---|---|---|"]
for ra, rb in zip(pa, pb): md.append(f"| {ra['prompt_tokens']:,} | {ra['prefill_tok_s']:,} | {rb['prefill_tok_s']:,} |")
md += ["", "## Per prompt type: aggregate tok/s / per-stream decode tok/s / acceptance length", "",
       "| type | c | DSpark agg | DFlash2 agg | speedup | DSpark per-stream | DFlash2 per-stream | DSpark AL | DFlash2 AL |", "|---|---|---|---|---|---|---|---|---|"]
for cat in CATS:
    for c in LEVELS:
        a_, b_ = lvl(A, cat, c, "agg"), lvl(B, cat, c, "agg")
        md.append(f"| {cat} | c{c} | {f(a_)} | {f(b_)} | {f(b_ / a_ if a_ and b_ else None, 2)}x | {f(lvl(A, cat, c, 'per'))} | {f(lvl(B, cat, c, 'per'))} | {f(lvl(A, cat, c, 'al'), 2)} | {f(lvl(B, cat, c, 'al'), 2)} |")
md += ["", "## Mean aggregate tok/s over the 8 types", "", "| c | DSpark | DFlash2 | speedup |", "|---|---|---|---|"]
for c, x, y in zip(LEVELS, aa, ab): md.append(f"| c{c} | {f(x)} | {f(y)} | {f(y / x, 2)}x |")
md += ["", "## Quality (auto-graded, c1)", "", "| type | DSpark | DFlash2 | TTFT c1 DSpark (s) | TTFT c1 DFlash2 (s) | tokens (med) |", "|---|---|---|---|---|---|"]
for cat in CATS:
    ea, eb = A["types"][cat]["c1"], B["types"][cat]["c1"]
    md.append(f"| {cat} | {f(ea['auto_score'], 2)} | {f(eb['auto_score'], 2)} | {ea['ttft_med_s']:.3f} | {eb['ttft_med_s']:.3f} | {ea['tokens_med']} / {eb['tokens_med']} |")
ia = {it["id"]: it for cat in CATS for it in A["types"][cat]["c1"]["items"]}
ib = {it["id"]: it for cat in CATS for it in B["types"][cat]["c1"]["items"]}
same = sum(ia[k]["output"] == ib[k]["output"] for k in ia if k in ib)
md += ["", f"Identical c1 outputs (same text, temperature 0): {same}/{len(ia)}.", ""]
fa = A.get("ttft", {}).get("foreign_requests", 0) + A.get("prefill", {}).get("foreign_requests", 0) + sum(A["types"][c]["c1"]["foreign_requests"] + sum(r["foreign_requests"] for r in A["types"][c]["levels"]) for c in CATS)
fb = B.get("ttft", {}).get("foreign_requests", 0) + B.get("prefill", {}).get("foreign_requests", 0) + sum(B["types"][c]["c1"]["foreign_requests"] + sum(r["foreign_requests"] for r in B["types"][c]["levels"]) for c in CATS)
md[-1] = f"Foreign requests seen during the runs: DSpark {fa}, DFlash2 {fb}."
open(OUT + ".md", "w").write("\n".join(md) + "\n"); print("->", OUT + ".md")
