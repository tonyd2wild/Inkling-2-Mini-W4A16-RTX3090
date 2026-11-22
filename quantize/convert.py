"""Inkling-2-Mini converter (2Wild, 2026-11-20).

  dequant  SRC OUT   NVFP4/MXFP8 experts -> BF16 checkpoint (sharded)
  int4     SRC OUT   experts -> INT4 W4A16 (symmetric, group G, compressed-
                     tensors pack-quantized), everything else unchanged
  verify   SRC BF16  round-trip check of the BF16 dequant

Streams one expert chunk at a time; peak RAM stays a few GB.
Only the routed experts are quantized in the source checkpoint; every other
tensor is copied byte-for-byte (same dtype) into both outputs.
"""

import argparse
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)
SHARD_BYTES = 5 * 1024**3
CHUNK_EXPERTS = 8
SCALE_SUFFIXES = (
    "_weight_scale",
    "_weight_amax",
    "_weight_outer_scale_denominator",
)
COPY_SKIP = {
    "model-00001-of-00001.safetensors",
    "model.safetensors.index.json",
    "hf_quant_config.json",
    "config.json",
    ".gitattributes",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def deq_nvfp4(packed, scale, amax, denom):
    lo = E2M1[(packed & 0xF).long()]
    hi = E2M1[(packed >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).flatten(-2)
    block = scale.float().repeat_interleave(16, dim=-1)
    return vals * block * (amax / denom).view(-1, 1, 1)


def deq_mxfp8(weight, scale):
    return weight.float() * torch.exp2(scale.float() - 127.0).repeat_interleave(
        32, dim=-1
    )


class Source:
    def __init__(self, src):
        self.src = src
        self.f = safe_open(f"{src}/model-00001-of-00001.safetensors", "pt", device="cpu")
        self.keys = list(self.f.keys())
        qc = json.load(open(f"{src}/hf_quant_config.json"))
        self.qcfg = qc
        self.nvfp4, self.mxfp8 = set(), set()
        for g in qc["model"]["parameter_groups"]:
            if g["compute_dtype"] == "nvfp4":
                self.nvfp4.update(g["tensors"])
            elif g["compute_dtype"] == "mxfp8":
                self.mxfp8.update(g["tensors"])
        self.experts = sorted(self.nvfp4 | self.mxfp8, key=_layer_key)

    def is_scale(self, key):
        base = None
        for suf in SCALE_SUFFIXES:
            if key.endswith(suf):
                base = key[: -len(suf)] + "_weight"
        return base is not None and base in (self.nvfp4 | self.mxfp8)

    def num_experts(self, name):
        return self.f.get_slice(name).get_shape()[0]

    def dequant_chunk(self, name, start, end):
        """fp32 [end-start, N, K] of expert weight `name`."""
        w = self.f.get_slice(name)[start:end]
        s = self.f.get_slice(name.replace("_weight", "_weight_scale"))[start:end]
        if name in self.nvfp4:
            amax = self.f.get_slice(name + "_amax")[start:end]
            den = self.f.get_tensor(name + "_outer_scale_denominator")
            return deq_nvfp4(w, s, amax, den)
        return deq_mxfp8(w, s)


def _layer_key(name):
    parts = name.split(".")
    return (int(parts[3]), parts[-1])


class ShardWriter:
    def __init__(self, out, total_hint):
        self.out, self.buf, self.size, self.files, self.wmap = out, {}, 0, [], {}

    def add(self, key, tensor):
        nbytes = tensor.numel() * tensor.element_size()
        if self.buf and self.size + nbytes > SHARD_BYTES:
            self.flush()
        self.buf[key] = tensor.contiguous()
        self.size += nbytes

    def flush(self):
        if not self.buf:
            return
        idx = len(self.files) + 1
        tmp = f"{self.out}/shard-{idx:05d}.tmp.safetensors"
        save_file(self.buf, tmp, metadata={"format": "pt"})
        self.files.append((tmp, list(self.buf)))
        log(f"wrote shard {idx} ({self.size / 1e9:.2f} GB, {len(self.buf)} tensors)")
        self.buf, self.size = {}, 0

    def finish(self):
        self.flush()
        n = len(self.files)
        total = 0
        for i, (tmp, keys) in enumerate(self.files, 1):
            final = f"model-{i:05d}-of-{n:05d}.safetensors"
            os.replace(tmp, f"{self.out}/{final}")
            total += os.path.getsize(f"{self.out}/{final}")
            for k in keys:
                self.wmap[k] = final
        index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(self.wmap.items()))}
        json.dump(index, open(f"{self.out}/model.safetensors.index.json", "w"), indent=2)
        return total


def copy_side_files(src, out):
    for name in os.listdir(src):
        if name in COPY_SKIP or name.startswith(".cache"):
            continue
        p = os.path.join(src, name)
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(out, name), dirs_exist_ok=True)
        else:
            shutil.copy2(p, os.path.join(out, name))


def cmd_dequant(a):
    src = Source(a.src)
    os.makedirs(a.out, exist_ok=True)
    wr = ShardWriter(a.out, None)
    expert_set = set(src.experts)
    for key in src.keys:
        if src.is_scale(key) or key in expert_set:
            continue
        wr.add(key, src.f.get_tensor(key))
    for name in src.experts:
        e = src.num_experts(name)
        parts = []
        for s in range(0, e, CHUNK_EXPERTS):
            parts.append(src.dequant_chunk(name, s, min(e, s + CHUNK_EXPERTS)).to(torch.bfloat16))
        wr.add(name, torch.cat(parts))
        log("dequant", name, "nvfp4" if name in src.nvfp4 else "mxfp8")
    total = wr.finish()
    copy_side_files(a.src, a.out)
    cfg = json.load(open(f"{a.src}/config.json"))
    cfg.pop("quantization_config", None)
    json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=2)
    log(f"BF16 checkpoint done: {total / 1e9:.2f} GB")


def quantize_int4(w, group, divisors):
    """Symmetric INT4 per (row, group) with an MSE search over the scale.
    w fp32 [C, N, K] -> (q int8 [C, N, K] in [-8, 7], scale bf16 [C, N, K/g],
    sse, sse of plain amax/7.5 RTN)."""
    c, n, k = w.shape
    wg = w.view(c, n, k // group, group)
    amax = wg.abs().amax(-1, keepdim=True).clamp_min(1e-20)
    best_err = None
    best_s = None
    base_err = None
    for d in divisors:
        s = (amax / d).to(torch.bfloat16).float()
        q = torch.round(wg / s).clamp_(-8, 7)
        err = (q.mul_(s).sub_(wg)).pow_(2).sum(-1, keepdim=True)
        if d == 7.5:
            base_err = err.sum().item()
        if best_err is None:
            best_err, best_s = err, s
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_s = torch.where(better, s, best_s)
    q = torch.round(wg / best_s).clamp_(-8, 7).to(torch.int8).view(c, n, k)
    return q, best_s.squeeze(-1).to(torch.bfloat16), best_err.sum().item(), base_err


def pack_int4(q):
    """compressed-tensors pack_to_int32: (q + 8) in 4-bit fields, element i
    of each 8 at bits 4i, along the last dim."""
    u = (q.to(torch.int64) + 8).view(*q.shape[:-1], q.shape[-1] // 8, 8)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    packed = (u << shifts).sum(-1)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)
    return packed.to(torch.int32)


def cmd_int4(a):
    src = Source(a.src)
    os.makedirs(a.out, exist_ok=True)
    wr = ShardWriter(a.out, None)
    expert_set = set(src.experts)
    divisors = [float(x) for x in a.divisors.split(",")]
    for key in src.keys:
        if src.is_scale(key) or key in expert_set:
            continue
        wr.add(key, src.f.get_tensor(key))
    stats = {}
    t0 = time.time()
    for name in src.experts:
        e = src.num_experts(name)
        packs, scales = [], []
        sse = base = ref = 0.0
        for s in range(0, e, CHUNK_EXPERTS):
            w = src.dequant_chunk(name, s, min(e, s + CHUNK_EXPERTS))
            q, sc, err, berr = quantize_int4(w, a.group, divisors)
            packs.append(pack_int4(q))
            scales.append(sc)
            sse += err
            base += berr if berr is not None else float("nan")
            ref += w.pow(2).sum().item()
        stem = name[: -len("_weight")]
        wr.add(stem + "_weight_packed", torch.cat(packs))
        wr.add(stem + "_weight_scale", torch.cat(scales))
        rel = (sse / ref) ** 0.5
        rel_base = (base / ref) ** 0.5
        stats[name] = {"rel_err": rel, "rel_err_rtn_7.5": rel_base,
                       "src": "nvfp4" if name in src.nvfp4 else "mxfp8"}
        log(f"int4 {name} rel_err={rel:.5f} (plain RTN amax/7.5: {rel_base:.5f}) "
            f"[{time.time() - t0:.0f}s]")
    total = wr.finish()
    copy_side_files(a.src, a.out)
    json.dump(stats, open(f"{a.out}/quantization_stats.json", "w"), indent=2)
    write_int4_configs(a, src)
    log(f"INT4 checkpoint done: {total / 1e9:.2f} GB")


def write_int4_configs(a, src):
    qc = json.loads(json.dumps(src.qcfg))
    groups = [g for g in qc["model"]["parameter_groups"]
              if g["compute_dtype"] not in ("nvfp4", "mxfp8")]
    groups.append({
        "compute_dtype": "w4a16",
        "output_dtype": "int32",
        "required_activation_quantization": None,
        "weight_block_shape": [1, a.group],
        "weight_transform": "int4_symmetric_mse_clip_rtn",
        "format": "pack-quantized",
        "num_bits": 4,
        "symmetric": True,
        "source_precision": {"nvfp4": sorted(src.nvfp4), "mxfp8": sorted(src.mxfp8)},
        "tensors": list(src.experts),
    })
    qc["model"]["parameter_groups"] = groups
    json.dump(qc, open(f"{a.out}/hf_quant_config.json", "w"), indent=2)
    cfg = json.load(open(f"{a.src}/config.json"))
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": [r"re:model\.llm\.layers\.\d+\.mlp\.experts\.w(13|2)_weight"],
                "weights": {
                    "num_bits": 4, "type": "int", "symmetric": True,
                    "strategy": "group", "group_size": a.group,
                    "dynamic": False, "actorder": None, "observer": "mse",
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
        "ignore": ["everything except the routed experts (BF16/F32 as in the original)"],
        "inkling2_note": "Per-tensor precisions are in hf_quant_config.json "
                         "(compute_dtype w4a16); vLLM loads them through the "
                         "inkling2 quant method with the 2Wild w4a16 overlay.",
    }
    json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=2)


def cmd_verify(a):
    """Re-quantize BF16 experts with the original scales; compare codes."""
    src = Source(a.src)
    idx = json.load(open(f"{a.bf16}/model.safetensors.index.json"))["weight_map"]
    lut = E2M1
    for name in src.experts[:: max(1, len(src.experts) // 6)]:
        fb = safe_open(f"{a.bf16}/{idx[name]}", "pt", device="cpu")
        e0, e1 = 0, 4
        wb = fb.get_slice(name)[e0:e1].float()
        if name in src.nvfp4:
            packed = src.f.get_slice(name)[e0:e1]
            s = src.f.get_slice(name.replace("_weight", "_weight_scale"))[e0:e1]
            amax = src.f.get_slice(name + "_amax")[e0:e1]
            den = src.f.get_tensor(name + "_outer_scale_denominator")
            unit = wb / (s.float().repeat_interleave(16, -1) * (amax / den).view(-1, 1, 1))
            codes = (unit[:1].unsqueeze(-1) - lut.view(1, 1, 1, -1)).abs().argmin(-1)
            packed = packed[:1]
            orig = torch.stack([packed & 0xF, packed >> 4], -1).flatten(-2).long()
            neg0 = orig == 8
            same = (codes == orig) | (neg0 & (codes == 0))
            print(f"{name} nvfp4 round-trip codes equal: {same.float().mean().item() * 100:.4f}% "
                  f"(-0 codes: {neg0.float().mean().item() * 100:.3f}%)")
        else:
            w8 = src.f.get_slice(name)[e0:e1]
            s = src.f.get_slice(name.replace("_weight", "_weight_scale"))[e0:e1]
            back = (wb / torch.exp2(s.float() - 127).repeat_interleave(32, -1)).to(torch.float8_e4m3fn)
            same = back.view(torch.uint8) == w8.view(torch.uint8)
            print(f"{name} mxfp8 round-trip bytes equal: {same.float().mean().item() * 100:.4f}%")
        amax_ok = None
        if name in src.nvfp4:
            amax_ok = (wb.abs().amax(dim=(1, 2)) / src.f.get_slice(name + "_amax")[e0:e1]).tolist()
            print(f"   max|W_bf16|/amax = {[round(x, 4) for x in amax_ok]}")


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "20")))
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dequant"); d.add_argument("src"); d.add_argument("out")
    q = sub.add_parser("int4"); q.add_argument("src"); q.add_argument("out")
    q.add_argument("--group", type=int, default=128)
    q.add_argument("--divisors", default="7.0,7.25,7.5,7.75,8.0,8.25,8.5,9.0")
    v = sub.add_parser("verify"); v.add_argument("src"); v.add_argument("bf16")
    a = p.parse_args()
    {"dequant": cmd_dequant, "int4": cmd_int4, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
