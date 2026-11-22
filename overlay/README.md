# Ampere overlay for `vllm/vllm-openai:inkling-2-mini`

The official Inkling 2 Mini image does not run on Ampere GPUs as shipped, and its Inkling code only accepts NVFP4 and MXFP8 experts. This overlay fixes both. It is 8 Python files that the launch scripts bind-mount **read-only** over the image's `vllm/models/inkling2/` code. The image and the host stay unchanged. Stop the container and the original code is back.

- Image: `vllm/vllm-openai:inkling-2-mini`, digest `sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e`, vLLM `0.1.dev25855+g7cb5dae22`, Python 3.14.
- In-container path of the code: `/usr/local/lib/python3.14/dist-packages/vllm/models/inkling2/`.
- Tested on RTX 3090 (SM86) only. Other Ampere and Ada cards (SM80, SM86, SM89) take the same code paths, but we have not run them.
- Every overlay file is a full copy of the image's file plus small changes. `patches/` has a unified diff of each one against the image original, so you can review exactly what changed.

> Only mount this overlay on Ampere or Ada. Almost every change is gated to GPUs below SM90 and does nothing on Hopper or Blackwell. The one exception is the vision encoder (`port/vision.py`), which picks FlashAttention-3 only on SM90, so a Blackwell SM100 card would get FA2 there instead of the upstream FA3.

## What fails without it

On an RTX 3090 the image fails in this order. Each fix lets it get one step further.

| # | Failure on SM86, as shipped | Fixed in |
|---|---|---|
| 0 | `ValueError: The quantization method inkling2 is not supported for the current GPU. Minimum capability: 90. Current capability: 86.` | `w4a16/quant_config.py` (or `native/quant_config.py`) |
| 1 | Vision encoder calls FlashAttention-3 (Hopper only): `Failed to initialize the TMA descriptor 801`, `flash_fwd_launch_template.h:200 invalid argument` | `port/vision.py` |
| 2 | Triton kernels use PDL `gdc_wait()`: `ptxas: Instruction 'griddepcontrol' requires .target sm_90` | `port/ops_init.py` |
| 3 | The large-prefill BF16 GEMM builds TMA tensor descriptors (SM90+) | `port/linear.py` |
| 4 | That Triton GEMM asks for ~144 KB of shared memory, the 3090 has ~99 KB per block: `OutOfResources` | `port/linear.py` |
| 5 | Compiled `inkling2_conv_mix` (`conv_mix.cu:978`): no kernel image for SM86 | `port/conv_mix.py` |
| 6 | Compiled `moe_align_block_size` (`_moe_C`) ships SM100 code only | `port/moe_sm90.py` |
| 7 | Compiled fused vocab-parallel embedding (used at TP > 1) ships SM100 code only | `port/ops_init.py` |
| 8 | vLLM's custom all-reduce kernel is not built for SM8x | `--disable-custom-all-reduce` flag (NCCL does the all-reduce) |
| 9 | DSpark: the draft layers' KV write (`reshape_and_cache_flash`) is compiled for SM90+ only. It fails with "no kernel image", reported asynchronously at the next op | `port/ops_init.py` |
| - | INT4 experts are not a known format | `w4a16/quant_config.py`, `w4a16/model.py`, `w4a16/moe.py`, `port/moe_sm90.py` |

## The files

| Overlay file | Mounted over | Diff |
|---|---|---|
| `port/ops_init.py` | `nvidia/ops/__init__.py` | `patches/nvidia_ops___init__.py.diff` |
| `port/vision.py` | `nvidia/vision.py` | `patches/nvidia_vision.py.diff` |
| `port/linear.py` | `nvidia/ops/linear.py` | `patches/nvidia_ops_linear.py.diff` |
| `port/conv_mix.py` | `nvidia/ops/conv_mix.py` | `patches/nvidia_ops_conv_mix.py.diff` |
| `port/moe_sm90.py` | `nvidia/ops/moe_sm90.py` | `patches/nvidia_ops_moe_sm90.py.diff` |
| `w4a16/quant_config.py` | `quant_config.py` | `patches/quant_config.py.w4a16.diff` |
| `w4a16/model.py` | `nvidia/model.py` | `patches/nvidia_model.py.diff` |
| `w4a16/moe.py` | `nvidia/moe.py` | `patches/nvidia_moe.py.diff` |
| `native/quant_config.py` (optional, original checkpoint only) | `quant_config.py` | `patches/quant_config.py.native.diff` |

### `port/ops_init.py` (Ampere shims, the DSpark fix)

It adds `_install_pre_sm90_shims()`, which runs when the module is imported and returns at once on SM90 and newer. Below SM90 it does four things:

1. **PDL off.** Triton's `gdc_wait()` and `gdc_launch_dependents()` become no-ops, and Triton's CUDA launcher is created with `launch_pdl=False`.
   - Why: the Inkling Triton kernels use programmatic dependent launch. That emits the `griddepcontrol` instruction, which ptxas rejects below sm_90.
   - PDL only lets the next kernel start launching early. Without it, kernels run one after another in normal stream order, so the math is unchanged.
2. **Unfused embedding.** `VocabParallelEmbedding` instances get `use_fused_embedding = False`.
   - Why: the image's fused vocab-parallel embedding kernel, used at TP > 1, is compiled for SM100 only.
   - vLLM's unfused torch path (mask, gather, fill) runs instead.
3. **Triton KV write (added 2026-11-21, this is what makes DSpark work).** The FLASH_ATTN backend's `reshape_and_cache_flash` is replaced with vLLM's own `triton_reshape_and_cache_flash`.
   - Why: the DSpark draft layers write their paged KV through `FlashAttentionImpl.do_kv_cache_update`, which calls `reshape_and_cache_flash`. This image compiles that op for SM90+ only, so DSpark died with "no kernel image" on SM86.
   - The Triton version takes the same arguments and writes the same cache layout.
   - The swap applies to the whole process below SM90: any layer that writes its KV through the FLASH_ATTN backend uses the Triton version.
   - Result: DSpark k=7 runs on the 3090s and lifts single-stream decode from 93.7 to 137.5 tok/s (1.47x).
4. Each shim sets a flag so it is installed once, even if the module is imported more than once.

### `port/vision.py` (vision encoder attention)

- One line changes: the FlashAttention version is `3 if SM90 else 2` instead of upstream's `2 if SM12x else 3`.
- Why: FA3 is Hopper-only. On SM86 it fails with the TMA descriptor error in row 1 of the table above.
- Only used when vision is on (`launch_tp2.sh`). The max lane runs `--language-model-only`, so the vision encoder is not loaded.

### `port/linear.py` (BF16 GEMM for large prefills)

- `has_tma = has_device_capability(90)`. The large-batch path (`tokens >= TMA_MIN_TOKENS`) builds Triton `TensorDescriptor`s (TMA). That path is now skipped without TMA, and the regular pointer-based Triton kernel runs instead.
- `num_stages` is capped at 2 without TMA, the same cap upstream already uses on SM12x.
- Why the cap: the default config needs about 144 KB of shared memory, the 3090 has about 99 KB per block, and Triton raises `OutOfResources`.

### `port/conv_mix.py` (short conv + AttnRes mix)

- SM8x now takes the same branch as SM12x: the Triton `short_conv` + `attn_res` path that upstream already uses where the compiled kernel is missing.
- Why: the compiled `inkling2_conv_mix` CUDA op has no SM86 code ("no kernel image").

### `port/moe_sm90.py` (expert routing + Humming expert GEMMs)

Two changes:

1. **`moe_align_block_size` in torch** (`_torch_moe_align_block_size`), used below SM90 only.
   - Why: the image's compiled `_moe_C` op ships SM100 code only.
   - It gives the same outputs as the CUDA op for the case this model uses (`expert_map=None`, `ignore_invalid_experts=True`).
   - It has fixed output sizes and no host syncs, so it can be captured in CUDA graphs.
2. **`HummingExperts(source_schema=...)`**: an optional argument that lets a caller pass its own Humming source weight schema with named tensors.
   - The INT4 experts use it to pass a compressed-tensors INT4 schema in place of the built-in ModelOpt NVFP4/MXFP8 schemas.
   - Without the argument, nothing changes.

### `w4a16/quant_config.py` (the `w4a16` format and its capability gate)

- `hf_quant_config.json` may now list parameter groups with `compute_dtype: "w4a16"`.
  - The group size comes from `weight_block_shape` (`[1, 128]` here).
  - The weights must be symmetric.
- New `is_w4a16_layer(layer_id)`.
- Minimum compute capability: **80** when every quantized expert is `w4a16`. It stays **90** for any checkpoint with NVFP4 or MXFP8 experts, so the original checkpoint keeps the upstream gate.

### `w4a16/model.py` (decoder layer)

- A layer whose experts are listed as `w4a16` builds its `Inkling2MoE` with `expert_format="w4a16"` and the group size from the quant config. Other layers are unchanged.

### `w4a16/moe.py` (INT4 experts)

- `Inkling2Experts` gains the `w4a16` format.
- **Load:** it registers `w13_weight_packed` / `w2_weight_packed` (int32 `[E, N, K/8]`) and `w13_weight_scale` / `w2_weight_scale` (bf16 `[E, N, K/group]`), the compressed-tensors `pack-quantized` layout in the checkpoint.
- **After load:** it repacks them into Humming's INT4 weight-only expert GEMMs through `CompressedTensorsWeightSchema(format="pack-quantized", type="int", num_bits=4, strategy="group", symmetric=True)`, then frees the checkpoint copies.
- The SM100 and SM120 native expert paths are turned off for this format. Humming's weight-only GEMMs run on SM80+.
- Checked on an RTX 3090: Humming's INT4 GEMM matches a plain torch GEMM on the unpacked INT4 weights to 0.2% relative (layers 0, 11, 20, 23).

### `native/quant_config.py` (optional)

- The original file with one change: minimum capability 90 becomes 80.
- Use it to serve the **original** NVFP4/MXFP8 checkpoint on Ampere. Mount the five `port/` files plus this file in place of the three `w4a16/` files.
- That is how the "native" reference numbers in the main README were measured: TP2, 9.47 GiB of weights per GPU, a 45K-token KV pool at the default 0.90 memory use. TP1 does not fit usefully (about 1.5K context).
- Checks on the 3090:
  - Humming's NVFP4 and MXFP8 expert GEMMs match torch GEMMs on the dequantized weights to 0.17% relative.
  - On the first test prompt, the native checkpoint gave the same answer and the same 116 completion tokens as a DGX Spark (GB10) running the stock image, with one word different.

## Using the patches instead of mounts

The diffs use paths relative to `site-packages`, so inside a container or a matching vLLM install:

```bash
cd /usr/local/lib/python3.14/dist-packages
for p in /path/to/overlay/patches/nvidia_*.diff /path/to/overlay/patches/quant_config.py.w4a16.diff; do
  patch -p1 < "$p"
done
```

Apply only one of the two `quant_config.py` diffs: `w4a16` for the INT4 checkpoint, or `native` for the original. The diffs are against this exact image build. Other vLLM versions will likely need the changes ported by hand.

## License

Every file in this directory is a modified copy of a vLLM source file. vLLM is Apache License 2.0, "Copyright contributors to the vLLM project".

- The SPDX headers are kept.
- Every change is marked inline with a "2Wild overlay" or "2Wild Ampere overlay" comment.
- vLLM's license is included verbatim in [`LICENSE`](LICENSE).
- The MIT license at the root of this repository covers the scripts and docs only, not these files.
