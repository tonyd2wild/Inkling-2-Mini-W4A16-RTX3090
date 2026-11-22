# Inkling 2 Mini (W4A16 INT4) on 2x RTX 3090: vLLM, NVLink, DSpark, 448K context

A tested recipe for serving [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on two RTX 3090s (Ampere, 24 GB each). It uses:
- the official vLLM image
- a small read-only overlay that ports the image's Inkling code to Ampere
- our INT4 weight-only checkpoint
- DSpark speculative decoding
- up to 458,752 tokens of context

- **Weights:** [Tech2wild/Inkling-2-Mini-W4A16](https://huggingface.co/Tech2wild/Inkling-2-Mini-W4A16) on Hugging Face (16.47 GB + 0.64 GB DSpark drafter).
- **Same model on one DGX Spark:** [Inkling-2-Mini-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark).

Unofficial community recipe and quantization. Not affiliated with Thinking Machines Lab, vLLM or NVIDIA.

Measured on 2026-11-20 and 2026-11-21, one request at a time:

| | **TP2 max** | TP2 vision | TP1 |
|---|---|---|---|
| Script | `launch_tp2_max.sh` | `launch_tp2.sh` | `launch_tp1.sh` |
| GPUs | 2x RTX 3090, NVLink bridge | 2x RTX 3090 | 1x RTX 3090 |
| Context per request | **458,752** | 131,072 | 8,192 default, 32,768 max |
| KV pool (BF16) | **483,850 tokens** | ~170K tokens | 33,392 tokens (at 32K) |
| Images | no (text-only) | yes | no |
| Speculative decoding | DSpark, k=7 | off (as measured) | off |
| Decode, one user | **137.5 tok/s** (100.5 with NVLink P2P off) | ~90 tok/s | ~74 tok/s |
| Long-context needles | PASS at 32K, 200K and 439K | not run | not run |
| 69-scenario tool eval: quality / responsiveness / deployability | **87.0 / 99.1 / 90.6** (55 pass / 10 partial / 4 fail) | not run | not run |
| Eval decode, median turn latency | 157.4 tok/s, 670 ms | | |
| 19 fixed prompts (math, reasoning, code, tool call, JSON) | | 19/19 | 18/19 |

For comparison, the original NVFP4 checkpoint on one DGX Spark scored 88.4 / 92.1 / 89.5 (56/10/3) on the same eval at 47.7 tok/s and a 2,035 ms median turn ([Spark repo](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark#10-69-scenario-tool-eval)).

**Which lane to use**

- **TP2 max**: the default. Long prompts, agents and tool use, text only. It works without an NVLink bridge too, at about 100 tok/s instead of 137.
- **TP2 vision**: when you need image input. 131K context, no speculative decoding as measured.
- **TP1**: one card only. The weights take 14.77 GiB of the 24 GB, so context is short.

## 1. The model and this checkpoint

**The model**
- **Inkling 2 Mini**, Thinking Machines Lab, released 2026-11-20 under Apache 2.0. Text and image in, text out.
- **Size:** 26.4B total parameters, 3.4B active per token.
  - It is a looped MoE: 24 layers, and layers 6 to 17 run 3 times with shared weights (48 layer applications per token).
  - Each layer routes to 8 of 128 experts.
  - Attention: 20 sliding-window layers (window 1024) plus 4 global layers, 24 query heads, 8 KV heads.
- **Context:** native maximum 1,048,576 tokens.
- **Original checkpoint:** routed experts are NVFP4 (layers 0 to 19) and MXFP8 (layers 20 to 23). Everything else is BF16/FP32. The `mtp.safetensors` file is the DSpark drafter: 3 BF16 layers, block size 8.

**This checkpoint** ([model card](https://huggingface.co/Tech2wild/Inkling-2-Mini-W4A16)), made from base revision `ed40e7da625f594fc9c4990f5f66be24873c22d7`:
- **What changed:** only the routed experts. They were re-quantized to **INT4, symmetric, one BF16 scale per 128 input channels**, stored in the compressed-tensors `pack-quantized` layout.
- **What did not change:** every other tensor, the tokenizer, the chat template and `mtp.safetensors` are byte-identical to the original.
- **Method:** round-to-nearest (RTN) with a per-group MSE search over the scale. See [`quantize/convert.py`](quantize/convert.py).
  - Each expert is first dequantized exactly from NVFP4/MXFP8 to fp32 (verified: the BF16 copy re-quantizes back to identical FP4 codes and FP8 bytes).
  - Then, for every row and group of 128, the scale is picked from `amax / d` for d in {7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 9}, whichever gives the lowest squared error.
  - Mean relative weight error: 10.4% on the NVFP4 layers and 10.3% on the MXFP8 layers. Plain `amax / 7.5` RTN gives 11.6% and 11.3%.
- **No GPTQ, AWQ or AutoRound.** Those tools need a transformers implementation of `inkling2_loop`, and none exists. The looped layers would also need custom activation capture. So this is a calibration-free quantization.
- **Size:** 16.47 GB of model shards plus the 0.64 GB drafter, against 19.53 GB plus 0.64 GB for the original. The INT4 weights take 8.04 GiB per GPU at TP2, against 9.47 GiB for the original on the same cards.
- **Why INT4:** the image's NVFP4/MXFP8 expert kernels need SM90+. INT4 weight-only runs on Ampere through vLLM's Humming kernels, and it is smaller.

## 2. Hardware and software (as tested)

| Item | Value |
|---|---|
| GPUs | 2x NVIDIA GeForce RTX 3090 24 GB, joined by an NVLink bridge (`nvidia-smi topo -m`: `NV2`, 2 links at 14.06 GB/s each) |
| Power limit | 240 W per card (stock is 350 W) |
| PCIe | both cards report x4 links on our board (see the NVLink note in section 9) |
| CPU / RAM | AMD Ryzen 9 5900X, 32 GB |
| OS | Ubuntu 24.04.4 LTS, kernel 6.8.0-147-generic |
| Driver | 580.178.04 |
| Docker | 29.5.2 with NVIDIA Container Toolkit 1.19.1 |
| Image | `vllm/vllm-openai:inkling-2-mini`, digest `sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e` (built 2026-11-20) |
| vLLM inside | `0.1.dev25855+g7cb5dae22`, Python 3.14, CUDA 13.0 |

The box has two more GPUs that were busy with other work. Only the two bridged cards were used.

## 3. Download

```bash
pip install -U "huggingface_hub[cli]"        # gives the `hf` command
hf download Tech2wild/Inkling-2-Mini-W4A16 --local-dir ~/models/Inkling-2-Mini-W4A16
docker pull vllm/vllm-openai:inkling-2-mini
# or the exact image we tested:
docker pull vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
```

The overlay replaces whole files from this image build. If the `inkling-2-mini` tag moves to a newer build, pin the digest: `IMAGE=vllm/vllm-openai@sha256:c2dc759f... ./launch_tp2_max.sh`.

## 4. Quick start

```bash
git clone https://github.com/tonyd2wild/Inkling-2-Mini-W4A16-RTX3090
cd Inkling-2-Mini-W4A16-RTX3090

./launch_tp2_max.sh                 # 448K, text-only, DSpark, NVLink   -> :8130
./launch_tp2.sh                     # 131K, images on
./launch_tp1.sh                     # one GPU, 8K   (MAXLEN=32768 ./launch_tp1.sh for 32K)
docker logs -f inkling-w4a16        # ready at "Application startup complete" (~2.5 min)

python3 bench/bench_decode.py http://localhost:8130/v1 inkling-2-mini-w4a16        # 6 prompts, decode tok/s + DSpark acceptance
python3 bench/needle.py http://localhost:8130/v1 inkling-2-mini-w4a16 200000       # 200K needle
./stop.sh
```

- Every launcher uses the same container name (`inkling-w4a16`) and port (`8130`), so starting one lane replaces the other.
- Overrides: `GPUS=`, `PORT=`, `NAME=`, `MODEL_DIR=`, `IMAGE=`, `MAXLEN=`, and `RESTART=unless-stopped` to keep it up across reboots. Extra vLLM flags go after the script name.
- The launchers refuse to start if the chosen GPUs already use more than 500 MiB. `FORCE=1` skips that check.
- Clone the repo into a path without spaces. The overlay mounts are built from the repo path.

Startup is about 2.5 minutes: weights load in about 45 s cold, then the warmup and CUDA graph capture.

## 5. The TP2 max command, every flag explained

What `launch_tp2_max.sh` runs (`$OVERLAY_MOUNTS` is the 8 read-only file mounts in `common.sh`):

```bash
docker run -d --name inkling-w4a16 --gpus '"device=0,1"' --ipc=host --restart no -p 8130:8000 \
  -e NCCL_P2P_LEVEL=NVL -e NCCL_DEBUG=INFO \
  -v ~/models/Inkling-2-Mini-W4A16:/models/model:ro $OVERLAY_MOUNTS vllm/vllm-openai:inkling-2-mini /models/model \
  --served-model-name inkling-2-mini-w4a16 inkling-2-mini \
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
  --tensor-parallel-size 2 --disable-custom-all-reduce --language-model-only \
  --max-model-len 458752 --max-num-batched-tokens 1024 --max-num-seqs 4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}'
```

| Flag | Why |
|---|---|
| `$OVERLAY_MOUNTS` | The Ampere port and the INT4 expert format (section 6). Without it the image refuses the GPU and the checkpoint. |
| `-e NCCL_P2P_LEVEL=NVL` | NCCL uses peer-to-peer only over NVLink. With the bridge the log shows `via P2P/CUMEM`. Without a bridge NCCL falls back to host memory, which is slower but works. |
| `-e NCCL_DEBUG=INFO` | Prints the NCCL transport in the log, so you can confirm NVLink is in use. |
| `--tensor-parallel-size 2` | Splits the model over both cards: 8.04 GiB of weights per GPU. |
| `--disable-custom-all-reduce` | vLLM's custom all-reduce has no SM86 kernel in this image. NCCL does the all-reduce instead. |
| `--language-model-only` | Skips the vision encoder. That frees about 0.53 GiB per GPU for KV cache. Use `launch_tp2.sh` for images. |
| `--max-model-len 458752` | The per-request context. The KV pool is 483,850 tokens (1.05x of this). |
| `--max-num-batched-tokens 1024` | The prefill chunk. Inkling's conv-state caches reserve blocks per request in proportion to the chunk, so 1024 instead of the default 2048 raises the context ceiling from about 369K to about 488K. Decode and prefill speed did not change (section 7). |
| `--max-num-seqs 4` | Up to 4 requests at once. Nothing was load-tested; all numbers here are one request at a time. |
| `--speculative-config {"method":"dspark","num_speculative_tokens":7}` | DSpark from `mtp.safetensors`. k must be 7 (block size 8 minus 1). It needs the overlay's Triton KV-write fix. |
| `--reasoning-parser inkling` | Puts thinking in `reasoning_content`, so `content` holds only the answer. |
| `--enable-auto-tool-choice --tool-call-parser inkling` | OpenAI-style `tools` / `tool_calls`. |
| no `--gpu-memory-utilization` | This vLLM build sizes the KV cache from the free memory measured after warmup. Setting 0.92 to 0.95 did not add KV here. |

The KV cache is BF16. FP8 KV is not possible on Ampere for this model: Inkling's attention allows FP8 KV only on SM12x, and Triton has no `fp8e4nv` type below SM89.

## 6. Why an overlay is needed

The image `vllm/vllm-openai:inkling-2-mini` does not run Inkling on Ampere as shipped. It refuses compute capability 8.6 at load time. With that gate lowered, it then fails in turn on 9 pieces of Hopper/Blackwell-only code:
- FlashAttention-3
- PDL `griddepcontrol`
- TMA descriptors
- shared-memory-hungry Triton configs
- four compiled CUDA ops built only for SM90+ or SM100
- the custom all-reduce

Its Inkling code also only knows NVFP4 and MXFP8 experts.

[`overlay/`](overlay/) fixes this with 8 files bind-mounted read-only over the image's `vllm/models/inkling2/` code:
- each overlay file is a full copy of the image's file plus small changes
- each change falls back to an existing vLLM path (Triton, torch or FA2) or adds the INT4 expert format
- [`overlay/README.md`](overlay/README.md) explains every change and why it is needed
- [`overlay/patches/`](overlay/patches/) has a diff of each file against the image original

## 7. Context and KV pool

- **Max lane:** a KV pool of **483,850 tokens** (9,504 blocks, 10.44 GiB per GPU) and `--max-model-len 458,752`. About 20.4 GB is used per GPU at idle.
- **Prefill chunk:** at 262,144 context with the default 2048-token prefill chunk, the pool was 320,653 tokens (1.22x).
  - Inkling keeps per-request conv-state caches whose block reservation grows with the prefill chunk.
  - Halving the chunk to 1024 freed enough blocks to raise the ceiling to about 488K.
  - Decode stayed at 136 to 137 tok/s. Prefill was not slower: a 32K prompt ran at 3,068 tok/s on the first request, and a 200K prompt at 3,367 tok/s.
- **Vision lane:** `--gpu-memory-utilization 0.92` with the vision encoder loaded leaves a KV pool of about 170K tokens, so 131,072 per request fits.
- **TP1:** 14.77 GiB of weights on one 24 GB card leaves a 33,392-token pool at `--gpu-memory-utilization 0.95`. That allows 32K at most. The script defaults to 8K for more headroom.

## 8. Using it

OpenAI-compatible at `http://localhost:8130/v1`. Model name `inkling-2-mini-w4a16` (or `inkling-2-mini`).

**Reasoning effort** goes in `chat_template_kwargs.reasoning_effort`.
- Named levels: none = 0.0, minimal = 0.1, low = 0.2, medium = 0.7, high = 0.9, xhigh = 0.99, max = 0.99. A number from 0.0 to 0.99 also works.
- The default is 0.9 (high). `none` still thinks briefly.
- With a low `max_tokens`, the model can spend all of it thinking and return an empty `content`. Give it a few hundred tokens or more.

```bash
curl -s http://localhost:8130/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini-w4a16",
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
  "max_tokens": 512,
  "chat_template_kwargs": {"reasoning_effort": "none"}
}' | python3 -c 'import sys,json; m=json.load(sys.stdin)["choices"][0]["message"]; print("THINK:", (m.get("reasoning_content") or "")[:200]); print("ANSWER:", m["content"])'
```

With the OpenAI Python client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8130/v1", api_key="none")
r = client.chat.completions.create(
    model="inkling-2-mini-w4a16",
    messages=[{"role": "user", "content": "Summarize the plot of Hamlet in two sentences."}],
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
)
print(r.choices[0].message.content)
```

**Tool calling:**

```bash
curl -s http://localhost:8130/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini-w4a16",
  "messages": [{"role": "user", "content": "What is the weather in Atlanta right now?"}],
  "tools": [{"type": "function", "function": {
     "name": "get_weather", "description": "Current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "tool_choice": "auto",
  "max_tokens": 512,
  "chat_template_kwargs": {"reasoning_effort": "none"}
}' | python3 -m json.tool
```

**Images** (vision lane only): standard OpenAI `image_url` content parts.

## 9. Speed and checks

**NVLink.** NCCL runs the all-reduce over the bridge (`docker logs inkling-w4a16 2>&1 | grep "via P2P"` shows `P2P/CUMEM`).
- Single-stream decode: 137.5 tok/s with it, 100.5 tok/s with `NCCL_P2P_DISABLE=1` (+37%).
- Stability: 9 boots and the full test set ran with no Xid errors and zero NVLink replay, recovery or CRC errors.
- Our two cards report x4 PCIe links, so the no-NVLink number may be better on a board with x16 slots.

**DSpark** (the checkpoint's own drafter, k=7):
- Single-stream decode went from 93.7 to 137.5 tok/s (**1.47x**).
- 19.8% of drafted tokens were accepted: 1.39 per draft, so **2.39 tokens per target step**. Position 0 was accepted 59% of the time.
- On tool-calling turns it averaged 3.31 tokens per step.
- No public DFlash, EAGLE or MTP head exists for Inkling. DSpark is the speculator it ships with.

**Decode bench** ([`bench/bench_decode.py`](bench/bench_decode.py): 6 fixed prompts, temperature 0, reasoning effort none, streaming):
- **136.6 tok/s aggregate**, median 129.7.
- It reads DSpark acceptance from the server's `/metrics`.

**Needles** ([`bench/needle.py`](bench/needle.py): a passphrase at 50% depth in synthetic log text, sized with the server's tokenizer):

| Prompt tokens | Result | Prefill | Decode after |
|---|---|---|---|
| ~32K | PASS | 3,068 tok/s | |
| 199,595 | PASS | 59.3 s (3,367 tok/s) | |
| 439,384 | **PASS** | 190.9 s (2,302 tok/s) | 119 tok/s |

**Quality against the original checkpoint**, both measured on the 3090s with greedy decoding:
- The original NVFP4/MXFP8 checkpoint ran through the same overlay with `native/quant_config.py`.
- Perplexity comes from [`bench/ppl.py`](bench/ppl.py): `prompt_logprobs` over three fixed passages. It is a small sanity check, not a benchmark.

| | original NVFP4/MXFP8, TP2 | **W4A16, TP2** | W4A16, TP1 |
|---|---|---|---|
| Perplexity, prose | 4.574 | 4.677 (+2.3%) | - |
| Perplexity, code | 1.153 | 1.154 (+0.1%) | - |
| Perplexity, math | 1.812 | 1.800 (-0.7%) | - |
| 19 fixed prompts | 19/19 | 19/19 | 18/19 (*) |
| Decode, no speculative decoding (median) | 87.2 tok/s | 89.8 tok/s | 73.5 tok/s |

The 19 prompts are:
- 12 math word problems
- 3 reasoning questions
- 2 coding tasks, run against unit tests
- 1 tool call
- 1 strict-JSON extraction

(*) On TP1, one riddle hit the 6,000-token answer cap. The same weights on TP2 answered it in 1,916 tokens.

**Warm-up.** The first few requests after a start can take about 1.7 s to the first token while Triton compiles kernels for new shapes. After that it is fast (median eval turn: 670 ms).

## 10. 69-scenario tool eval

Our 69-scenario tool-calling eval ([2Wild-Model-Eval](https://github.com/tonyd2wild/2Wild-Model-Eval)) ran against the TP2 max lane.
- It runs multi-turn tool loops with mock tool results and scores Pass/Partial/Fail as 2/1/0.
- Settings: temperature 0, `reasoning_effort: "none"`, tools through `--tool-call-parser inkling`, one request at a time. These are the same settings as the Spark runs.
- Scorecard: [`results/`](results/).

| | **W4A16, 2x 3090 TP2 NVLink, DSpark** | NVFP4, 1x DGX Spark, BF16 KV ([Spark repo](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark)) |
|---|---|---|
| Quality | **87.0** / 100 (120 of 138 points) | 88.4 / 100 (122 of 138) |
| Result | 55 pass, 10 partial, 4 fail | 56 pass, 10 partial, 3 fail |
| Responsiveness | **99.1** (median turn latency 670 ms) | 92.1 (2,035 ms) |
| Deployability | **90.6** | 89.5 |
| Decode | **157.4 tok/s** (142.7 effective) | 47.7 tok/s |
| Tokens for the run | 79,018 | 82,637 |

**68 of 69 scenarios scored the same as the original checkpoint.**
- The one change is TC-22 (Toolset Scale): asked for a translation with a large tool list offered, it answered directly instead of calling the translate tool.
- The Spark's FP8-KV run flipped the same scenario the same way.
- The per-category results match the Spark FP8-KV run exactly.

The remaining lost points are the model's own, shared with the original checkpoint:
- four "hallucinated tools" partials (no invented tool, but the limitation was not stated clearly)
- error-recovery misses
- refusal calibration (TC-58 not refused, two unclear refusals)
- one skipped tool call (TC-15)
- two ambiguity guesses
- one chain without the final comparison

## 11. Limits

- **Text-only in the max-context lane.** `--language-model-only` drops the vision encoder. Use `launch_tp2.sh` for images (131K).
- **BF16 KV only on Ampere.** FP8 KV is not available for this model below SM12x.
- **First requests are slow while kernels compile** (about 1.7 s TTFT for the first few requests after a start).
- **Needs the overlay.** Stock vLLM cannot load this checkpoint, and there is no transformers implementation of the model.
- **Calibration-free quantization.** Prose perplexity is about 2% higher than the original. Code and math were unchanged in our checks.
- **Single requests only.** No concurrency or load tests were run.
- DSpark was measured on the text-only lane. The vision and TP1 numbers above have speculative decoding off.
- One box, cards capped at 240 W. Stock-power cards should be the same or faster.

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `Minimum capability: 90. Current capability: 86` | The overlay is not mounted. Run from the repo with the launch scripts, and check `docker inspect inkling-w4a16` for the 8 mounts. |
| `no kernel image is available for execution on the device` | Same cause, or a different image build. Pin the tested digest (section 3). |
| A custom all-reduce error at startup | Keep `--disable-custom-all-reduce`. |
| Decode about 100 tok/s instead of about 137 on TP2 max | NCCL is not using NVLink. `nvidia-smi topo -m` should show `NV#` between the two cards, and the log should show `via P2P/CUMEM`. Do not set `NCCL_P2P_DISABLE=1`. |
| Startup fails because `max_model_len` is larger than the KV cache can hold | Something else is using GPU memory, or you raised `MAXLEN`. The pool is sized from free memory after warmup. Free the GPUs or lower `MAXLEN`. |
| `content` is empty but tokens were used | The model spent `max_tokens` thinking. Raise `max_tokens` or lower `reasoning_effort`. |
| `num_speculative_tokens` error | It must be 7 for this checkpoint. |
| `Inkling2 KV cache dtype must be one of ...` | FP8 KV is SM12x-only. Leave `--kv-cache-dtype` at the default (BF16). |
| Low DSpark acceptance in a benchmark | Use realistic text. Random-token prompts accept almost no drafts. |
| `busy: GPU0(...)` from a launcher | Something is on those GPUs. Free them, pick others with `GPUS=`, or set `FORCE=1`. |

## 13. Files

| File | What |
|---|---|
| `launch_tp2_max.sh` | TP2, NVLink, DSpark k=7, 458,752 context, text-only |
| `launch_tp2.sh` | TP2, vision on, 131,072 context |
| `launch_tp1.sh` | one GPU, text-only, 8K default (32K max) |
| `common.sh` | shared settings, the overlay mount list and the GPU preflight |
| `stop.sh` | stops and removes the container (any lane) |
| `overlay/` | the Ampere port + INT4 expert format (Apache 2.0, from vLLM), with `README.md`, `patches/` and vLLM's `LICENSE` |
| `bench/bench_decode.py` | 6-prompt single-stream decode bench with DSpark acceptance |
| `bench/needle.py` | long-context needle at any length |
| `bench/ppl.py` | perplexity of three fixed passages via `prompt_logprobs` |
| `quantize/convert.py` | how the checkpoint was made: `dequant`, `int4` and `verify` from the original checkpoint |
| `results/` | the 69-scenario scorecard behind section 10 |

To rebuild the weights yourself from the original checkpoint (CPU only, a few GB of RAM, needs `torch` and `safetensors`):

```bash
python3 quantize/convert.py int4 ~/models/Inkling-2-Mini ~/models/Inkling-2-Mini-W4A16
```

## Credits

- [Thinking Machines Lab](https://huggingface.co/thinkingmachines) for Inkling 2 Mini, its checkpoint and the DSpark drafter ([model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache 2.0, [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/)).
- The [vLLM](https://github.com/vllm-project/vllm) project for Inkling 2 support, the `vllm/vllm-openai:inkling-2-mini` image, the Humming expert kernels, the Triton KV-cache write used by the DSpark fix, and the [Inkling 2 Mini recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini).
- The INT4 layout follows the [compressed-tensors](https://github.com/vllm-project/compressed-tensors) `pack-quantized` convention.
- The 69-scenario tool eval ([2Wild-Model-Eval](https://github.com/tonyd2wild/2Wild-Model-Eval)) follows the Pass/Partial/Fail method of wolttam's `tool-eval-bench`.

## License

- The scripts and docs in this repository are MIT ([LICENSE](LICENSE)).
- The files in [`overlay/`](overlay/) are modified vLLM source files under Apache 2.0 ([overlay/LICENSE](overlay/LICENSE)).
- The weights are not in this repository. They are on [Hugging Face](https://huggingface.co/Tech2wild/Inkling-2-Mini-W4A16) under Apache 2.0, as a derivative of Inkling 2 Mini by Thinking Machines Lab. Use of the model is also subject to Thinking Machines' [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/).
