# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tensor precision metadata of Inkling2 checkpoints (hf_quant_config.json).

Only the routed experts are quantized (NVFP4 or MXFP8); they are handled by
`Inkling2MoE`. Every other layer runs unquantized.

2Wild overlay (2026-11-20): adds a `w4a16` compute dtype, INT4 weights with
BF16 group scales in the compressed-tensors pack-quantized layout
(`<name>_weight_packed` int32 [E, N, K/8], `<name>_weight_scale` bf16
[E, N, K/group]), run by the Humming weight-only expert GEMMs. A checkpoint
whose quantized experts are all `w4a16` needs SM80+ instead of SM90+.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.models.inkling2.configs import INKLING2_ARCHITECTURES

_COMPUTE_DTYPES = ("float32", "bfloat16", "mxfp8", "nvfp4", "w4a16")


class Inkling2QuantConfig(QuantizationConfig):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.mxfp8_weights: set[str] = set()
        self.w4a16_weights: set[str] = set()
        self.w4a16_group_size = 128
        self.w4a16_symmetric = True
        has_fp_experts = False
        for group in config["model"]["parameter_groups"]:
            compute = group["compute_dtype"]
            if compute not in _COMPUTE_DTYPES:
                raise ValueError(f"Unsupported Inkling2 compute dtype: {compute}")
            if compute == "mxfp8":
                self.mxfp8_weights.update(group["tensors"])
            if compute in ("mxfp8", "nvfp4"):
                has_fp_experts = True
            if compute == "w4a16":
                self.w4a16_weights.update(group["tensors"])
                block = group.get("weight_block_shape") or [1, 128]
                self.w4a16_group_size = int(block[-1])
                self.w4a16_symmetric = bool(group.get("symmetric", True))
                if not self.w4a16_symmetric:
                    raise ValueError("Inkling2 w4a16 experts must be symmetric")
        # Instance attribute shadows the classmethod: an all-INT4 checkpoint
        # only needs the SM80 weight-only Humming kernels.
        min_capability = 90 if has_fp_experts or not self.w4a16_weights else 80
        self.get_min_capability = lambda: min_capability  # type: ignore[method-assign]

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "inkling2"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Inkling2QuantConfig:
        return cls(config)

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> QuantizationMethods | None:
        # Released checkpoints also declare a ModelOpt `quantization_config`
        # in config.json; their per-tensor precisions are in
        # hf_quant_config.json.
        architectures = getattr(hf_config, "architectures", None) or []
        if INKLING2_ARCHITECTURES & set(architectures):
            return "inkling2"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        return None

    def is_mxfp8_layer(self, layer_id: int) -> bool:
        name = f"model.llm.layers.{layer_id}.mlp.experts.w13_weight"
        return name in self.mxfp8_weights

    def is_w4a16_layer(self, layer_id: int) -> bool:
        name = f"model.llm.layers.{layer_id}.mlp.experts.w13_weight"
        return name in self.w4a16_weights
