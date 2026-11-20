"""File-name contract shared by export, MXQ compilation and the NPU benchmark.

Standard library only: this module is imported on the NPU host, where the
training packages are not installed.
"""

from __future__ import annotations

GROUPS = ("tiny30", "tiny50", "small30", "small50")
VARIANTS = ("baseline", "reduced", "patch_slimming")
MODEL_IDS = tuple(f"{group}__{variant}" for group in GROUPS for variant in VARIANTS)

ONNX_TAG = "__nhwc_b1_npusafe"
INFERENCE_SCHEME = "global8"
INFERENCE_SCHEMES = ("single", "multi", "global", "global4", "global8")


def onnx_stem(group: str, variant: str) -> str:
    return f"{group}__{variant}{ONNX_TAG}"


def onnx_name(group: str, variant: str) -> str:
    return onnx_stem(group, variant) + ".onnx"


def mxq_name(onnx_stem_: str, scheme: str = INFERENCE_SCHEME) -> str:
    return f"{onnx_stem_}_vt_{scheme}.mxq"


def mxq_to_onnx_stem(mxq_stem: str) -> str:
    return mxq_stem.rsplit("_vt_", 1)[0] if "_vt_" in mxq_stem else mxq_stem
