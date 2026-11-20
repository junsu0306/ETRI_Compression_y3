#!/usr/bin/env python3
"""Export a baseline, structurally reduced, or Patch Slimming ViT to NHWC ONNX.

ONNX contract (docs/METHODOLOGY.md): input "input" float32 NHWC [1,224,224,3],
output "output" [1,1000], fixed batch 1, opset 17, normalization outside the
graph. With --npu-safe the graph must not contain dynamic selection operators
or identity token selections, and the export fails unless PyTorch and ONNX
Runtime agree to within 1e-4. A <name>.verified.json is written on success.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm

from common.files import write_json
from patch_slimming.psvit.architecture import ArchitectureSpec
from patch_slimming.psvit.compact import CompactPSViT
from patch_slimming.psvit.model_utils import apply_checkpoint_mlp_dims, load_reduced_vit

BANNED_OPS = {"Gather", "GatherElements", "GatherND", "TopK", "Equal", "Where", "Expand"}


class NHWCWrapper(nn.Module):
    def __init__(self, model, npu_safe=False):
        super().__init__(); self.model = model; self.npu_safe = npu_safe

    def forward(self, image):
        image = image.permute(0, 3, 1, 2)
        if not self.npu_safe:
            return self.model(image)
        # The supported Tiny/Small models use one CLS prefix and token pooling.
        # Fixed B=1 avoids dynamic CLS Expand/Where and the final Gather.
        x = self.model.patch_embed(image)
        x = torch.cat((self.model.cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.norm_pre(self.model.patch_drop(x))
        x = self.model.norm(self.model.blocks(x))
        x = x[:, :1, :].squeeze(1)
        return self.model.head(self.model.head_drop(self.model.fc_norm(x)))


def identity_selections(graph) -> list[str]:
    """Names of MatMul nodes whose constant left operand is an identity matrix."""
    from onnx import numpy_helper
    constants = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    for node in graph.graph.node:
        if node.op_type == "Constant":
            value = next((a for a in node.attribute if a.name == "value"), None)
            if value is not None:
                constants[node.output[0]] = numpy_helper.to_array(value.t)
    found = []
    for node in graph.graph.node:
        left = constants.get(node.input[0]) if node.op_type == "MatMul" else None
        if (left is not None and left.ndim == 2 and left.shape[0] == left.shape[1]
                and np.array_equal(left, np.eye(left.shape[0], dtype=left.dtype))):
            found.append(node.name)
    return found


def baseline_wrapper(model, args):
    npu_safe = getattr(args, "npu_safe", False)
    if npu_safe:
        if model.num_prefix_tokens != 1 or model.global_pool != "token" or model.no_embed_class:
            raise ValueError("NPU static wrapper requires a standard single-CLS ViT")
        for block in model.blocks:
            block.attn.fused_attn = False
    return NHWCWrapper(model.eval(), npu_safe=npu_safe).eval()


def load(args):
    if args.kind == "baseline":
        if not args.model: raise ValueError("--model is required for baseline")
        model = timm.create_model(args.model, pretrained=False).eval()
        checkpoint = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["model_name"] != args.model:
            raise ValueError("Baseline model mismatch")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return baseline_wrapper(model, args), {"kind": "baseline", "model": args.model}
    if args.kind == "reduced":
        if not args.reduced: raise ValueError("--reduced is required")
        model, meta = load_reduced_vit(args.reduced)
        return baseline_wrapper(model, args), {
            "kind": "reduced", "model": meta["model_name"],
            "compression_rate": meta.get("compression_rate"), "mlp_dims": meta["mlp_dims"]}
    if not args.arch or not args.weights:
        raise ValueError("--arch and --weights are required for patch")
    spec = ArchitectureSpec.from_json(args.arch); spec.validate()
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    base = timm.create_model(spec.model_name, pretrained=False,
                             num_classes=state["head.weight"].shape[0])
    apply_checkpoint_mlp_dims(base, checkpoint)
    base.load_state_dict(state, strict=True); base.eval()
    compact = CompactPSViT(base, spec.keep_ids_list(), select_mode="matmul",
                           nhwc_input=True, npu_static_batch_one=True).eval()
    compact.bake_for_export(torch.device("cpu"), torch.float32)
    return compact, {"kind": "patch", "model": spec.model_name,
                     "token_schedule": spec.token_schedule()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=["baseline", "reduced", "patch"])
    parser.add_argument("--model", default="")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--reduced", default="")
    parser.add_argument("--arch", default="")
    parser.add_argument("--weights", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--npu-safe", action="store_true",
                        help="Static B=1 CLS path for all variants; reject dynamic selection ops")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--verify-samples", type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1 or args.verify_samples < 1:
        parser.error("threads and verify-samples must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(2022)

    model, metadata = load(args)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    marker = output.with_suffix(".verified.json")
    if marker.exists():
        raise FileExistsError(f"Verified export already exists; use another output path: {marker}")
    dummy = torch.randn(1, 224, 224, 3)
    torch.onnx.export(model, dummy, output, opset_version=args.opset,
                      input_names=["input"], output_names=["output"],
                      do_constant_folding=True,
                      training=torch.onnx.TrainingMode.EVAL)

    import onnx
    graph = onnx.load(str(output)); onnx.checker.check_model(graph)
    ops = sorted({node.op_type for node in graph.graph.node})
    shape = [d.dim_value or d.dim_param for d in graph.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [d.dim_value or d.dim_param for d in graph.graph.output[0].type.tensor_type.shape.dim]
    print("metadata=%s" % metadata)
    print("input=%s nodes=%d size=%.2f MiB" % (shape, len(graph.graph.node), output.stat().st_size / 2**20))
    print("ops=%s" % ops)
    if args.npu_safe:
        banned = set(ops) & BANNED_OPS
        if banned:
            raise RuntimeError(f"NPU static graph contains unwanted operators: {sorted(banned)}")
        if shape != [1, 224, 224, 3] or output_shape != [1, 1000]:
            raise RuntimeError(f"Unexpected input/output shape: {shape} -> {output_shape}")
        redundant = identity_selections(graph)
        if redundant:
            raise RuntimeError(f"Identity token selections remain in the graph: {redundant}")
    differences = []
    eager_differences = []
    import onnxruntime as ort
    options = ort.SessionOptions(); options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    if isinstance(model, NHWCWrapper):
        eager = NHWCWrapper(model.model).eval()
    else:
        eager = CompactPSViT(model.model, [model._keep(i) for i in range(len(model.slim))],
                             select_mode="index_select", nhwc_input=True).eval()
    for _ in range(args.verify_samples):
        image = torch.randn_like(dummy)
        with torch.inference_mode():
            expected = model(image).numpy()
            original = eager(image).numpy()
        actual = session.run(["output"], {"input": image.numpy()})[0]
        differences.append(float(np.max(np.abs(expected - actual))))
        eager_differences.append(float(np.max(np.abs(expected - original))))
    print("PyTorch vs ORT max_abs_diff=%.6e; eager vs export wrapper=%.6e" % (
        max(differences), max(eager_differences)))
    if not all(np.isfinite(d) and d < 1e-4 for d in differences + eager_differences):
        raise RuntimeError("ONNX verification failed")
    metadata.update(onnx_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    input_shape=shape, ops=ops, npu_safe=args.npu_safe,
                    verify_samples=args.verify_samples, pytorch_ort_max_abs=max(differences),
                    eager_wrapper_max_abs=max(eager_differences),
                    output_shape=output_shape, opset=args.opset,
                    torch_version=torch.__version__, timm_version=timm.__version__,
                    note="FP32 graph parity only; MXQ compilation and NPU accuracy not verified")
    write_json(marker, metadata)


if __name__ == "__main__":
    main()
