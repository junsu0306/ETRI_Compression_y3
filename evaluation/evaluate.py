"""ImageNet validation accuracy, FP32 (AMP off), of one saved stage of a run.

Stages: baseline, pruned (zero-masked, dense shape), reduced, patch_slimming.
Images are loaded with the timm/PIL validation transform of the model
(bicubic, crop_pct 0.9, mean/std 0.5). This is the server-side reference; the
NPU benchmark uses the OpenCV preprocessing in common/preprocess.py.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import timm
import torch
from torchvision.datasets import ImageFolder

from common.files import write_json
from compression.engine import evaluate
from compression.vit_reducing import transfer_pruning_mask
from evaluation.profile_memory import load_candidate
from patch_slimming.psvit.architecture import ArchitectureSpec
from patch_slimming.psvit.compact import CompactPSViT
from patch_slimming.psvit.model_utils import apply_checkpoint_mlp_dims


def load_stage(kind, name, checkpoint, architecture=None, terminal_epochs=None):
    if kind in ("baseline", "reduced"):
        model, _ = load_candidate(checkpoint)
        return model
    if kind == "pruned":
        if terminal_epochs is not None and checkpoint["epoch"] + 1 != terminal_epochs:
            raise ValueError("Pruned evaluation requires the exact structural terminal epoch")
        raw = timm.create_model(name, pretrained=False)
        raw.load_state_dict(checkpoint["model"], strict=True)
        model = timm.create_model(name, pretrained=False)
        model.load_state_dict(checkpoint["model_ema"], strict=True)
        transfer_pruning_mask(raw, model)
        return model
    model = timm.create_model(name, pretrained=False)
    apply_checkpoint_mlp_dims(model, checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)
    spec = ArchitectureSpec.from_json(str(architecture))
    return CompactPSViT(model, spec.keep_ids_list(), select_mode="index_select")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["baseline", "pruned", "reduced", "patch_slimming"], required=True)
    ap.add_argument("--model", required=True, help="timm model name")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arch", help="architecture.json (patch_slimming only)")
    ap.add_argument("--terminal-epochs", type=int, help="required structural epoch count (pruned only)")
    ap.add_argument("--data-path", required=True, help="ImageNet root containing val/")
    ap.add_argument("--expected-images", type=int, default=50000)
    ap.add_argument("--expected-classes", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--label", default="", help="identifier stored in the result, e.g. tiny30")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.kind == "patch_slimming" and not args.arch:
        ap.error("--arch is required for patch_slimming")
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = load_stage(args.kind, args.model, checkpoint, args.arch, args.terminal_epochs)
    base = model.model if args.kind == "patch_slimming" else model
    data_config = timm.data.resolve_model_data_config(base)
    val_dir = Path(args.data_path) / "val"
    dataset = ImageFolder(str(val_dir), timm.data.create_transform(**data_config, is_training=False))
    if len(dataset) != args.expected_images or len(dataset.classes) != args.expected_classes:
        raise ValueError(f"Expected {args.expected_images} images in {args.expected_classes} classes, "
                         f"got {len(dataset)} in {len(dataset.classes)}: {val_dir}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, pin_memory=True)
    model = model.cuda().eval()
    start = time.perf_counter()
    metrics = evaluate(loader, model, torch.device("cuda:0"), amp=False)
    result = {"key": args.label, "kind": args.kind, "model": args.model,
              "source": str(args.checkpoint), "dataset": str(val_dir),
              "samples": len(dataset), "dtype": "float32", "amp": False,
              "batch_size": args.batch_size, "data_config": data_config,
              "parameters": sum(p.numel() for p in model.parameters()),
              "metrics": metrics, "seconds": time.perf_counter() - start}
    write_json(Path(args.output), result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
