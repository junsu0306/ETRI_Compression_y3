"""Build the un-fine-tuned target structure and measure its isolated CUDA peak.

The pretrained baseline is pruned to the target compression and physically
reduced with no training step. The FP32 inference peak of that structure is
the fixed memory budget M_max for the structural training stage.

Writes into --output-dir: baseline.pt, target_untrained.pt, the two memory
profiles and budget.json. The budget is immutable once written.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import timm
import torch

from common.files import sha256_file
from compression.vit_pruning import ViTPruner
from compression.vit_reducing import reduce_vit_model, get_reduced_config

ROOT = Path(__file__).resolve().parents[1]


def build_target(baseline, compression, max_sparsity):
    target = copy.deepcopy(baseline).eval()
    pruner = ViTPruner(target, compression, max_sparsity=max_sparsity)
    pruner.apply(target)
    reduce_vit_model(target)
    return target


def save_model(path, model, name):
    temporary = path.with_suffix(".tmp")
    torch.save({"model_name": name, "state_dict": model.state_dict(),
                "mlp_dims": get_reduced_config(model), "trained_steps": 0}, temporary)
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="timm model name")
    ap.add_argument("--target-compression", type=float, required=True)
    ap.add_argument("--max-sparsity", type=float, default=0.95)
    ap.add_argument("--method", default="baseline_target_peak_global_l2_kd_static_patch")
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--memory-batch-size", type=int, default=1)
    ap.add_argument("--memory-warmups", type=int, default=10)
    ap.add_argument("--memory-runs", type=int, default=50)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    folder = Path(args.output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "budget.json"
    if output.exists():
        raise FileExistsError(f"Budget is immutable: {output}")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    baseline_path = folder / "baseline.pt"
    target_path = folder / "target_untrained.pt"
    if baseline_path.exists():
        checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
        if checkpoint["model_name"] != args.model:
            raise ValueError("Baseline identity mismatch")
        baseline = timm.create_model(args.model, pretrained=False).eval()
        baseline.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        baseline = timm.create_model(args.model, pretrained=True).eval()
        save_model(baseline_path, baseline, args.model)
    target = build_target(baseline, args.target_compression, args.max_sparsity)
    save_model(target_path, target, args.model)
    profiles = {}
    for label, checkpoint in (("baseline", baseline_path), ("target", target_path)):
        profile = folder / f"{label}_profile.json"
        subprocess.run([sys.executable, "-m", "evaluation.profile_memory",
                        "--checkpoint", str(checkpoint), "--output", str(profile), "--details",
                        "--device", "cuda:0", "--batch-size", str(args.memory_batch_size),
                        "--warmups", str(args.memory_warmups),
                        "--runs", str(args.memory_runs)], cwd=ROOT, check=True)
        profiles[label] = json.loads(profile.read_text())
    result = {"method": args.method, "model": args.model,
              "target_compression": args.target_compression, "trained_steps": 0,
              "metric": "cuda_peak_allocated_bytes", "max_mib": profiles["target"]["peak_allocated_bytes"] / 2**20,
              "baseline_checkpoint": str(baseline_path), "baseline_sha256": sha256_file(baseline_path),
              "target_checkpoint": str(target_path), "target_sha256": sha256_file(target_path),
              "baseline_profile": profiles["baseline"], "target_profile": profiles["target"],
              "pruning": {"method": "global_l2", "max_sparsity": args.max_sparsity}}
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    print(f"{args.model} -{100 * args.target_compression:.0f}%: target parameters="
          f"{profiles['target']['parameters']:,}  M_max={result['max_mib']:.6f} MiB; "
          "no fine-tuning performed", flush=True)


if __name__ == "__main__":
    main()
