"""Search nested static token masks on a structurally reduced ViT.

Three disjoint ImageNet training subsets serve scoring, block reconstruction,
and acceptance. Resume uses the checkpoint written after each searched layer.
scripts/run_pipeline.py generates this module's YAML from the experiment config.
"""

from __future__ import annotations

import os
import copy
import json
import argparse
import random
import numpy as np

import yaml
import torch

import timm
from patch_slimming.psvit.model_utils import get_shape, validate_vit, load_reduced_vit
from patch_slimming.psvit.data import build_subset_loader
from patch_slimming.psvit.search import run_patch_slimming_search
from patch_slimming.psvit.architecture import build_spec


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser("PS-ViT top-down mask search")
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    mcfg = cfg["model"]
    model_name = mcfg["name"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ── student(탐색 대상) + teacher(frozen 원본) ────────────────────────────────
    reduced_path = mcfg.get("reduced_checkpoint")
    reduced_meta = None
    if reduced_path:
        student, reduced_meta = load_reduced_vit(reduced_path, device=device)
        if student.num_classes != int(mcfg.get("num_classes", 1000)):
            raise ValueError("reduced checkpoint class count does not match config")
    else:
        student = timm.create_model(model_name, pretrained=True,
                                    num_classes=mcfg.get("num_classes", 1000)).to(device)
    validate_vit(student)
    shp = get_shape(student)
    print(f"[model] {model_name}  L={shp.num_blocks}  D={shp.embed_dim}  H={shp.num_heads}  "
          f"N={shp.num_global_tokens}  classes={shp.num_classes}")

    teacher = copy.deepcopy(student).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── independent score / reconstruction train / acceptance eval sets ────────
    score_cfg = cfg["score_data"]
    train_cfg = cfg["reconstruction_train"]
    eval_cfg = cfg["reconstruction_eval"]

    def build_subset(name, subset_cfg, excluded=(), shuffle=False):
        return build_subset_loader(
            model_name=model_name, data_path=cfg["data_path"],
            num_samples=subset_cfg["num_samples"], batch_size=subset_cfg["batch_size"],
            seed=subset_cfg.get("seed", 2022), split=subset_cfg.get("split", "train"),
            num_workers=subset_cfg.get("num_workers", 8),
            sample_id_file=os.path.join(out_dir, f"{name}_ids.txt"),
            excluded_indices=excluded, shuffle=shuffle,
        )

    score_loader, data_cfg, score_indices = build_subset("score", score_cfg)
    train_loader, _, train_indices = build_subset("reconstruction_train", train_cfg, score_indices, True)
    eval_loader, _, eval_indices = build_subset(
        "reconstruction_eval", eval_cfg, list(score_indices) + list(train_indices), False)
    assert set(score_indices).isdisjoint(train_indices)
    assert set(score_indices).isdisjoint(eval_indices)
    assert set(train_indices).isdisjoint(eval_indices)
    print(f"[data] score/train/eval={len(score_indices)}/{len(train_indices)}/{len(eval_indices)} "
          f"disjoint=True mean/std={data_cfg['mean']}/{data_cfg['std']}")

    # ── search config ──────────────────────────────────────────────────────────
    scfg = cfg["search"]
    search_cfg = {
        "epsilon": scfg["epsilon"],
        "error_metric": scfg["error_metric"],
        "search_step": scfg["search_step"],
        "max_search_iters": scfg.get("max_search_iters", 10_000),
        "score_max_batches": scfg.get("score_max_batches"),
        "block_finetune": cfg.get("block_finetune", {}),
    }

    keep_ids, accepted_errors, records = run_patch_slimming_search(
        student, teacher, score_loader, device, search_cfg,
        checkpoint_dir=out_dir, resume=not args.no_resume,
        reconstruction_train_loader=train_loader,
        reconstruction_eval_loader=eval_loader,
    )

    # ── 저장: architecture.json + searched student + records ────────────────────
    spec = build_spec(
        model_name, shp, keep_ids, accepted_errors,
        epsilon=scfg["epsilon"], error_metric=scfg["error_metric"],
        search_step=scfg["search_step"], score_mode=cfg.get("score", {}).get("mode", "paper_path_energy_dp"),
    )
    saved = {"model": student.state_dict(), "model_name": model_name,
             "source_reduced_checkpoint": reduced_path}
    if reduced_meta is not None:
        saved["mlp_dims"] = reduced_meta["mlp_dims"]
        saved["structural_compression_rate"] = reduced_meta.get("compression_rate")
    student_path = os.path.join(out_dir, "searched_student.pt")
    torch.save(saved, student_path + ".tmp")
    os.replace(student_path + ".tmp", student_path)
    with open(os.path.join(out_dir, "search_records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    spec.to_json(os.path.join(out_dir, "architecture.json"))  # last: marks the search as finished

    print(f"\n[done] architecture.json / searched_student.pt / search_records.jsonl → {out_dir}")
    print(f"       token schedule: {spec.token_schedule()}")


if __name__ == "__main__":
    main()
