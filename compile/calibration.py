#!/usr/bin/env python3
"""Turn a fixed set of ImageNet training images into calibration tensors for MXQ compilation.

Which images are used is decided in one of two ways:

    --image-list FILE   paths relative to --image-root, one per line
    --seed N            draw one image per class folder of --image-root (seeded)

Every image is preprocessed with common.preprocess, the same function the NPU
benchmark uses, and saved as an HWC float32 .npy. The output folder records
exactly which images were used:

    0000.npy ...              calibration tensors, in list order
    images.tsv                index, relative image path, image SHA-256
    preprocess_profile.json   preprocessing settings and source description
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from common.files import sha256_file, write_json
from common.preprocess import CROP, PROFILE, preprocess

PROFILE_FILE = "preprocess_profile.json"
IMAGE_TABLE = "images.tsv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_image_list(path) -> list[str]:
    lines = [line.strip() for line in Path(path).read_text().splitlines()]
    entries = [line for line in lines if line and not line.startswith("#")]
    if len(set(entries)) != len(entries):
        raise ValueError(f"Duplicate entries in calibration list: {path}")
    return entries


def sample_one_per_class(image_root, num_images, seed) -> list[str]:
    """Seeded draw of one image from each class folder (sorted folders, sorted files).

    With fewer images than classes, a seeded subset of the classes is used.
    The result depends only on the seed and on the folder and file names.
    """
    image_root = Path(image_root)
    classes = sorted(path for path in image_root.iterdir() if path.is_dir())
    if num_images > len(classes):
        raise ValueError(f"{num_images} images requested but only {len(classes)} class folders in {image_root}")
    chosen = set(random.Random(seed).sample(range(len(classes)), num_images))
    entries = []
    for index, folder in enumerate(classes):
        if index not in chosen:
            continue
        images = sorted(path.name for path in folder.iterdir()
                        if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"No image in {folder}")
        entries.append(f"{folder.name}/{random.Random(seed + index).choice(images)}")
    return entries


def generate(image_root, output_dir, num_images=1000, image_list=None, seed=None) -> Path:
    image_root, output_dir = Path(image_root), Path(output_dir)
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a new/empty output directory, not {output_dir}")
    if (image_list is None) == (seed is None):
        raise ValueError("Give exactly one of image_list and seed")
    entries = read_image_list(image_list) if image_list is not None \
        else sample_one_per_class(image_root, num_images, seed)
    if len(entries) != num_images:
        raise ValueError(f"Calibration list has {len(entries)} images; {num_images} required")
    missing = [entry for entry in entries if not (image_root / entry).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} listed images are missing under {image_root} (first: {missing[0]}). "
            "The list must match the ImageNet copy; use --seed to sample from this copy instead."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, entry in enumerate(entries):
        source = image_root / entry
        np.save(output_dir / f"{index:04d}.npy", preprocess(source))
        rows.append(f"{index:04d}\t{entry}\t{sha256_file(source)}")
    (output_dir / IMAGE_TABLE).write_text("\n".join(rows) + "\n")
    write_json(output_dir / PROFILE_FILE, {
        **PROFILE,
        "num_tensors": len(entries),
        "image_root": str(image_root.resolve()),
        "selection": {"image_list": str(image_list)} if image_list is not None
        else {"one_per_class_seed": seed},
        "image_table_sha256": sha256_file(output_dir / IMAGE_TABLE),
    })
    return output_dir


def validate(directory, expected_count=1000) -> list[Path]:
    """Refuse calibration tensors whose preprocessing or count is not the final setting."""
    directory = Path(directory)
    profile_path = directory / PROFILE_FILE
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"Missing {profile_path}; regenerate the tensors with compile/calibration.py. "
            "Do not reuse calibration tensors of unknown origin."
        )
    profile = json.loads(profile_path.read_text())
    for key, value in {**PROFILE, "num_tensors": expected_count}.items():
        if profile.get(key) != value:
            raise ValueError(f"Calibration metadata mismatch: {key} must be {value!r}")
    paths = sorted(directory.glob("*.npy"))
    if len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count} calibration tensors, got {len(paths)}")
    for path in paths:
        tensor = np.load(path, allow_pickle=False)
        if tensor.shape != (CROP, CROP, 3) or tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise ValueError(f"Invalid HWC float32 calibration tensor: {path}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-root", required=True, help="folder the list paths are relative to, e.g. <imagenet>/train")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--image-list", help="text file of relative image paths")
    choice.add_argument("--seed", type=int, help="draw one image per class folder with this seed")
    parser.add_argument("--output-dir", required=True, help="new or empty folder")
    parser.add_argument("--num-images", type=int, default=1000)
    args = parser.parse_args()
    if args.num_images < 1:
        parser.error("--num-images must be positive")
    output = generate(args.image_root, args.output_dir, args.num_images, args.image_list, args.seed)
    print(f"Saved {args.num_images} calibration tensors to {output}")


if __name__ == "__main__":
    main()
