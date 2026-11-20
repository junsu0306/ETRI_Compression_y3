#!/usr/bin/env python3
"""Prepare ImageNet-1k (ILSVRC2012) in the folder layout this repository reads.

    <root>/train/<class>/<image>     1,281,167 images in the official release
    <root>/val/<class>/<image>       50,000 images, 50 per class

<class> is either the WordNet ID (n01440764 ...) or the class index (00000 ...).
Sorted folder order is the label order in both cases. Images are not part of
this repository; obtain them under the ImageNet terms of access.

Commands:

    from-hf      extract the parquet shards of the Hugging Face dataset ILSVRC/imagenet-1k
    val-folders  sort the flat official validation folder into class folders
    subset       link a small per-class subset for configs/smoke.yaml
    check        count classes and images of a prepared root
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from multiprocessing import Pool
import os
from pathlib import Path
import random
import re

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
OFFICIAL_COUNTS = {"train": 1281167, "val": 50000}
HF_NAME = re.compile(r"^(?P<stem>.+)_(?P<wnid>n\d{8})\.JPEG$")


def class_folders(split_dir: Path) -> list[Path]:
    return sorted(path for path in split_dir.iterdir() if path.is_dir())


def images_in(folder: Path) -> list[str]:
    return sorted(entry.name for entry in os.scandir(folder)
                  if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES)


# ── from-hf ───────────────────────────────────────────────────────────────────

def extract_shard(job) -> dict[int, set[str]]:
    """Write every image of one parquet shard to <output>/<split>/<wnid>/<original name>."""
    import pyarrow.parquet as pq

    shard, split_dir = job
    labels: dict[int, set[str]] = {}
    parquet = pq.ParquetFile(shard)
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=["image", "label"])
        for image, label in zip(table.column("image").to_pylist(), table.column("label").to_pylist()):
            match = HF_NAME.match(image["path"])
            if match is None:
                raise ValueError(f"Unexpected image name in {shard}: {image['path']}")
            wnid = match.group("wnid")
            labels.setdefault(int(label), set()).add(wnid)
            folder = split_dir / wnid
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / (match.group("stem") + ".JPEG")
            if not target.exists() or target.stat().st_size != len(image["bytes"]):
                temporary = folder / f".{target.name}.{os.getpid()}.tmp"
                temporary.write_bytes(image["bytes"])  # original file bytes, not re-encoded
                temporary.replace(target)
    return labels


def from_hf(args) -> None:
    source, output = Path(args.parquet_dir), Path(args.output)
    for split, prefix in (("train", "train-"), ("val", "validation-")):
        shards = sorted(source.glob(f"{prefix}*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No {prefix}*.parquet in {source}")
        expected = int(shards[0].stem.rsplit("-of-", 1)[1])
        if len(shards) != expected:
            raise FileNotFoundError(f"{split}: found {len(shards)} of {expected} parquet shards in {source}")
        labels: dict[int, set[str]] = {}
        with Pool(args.workers) as pool:
            jobs = [(shard, output / split) for shard in shards]
            for done, result in enumerate(pool.imap_unordered(extract_shard, jobs), 1):
                for label, wnids in result.items():
                    labels.setdefault(label, set()).update(wnids)
                if done % 20 == 0 or done == len(shards):
                    print(f"{split}: {done}/{len(shards)} shards", flush=True)
        # The dataset's integer label must equal the sorted-folder index used by ImageFolder.
        wnids = sorted({wnid for group in labels.values() for wnid in group})
        if any(labels.get(index) != {wnid} for index, wnid in enumerate(wnids)):
            raise ValueError(f"{split}: dataset labels do not follow sorted WordNet ID order")
    check_root(output, strict=True)


# ── val-folders ───────────────────────────────────────────────────────────────

def read_validation_labels(path: Path) -> dict[str, str]:
    """Image stem -> WordNet ID, from LOC_val_solution.csv or a 'name wnid' text file."""
    labels = {}
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                labels[Path(row["ImageId"]).stem] = row["PredictionString"].split()[0]
    else:
        for line in path.read_text().split("\n"):
            if line.strip():
                name, wnid = line.split()[:2]
                labels[Path(name).stem] = wnid
    return labels


def val_folders(args) -> None:
    source, output = Path(args.val_dir), Path(args.output)
    labels = read_validation_labels(Path(args.labels))
    images = [source / name for name in images_in(source)]
    missing = [image.name for image in images if image.stem not in labels]
    if missing:
        raise ValueError(f"{len(missing)} images have no label (first: {missing[0]})")
    for image in images:
        folder = output / labels[image.stem]
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / image.name
        if not target.exists():
            if args.move:
                image.replace(target)
            else:
                os.symlink(image.resolve(), target)
    print(f"Arranged {len(images)} validation images into {len(set(labels.values()))} class folders: {output}")


# ── subset ────────────────────────────────────────────────────────────────────

def subset(args) -> None:
    source, output = Path(args.source).resolve(), Path(args.output)
    for split, count in (("train", args.train_per_class), ("val", args.val_per_class)):
        total = 0
        for index, folder in enumerate(class_folders(source / split)):
            names = images_in(folder)
            chosen = random.Random(args.seed + index).sample(names, min(count, len(names)))
            target_folder = output / split / folder.name
            target_folder.mkdir(parents=True, exist_ok=True)
            for name in chosen:
                target = target_folder / name
                if not target.exists():
                    os.symlink(folder / name, target)
            total += len(chosen)
        print(f"{split}: linked {total} images")
    print(f"Subset root: {output}  (set evaluation.val_images to the val count above)")


# ── check ─────────────────────────────────────────────────────────────────────

def check_root(root: Path, strict: bool = False) -> bool:
    ok = True
    for split in ("train", "val"):
        counts = Counter({folder.name: len(images_in(folder)) for folder in class_folders(root / split)})
        total = sum(counts.values())
        low, high = (min(counts.values()), max(counts.values())) if counts else (0, 0)
        numeric = bool(counts) and all(name.isdigit() for name in counts)
        note = "" if total == OFFICIAL_COUNTS[split] else f"  (official release: {OFFICIAL_COUNTS[split]:,})"
        print(f"{split}: {len(counts)} classes ({'index' if numeric else 'WordNet ID'} folders), "
              f"{total:,} images, {low}-{high} per class{note}")
        ok &= len(counts) == 1000 and low > 0
        if split == "val":
            ok &= total == 50000 and low == high == 50
    train = [folder.name for folder in class_folders(root / "train")]
    val = [folder.name for folder in class_folders(root / "val")]
    if train != val:
        print("train and val class folders differ; labels would not match")
        ok = False
    print("OK" if ok else "NOT a complete ImageNet-1k layout")
    if strict and not ok:
        raise SystemExit(1)
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    hf = commands.add_parser("from-hf", help="extract Hugging Face parquet shards")
    hf.add_argument("--parquet-dir", required=True, help="folder with train-*.parquet and validation-*.parquet")
    hf.add_argument("--output", required=True)
    hf.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    hf.set_defaults(function=from_hf)

    val = commands.add_parser("val-folders", help="sort flat official validation images into class folders")
    val.add_argument("--val-dir", required=True, help="folder with ILSVRC2012_val_*.JPEG")
    val.add_argument("--labels", required=True,
                     help="LOC_val_solution.csv, or a text file with '<image name> <WordNet ID>' per line")
    val.add_argument("--output", required=True, help="e.g. <root>/val")
    val.add_argument("--move", action="store_true", help="move files instead of creating symbolic links")
    val.set_defaults(function=val_folders)

    sub = commands.add_parser("subset", help="link a small subset for the smoke configuration")
    sub.add_argument("--source", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--train-per-class", type=int, default=8)
    sub.add_argument("--val-per-class", type=int, default=2)
    sub.add_argument("--seed", type=int, default=2022)
    sub.set_defaults(function=subset)

    check = commands.add_parser("check", help="count classes and images")
    check.add_argument("--root", required=True)
    check.set_defaults(function=lambda a: check_root(Path(a.root), strict=True))

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
