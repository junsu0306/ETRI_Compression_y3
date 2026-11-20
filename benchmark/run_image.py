#!/usr/bin/env python3
"""Run one image through an MXQ model on the NPU: latency and Top-5.

A quick check that the runtime, the device and a model work before the full
benchmark. Run on the NPU host from the repository root:

    python3 -m benchmark.run_image --mxq MODEL.mxq --image IMAGE.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from benchmark.eval_imagenet import build_config, require_qbruntime
from common.naming import INFERENCE_SCHEME, INFERENCE_SCHEMES
from common.preprocess import preprocess

LABELS = Path(__file__).resolve().parent / "labels" / "imagenet_classes.txt"


def load_labels(path) -> list[str]:
    if path and Path(path).is_file():
        return [line.strip() for line in Path(path).read_text().splitlines()]
    return [str(index) for index in range(1000)]


def print_top5(logits: np.ndarray, labels: list[str]) -> None:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    for rank, index in enumerate(np.argsort(probabilities)[::-1][:5], 1):
        print(f"  {rank}. [{index:4d}] {labels[index]:<45s} {probabilities[index] * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mxq", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", default=str(LABELS))
    parser.add_argument("--core-mode", choices=INFERENCE_SCHEMES, default=INFERENCE_SCHEME,
                        help="must match the inference scheme the MXQ was compiled with")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0:
        parser.error("--runs must be positive and --warmups nonnegative")
    if not Path(args.mxq).is_file():
        parser.error(f"MXQ file not found: {args.mxq}")

    tensor = preprocess(args.image)
    require_qbruntime()
    import qbruntime

    accelerator = qbruntime.Accelerator(args.device_id)
    model = qbruntime.Model(args.mxq, build_config(args.core_mode))
    model.launch(accelerator)
    try:
        for _ in range(args.warmups):
            model.infer([tensor])
        started = time.perf_counter()
        for _ in range(args.runs):
            output = model.infer([tensor])
        seconds = time.perf_counter() - started
    finally:
        model.dispose()

    print(f"model : {args.mxq}")
    print(f"image : {args.image}")
    print(f"core  : {args.core_mode}, warmups={args.warmups}, runs={args.runs}")
    print(f"mean latency: {seconds / args.runs * 1000:.3f} ms ({args.runs / seconds:.1f} FPS)")
    print_top5(output[0], load_labels(args.labels))


if __name__ == "__main__":
    main()
