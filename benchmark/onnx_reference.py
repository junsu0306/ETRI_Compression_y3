#!/usr/bin/env python3
"""Run one image through an ONNX model on the CPU (FP32 reference for an MXQ).

Uses the same preprocessing as the NPU benchmark, so the Top-5 can be compared
with benchmark.run_image for the matching MXQ. Requires onnxruntime.

    python3 -m benchmark.onnx_reference --model MODEL.onnx --image IMAGE.jpg
"""

from __future__ import annotations

import argparse

import numpy as np

from benchmark.run_image import LABELS, load_labels, print_top5
from common.preprocess import preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="NHWC [1,224,224,3] ONNX file")
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", default=str(LABELS))
    args = parser.parse_args()

    import onnxruntime as ort

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    if list(input_meta.shape) != [1, 224, 224, 3]:
        parser.error(f"expected NHWC input [1,224,224,3], got {input_meta.shape}")
    tensor = preprocess(args.image)[None]
    logits = session.run(None, {input_meta.name: tensor})[0][0]
    print(f"model : {args.model}")
    print(f"image : {args.image}")
    print_top5(np.asarray(logits), load_labels(args.labels))


if __name__ == "__main__":
    main()
