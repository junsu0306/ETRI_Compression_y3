"""The one image preprocessing used for MXQ calibration and for NPU evaluation.

Calibration tensors and NPU inputs must be produced by exactly the same code,
so both sides import this function. It depends on NumPy and OpenCV only and
runs on the NPU host (NumPy 2.x) as well as in the compile container (1.26).

Server-side FP32 training and evaluation use timm/PIL transforms instead
(crop_pct 0.9). The two are the same setting but are not pixel-identical; see
docs/METHODOLOGY.md.
"""

from __future__ import annotations

import cv2
import numpy as np

RESIZE_SHORT = 249
CROP = 224
MEAN = np.float32([0.5, 0.5, 0.5])
STD = np.float32([0.5, 0.5, 0.5])

PROFILE_NAME = "opencv_bicubic249_crop224"
PROFILE = {
    "profile": PROFILE_NAME,
    "color": "RGB",
    "resize_short": RESIZE_SHORT,
    "interpolation_name": "bicubic",
    "crop": CROP,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "layout": "HWC",
    "shape": [CROP, CROP, 3],
    "dtype": "float32",
}


def preprocess(image_path) -> np.ndarray:
    """Image file -> normalized RGB HWC float32 array of shape (224, 224, 3)."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot read image: %s" % image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    scale = float(RESIZE_SHORT) / min(height, width)
    image = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    height, width = image.shape[:2]
    top, left = (height - CROP) // 2, (width - CROP) // 2
    image = image[top:top + CROP, left:left + CROP]
    image = image.astype(np.float32) / 255.0
    return np.ascontiguousarray((image - MEAN) / STD, dtype=np.float32)
