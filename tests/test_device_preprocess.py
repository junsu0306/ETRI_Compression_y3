"""Preprocessing shared by calibration and NPU evaluation. Runs on the NPU host too."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from common import preprocess as shared

ROOT = Path(__file__).resolve().parents[1]


def reference(path) -> np.ndarray:
    """Independent statement of the contract: RGB, short side 249 bicubic, center crop 224, (x-0.5)/0.5."""
    image = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    scale = 249.0 / min(height, width)
    image = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))),
                       interpolation=cv2.INTER_CUBIC)
    height, width = image.shape[:2]
    top, left = (height - 224) // 2, (width - 224) // 2
    image = image[top:top + 224, left:left + 224].astype(np.float32) / 255.0
    return (image - np.float32(0.5)) / np.float32(0.5)


class PreprocessingTests(unittest.TestCase):
    def test_matches_the_contract_for_different_aspect_ratios(self):
        with tempfile.TemporaryDirectory() as tmp:
            for height, width in ((301, 507), (503, 299), (225, 225)):
                with self.subTest(height=height, width=width):
                    path = Path(tmp) / "test.png"
                    pixels = np.random.default_rng(7).integers(0, 256, (height, width, 3), dtype=np.uint8)
                    self.assertTrue(cv2.imwrite(str(path), pixels))
                    result = shared.preprocess(path)
                    np.testing.assert_array_equal(result, reference(path))
                    self.assertEqual(result.shape, (224, 224, 3))
                    self.assertEqual(result.dtype, np.float32)
                    self.assertTrue(result.flags.c_contiguous)

    def test_calibration_and_benchmark_use_the_same_function(self):
        from benchmark import eval_imagenet, onnx_reference, run_image
        from compile import calibration
        for module in (calibration, eval_imagenet, run_image, onnx_reference):
            self.assertIs(module.preprocess, shared.preprocess, module.__name__)

    def test_profile_describes_the_function(self):
        self.assertEqual(shared.PROFILE["resize_short"], 249)
        self.assertEqual(shared.PROFILE["interpolation_name"], "bicubic")
        self.assertEqual(shared.PROFILE["mean"], [0.5, 0.5, 0.5])
        self.assertEqual(shared.PROFILE["std"], [0.5, 0.5, 0.5])
        self.assertEqual(shared.PROFILE["shape"], [224, 224, 3])

    def test_unreadable_image_is_an_error(self):
        with self.assertRaises(ValueError):
            shared.preprocess("/nonexistent/image.jpg")

    def test_device_code_does_not_import_training_packages(self):
        code = ("import sys\n"
                "import benchmark.eval_imagenet, benchmark.profile_memory, benchmark.run_image\n"
                "import common.naming, common.preprocess, common.files\n"
                "bad = sorted(m for m in ('torch', 'timm', 'torchvision', 'yaml') if m in sys.modules)\n"
                "raise SystemExit('imported: %s' % bad if bad else 0)\n")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
