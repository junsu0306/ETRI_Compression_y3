"""Calibration generation and the MXQ compiler call. No SDK, GPU or NPU required."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from common.files import sha256_file
from common.preprocess import PROFILE, preprocess
from compile import calibration, compile_mxq


def write_images(root: Path, classes=4, per_class=3) -> None:
    rng = np.random.default_rng(3)
    for index in range(classes):
        folder = root / f"{index:05d}"
        folder.mkdir(parents=True)
        for item in range(per_class):
            pixels = rng.integers(0, 256, (240 + index, 260 + item, 3), dtype=np.uint8)
            assert cv2.imwrite(str(folder / f"image_{item}.png"), pixels)


class CalibrationGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / "train"
        write_images(self.images)

    def test_seeded_selection_is_reproducible_and_recorded(self):
        first = calibration.sample_one_per_class(self.images, 4, seed=2022)
        self.assertEqual(first, calibration.sample_one_per_class(self.images, 4, seed=2022))
        self.assertEqual([entry.split("/")[0] for entry in first], ["00000", "00001", "00002", "00003"])
        self.assertNotEqual({tuple(calibration.sample_one_per_class(self.images, 4, seed=s)) for s in range(8)},
                            {tuple(first)})
        output = calibration.generate(self.images, self.root / "calib", num_images=4, seed=2022)
        rows = [line.split("\t") for line in (output / calibration.IMAGE_TABLE).read_text().splitlines()]
        self.assertEqual([row[1] for row in rows], first)
        self.assertTrue(all(len(row[2]) == 64 for row in rows))
        for index, entry in enumerate(first):
            np.testing.assert_array_equal(np.load(output / f"{index:04d}.npy"), preprocess(self.images / entry))
        self.assertEqual(len(calibration.validate(output, expected_count=4)), 4)

    def test_explicit_list_is_used_in_order(self):
        listing = self.root / "list.txt"
        listing.write_text("# comment\n00002/image_1.png\n00000/image_2.png\n")
        output = calibration.generate(self.images, self.root / "calib", num_images=2, image_list=listing)
        np.testing.assert_array_equal(np.load(output / "0000.npy"), preprocess(self.images / "00002/image_1.png"))
        profile = json.loads((output / calibration.PROFILE_FILE).read_text())
        self.assertEqual(profile["selection"], {"image_list": str(listing)})

    def test_wrong_count_missing_image_and_nonempty_output_are_rejected(self):
        listing = self.root / "list.txt"
        listing.write_text("00000/image_0.png\n")
        with self.assertRaisesRegex(ValueError, "required"):
            calibration.generate(self.images, self.root / "a", num_images=2, image_list=listing)
        listing.write_text("00000/absent.png\n")
        with self.assertRaises(FileNotFoundError):
            calibration.generate(self.images, self.root / "b", num_images=1, image_list=listing)
        occupied = self.root / "c"
        occupied.mkdir()
        (occupied / "keep.txt").write_text("preserve")
        with self.assertRaises(FileExistsError):
            calibration.generate(self.images, occupied, num_images=4, seed=1)
        self.assertEqual((occupied / "keep.txt").read_text(), "preserve")


class CalibrationValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profile = {**PROFILE, "num_tensors": 1}
        self.write_profile()
        np.save(self.root / "0000.npy", np.zeros((224, 224, 3), np.float32))

    def write_profile(self):
        (self.root / calibration.PROFILE_FILE).write_text(json.dumps(self.profile))

    def test_valid(self):
        self.assertEqual(len(calibration.validate(self.root, expected_count=1)), 1)

    def test_missing_metadata(self):
        (self.root / calibration.PROFILE_FILE).unlink()
        with self.assertRaises(FileNotFoundError):
            calibration.validate(self.root, expected_count=1)

    def test_wrong_normalization(self):
        self.profile["mean"] = [0.485, 0.456, 0.406]
        self.write_profile()
        with self.assertRaisesRegex(ValueError, "mean"):
            calibration.validate(self.root, expected_count=1)

    def test_extra_tensor(self):
        np.save(self.root / "0001.npy", np.zeros((224, 224, 3), np.float32))
        with self.assertRaisesRegex(ValueError, "Expected 1"):
            calibration.validate(self.root, expected_count=1)

    def test_invalid_tensor(self):
        for value in (np.zeros((3, 224, 224), np.float32), np.zeros((224, 224, 3), np.float64),
                      np.full((224, 224, 3), np.nan, np.float32)):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                np.save(self.root / "0000.npy", value)
                with self.assertRaisesRegex(ValueError, "Invalid HWC"):
                    calibration.validate(self.root, expected_count=1)


class CompilerCallTests(unittest.TestCase):
    def test_compiler_receives_exactly_the_final_arguments(self):
        sdk = types.ModuleType("qbcompiler")
        calls = []
        sdk.mxq_compile = lambda **kwargs: calls.append(kwargs)
        with patch.dict(sys.modules, {"qbcompiler": sdk}):
            compile_mxq.compile_one("model.onnx", "calib.txt", "model.mxq")
        self.assertEqual(calls, [{
            "model": "model.onnx", "calib_data_path": "calib.txt", "save_path": "model.mxq",
            "backend": "onnx", "inference_scheme": "global8", "cpu_offload": True,
            "config_preset": "vision_transformer"}])

    def test_dry_run_lists_models_without_the_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "onnx").mkdir()
            (root / "onnx/tiny30__reduced__nhwc_b1_npusafe.onnx").write_bytes(b"test-only")
            (root / "onnx/unrelated.onnx").write_bytes(b"test-only")
            arguments = ["compile", "--onnx-dir", str(root / "onnx"), "--calib-dir", str(root),
                         "--output-dir", str(root / "mxq"), "--dry-run"]
            with patch.object(compile_mxq, "validate", return_value=[root / "0000.npy"]), \
                    patch.object(sys, "argv", arguments), patch.dict(sys.modules, {"qbcompiler": None}):
                compile_mxq.main()
            self.assertFalse((root / "mxq").exists())

    def test_mxq_without_record_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "onnx").mkdir()
            (root / "mxq").mkdir()
            (root / "onnx/tiny30__reduced__nhwc_b1_npusafe.onnx").write_bytes(b"test-only")
            (root / "mxq/tiny30__reduced__nhwc_b1_npusafe_vt_global8.mxq").write_bytes(b"partial")
            arguments = ["compile", "--onnx-dir", str(root / "onnx"), "--calib-dir", str(root),
                         "--output-dir", str(root / "mxq"), "--dry-run"]
            with patch.object(compile_mxq, "validate", return_value=[]), patch.object(sys, "argv", arguments):
                with self.assertRaises(FileExistsError):
                    compile_mxq.main()

    def test_mxq_of_a_different_onnx_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "onnx").mkdir()
            (root / "mxq").mkdir()
            onnx = root / "onnx/tiny30__reduced__nhwc_b1_npusafe.onnx"
            onnx.write_bytes(b"first export")
            mxq = root / "mxq/tiny30__reduced__nhwc_b1_npusafe_vt_global8.mxq"
            mxq.write_bytes(b"compiled")
            mxq.with_suffix(".compile.json").write_text(json.dumps({"onnx_sha256": sha256_file(onnx)}))
            arguments = ["compile", "--onnx-dir", str(root / "onnx"), "--calib-dir", str(root),
                         "--output-dir", str(root / "mxq"), "--dry-run"]
            with patch.object(compile_mxq, "validate", return_value=[]), patch.object(sys, "argv", arguments):
                compile_mxq.main()  # same ONNX: skipped
                onnx.write_bytes(b"second export")
                with self.assertRaisesRegex(RuntimeError, "different"):
                    compile_mxq.main()

    def test_record_is_complete_before_the_mxq_appears(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx = root / "tiny30__reduced__nhwc_b1_npusafe.onnx"
            onnx.write_bytes(b"model")
            (root / calibration.PROFILE_FILE).write_text("{}")
            mxq = root / "tiny30__reduced__nhwc_b1_npusafe_vt_global8.mxq"
            seen = {}

            def fake_compile(onnx_path, calib_txt, save_path, scheme):
                Path(save_path).write_bytes(b"compiled bytes")

            def fake_environment():
                seen["mxq_exists_when_recording"] = mxq.exists()
                return {"qbcompiler": "test"}

            args = types.SimpleNamespace(worker_onnx=str(onnx), worker_mxq=str(mxq), calib_dir=str(root),
                                         worker_calib_txt=str(root / "calib.txt"), scheme="global8")
            with patch.object(compile_mxq, "compile_one", fake_compile), \
                    patch.object(compile_mxq, "compiler_environment", fake_environment):
                compile_mxq.worker(args)
            record = json.loads(mxq.with_suffix(".compile.json").read_text())
            self.assertFalse(seen["mxq_exists_when_recording"])
            self.assertEqual(record["mxq_sha256"], sha256_file(mxq))
            self.assertEqual(record["onnx_sha256"], sha256_file(onnx))
            self.assertFalse(list(root.glob("*.partial.mxq")))


if __name__ == "__main__":
    unittest.main()
