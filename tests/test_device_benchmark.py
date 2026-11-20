"""NPU benchmark logic that does not need the device: sampling, naming, report rendering."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmark import eval_imagenet as evaluation
from benchmark import profile_memory
from common import naming

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference/final_20261006"
REPORT = REFERENCE / "npu_benchmark"


def make_validation(root: Path, names, images=3) -> None:
    for name in names:
        (root / name).mkdir(parents=True)
        for index in range(images):
            (root / name / f"img_{index}.JPEG").write_bytes(b"x")


class NamingTests(unittest.TestCase):
    def test_twelve_model_ids_and_file_names(self):
        self.assertEqual(len(naming.MODEL_IDS), 12)
        self.assertEqual(naming.onnx_name("tiny30", "reduced"), "tiny30__reduced__nhwc_b1_npusafe.onnx")
        mxq = naming.mxq_name(naming.onnx_stem("tiny30", "reduced"))
        self.assertEqual(mxq, "tiny30__reduced__nhwc_b1_npusafe_vt_global8.mxq")
        self.assertEqual(naming.mxq_to_onnx_stem(Path(mxq).stem), "tiny30__reduced__nhwc_b1_npusafe")
        self.assertEqual(evaluation.model_metadata(Path(mxq).stem), ("tiny30", "tiny", "30", "reduced"))

    def test_reference_models_follow_the_naming_contract(self):
        expected_onnx = {naming.onnx_name(*model_id.split("__")) for model_id in naming.MODEL_IDS}
        expected_mxq = {naming.mxq_name(Path(name).stem) for name in expected_onnx}
        self.assertEqual({p.name for p in (REFERENCE / "onnx").glob("*.onnx")}, expected_onnx)
        self.assertEqual({p.name for p in (REFERENCE / "mxq").glob("*.mxq")}, expected_mxq)

    def test_memory_profile_finds_models_by_id(self):
        args = type("Args", (), {"model_id": ["tiny30__reduced"], "models_dir": REFERENCE / "mxq",
                                 "core_mode": "global8"})()
        (model_id, path), = profile_memory.discover(args)
        self.assertEqual(path.name, "tiny30__reduced__nhwc_b1_npusafe_vt_global8.mxq")


class SampleSelectionTests(unittest.TestCase):
    def test_numeric_folders_use_their_number_as_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_validation(Path(tmp), ["00003", "00001", "00002"])
            samples = evaluation.collect_samples(Path(tmp), 2, 3, seed=42, limit=0)
            self.assertEqual([label for _, label in samples], [1, 1, 2, 2, 3, 3])

    def test_synset_folders_are_labelled_in_sorted_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_validation(Path(tmp), ["n01443537", "n01440764", "n01484850"])
            samples = evaluation.collect_samples(Path(tmp), 1, 3, seed=42, limit=0)
            self.assertEqual([(path.parent.name, label) for path, label in samples],
                             [("n01440764", 0), ("n01443537", 1), ("n01484850", 2)])

    def test_stray_folders_do_not_change_numeric_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_validation(Path(tmp), ["00001", "00002"])
            expected = evaluation.collect_samples(Path(tmp), 2, 2, seed=42, limit=0)
            (Path(tmp) / ".ipynb_checkpoints").mkdir()
            (Path(tmp) / "@eaDir").mkdir()
            self.assertEqual(evaluation.collect_samples(Path(tmp), 2, 2, seed=42, limit=0), expected)

    def test_unexpected_folder_among_synsets_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_validation(Path(tmp), ["n01440764", "n01443537", "extra"])
            with self.assertRaisesRegex(ValueError, "unexpected: extra"):
                evaluation.collect_samples(Path(tmp), 1, 2, seed=42, limit=0)

    def test_measurement_defaults_follow_the_documented_protocol(self):
        with patch.object(sys, "argv", ["eval", "--models-dir", "m", "--val-dir", "v"]):
            args = evaluation.parse_args()
        self.assertEqual((args.images_per_class, args.seed, args.warmup_runs, args.core_mode),
                         (50, 42, 10, "global8"))
        self.assertEqual((args.memory_sample_ms, args.npu_sample_ms), (20.0, 500.0))

    def test_selection_is_deterministic_and_checks_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_validation(Path(tmp), ["00000", "00001"])
            first = evaluation.collect_samples(Path(tmp), 2, 2, seed=42, limit=0)
            self.assertEqual(first, evaluation.collect_samples(Path(tmp), 2, 2, seed=42, limit=0))
            with self.assertRaises(ValueError):
                evaluation.collect_samples(Path(tmp), 4, 2, seed=42, limit=0)
            with self.assertRaises(ValueError):
                evaluation.collect_samples(Path(tmp), 2, 3, seed=42, limit=0)


class RuntimeTests(unittest.TestCase):
    def test_no_sdk_required_for_import(self):
        self.assertTrue(callable(evaluation.discover_models))
        with patch.object(evaluation, "qbruntime", None):
            with self.assertRaisesRegex(RuntimeError, "qbruntime"):
                evaluation.require_qbruntime()


class ReportTests(unittest.TestCase):
    """Render the report from the recorded reference benchmark."""

    def setUp(self):
        self.record = json.loads((REPORT / "benchmark.json").read_text())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "summary.md"

    def render(self):
        evaluation.write_summary_markdown(
            self.path, self.record["run"], self.record["models"], self.record["comparisons"])
        return self.path.read_text()

    def test_complete_run_is_pass(self):
        self.assertIn("**PASS**", self.render())

    def test_incomplete_telemetry_is_not_pass(self):
        self.record["models"][0]["npu_telemetry"]["status"] = "failed"
        self.assertIn("**CHECK**", self.render())

    def test_isolated_peak_mismatch_reported(self):
        source = json.loads((REPORT / "isolated_memory/summary.json").read_text())
        item = copy.deepcopy(source["models"][0])
        item["npu_peak_reported_mb"] = 999
        destination = self.path.parent / "isolated_memory"
        destination.mkdir()
        (destination / "summary.json").write_text(json.dumps({"models": [item]}))
        self.assertIn("독립 측정 1개 중 peak 일치 0개", self.render())


if __name__ == "__main__":
    unittest.main()
