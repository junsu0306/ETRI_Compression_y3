"""ONNX export: the static batch-1 wrapper and the real exporter on untrained fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import onnx
import timm
import torch

from compression.prepare_budget import build_target, save_model
from export import export_onnx as exporter
from patch_slimming.psvit.architecture import build_spec
from patch_slimming.psvit.model_utils import get_shape

ROOT = Path(__file__).resolve().parents[1]
TOKEN_SCHEDULE = [197, 197, 151, 131, 131, 111, 111, 91, 91, 81, 81, 1]


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(2022)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_static_wrapper_matches_original_model(self):
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False).eval()
        wrapper = exporter.baseline_wrapper(model, SimpleNamespace(npu_safe=True))
        image = torch.randn(1, 224, 224, 3)
        with torch.inference_mode():
            torch.testing.assert_close(wrapper(image), model(image.permute(0, 3, 1, 2)), rtol=0, atol=1e-6)

    def test_exporter_satisfies_the_onnx_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("vit_tiny_patch16_224", "vit_small_patch16_224"):
                baseline = timm.create_model(name, pretrained=False).eval()
                save_model(folder / "baseline.pt", baseline, name)
                reduced = build_target(baseline, 0.3, 0.95)
                save_model(folder / "reduced.pt", reduced, name)
                # Two leading identity selections must not appear as MatMul nodes.
                keep = [torch.arange(count) for count in TOKEN_SCHEDULE]
                spec = build_spec(name, get_shape(reduced), keep, [0.0] * 12, epsilon=0.02,
                                  error_metric="sample_relative_frobenius_sq_mean",
                                  search_step=10, score_mode="paper_path_energy_dp")
                spec.to_json(str(folder / "architecture.json"))
                torch.save({"model": reduced.state_dict(),
                            "mlp_dims": [b.mlp.fc1.out_features for b in reduced.blocks]}, folder / "patch.pt")
                stages = [("baseline", ["--model", name, "--baseline-checkpoint", folder / "baseline.pt"]),
                          ("reduced", ["--reduced", folder / "reduced.pt"]),
                          ("patch", ["--arch", folder / "architecture.json", "--weights", folder / "patch.pt"])]
                for kind, flags in stages:
                    with self.subTest(model=name, kind=kind):
                        output = folder / f"{name}_{kind}.onnx"
                        subprocess.run([sys.executable, "-m", "export.export_onnx", "--kind", kind,
                                        *map(str, flags), "--output", str(output), "--npu-safe", "--threads", "2"],
                                       cwd=ROOT, check=True, capture_output=True)
                        record = json.loads(output.with_suffix(".verified.json").read_text())
                        self.assertLess(record["pytorch_ort_max_abs"], 1e-4)
                        self.assertEqual(record["input_shape"], [1, 224, 224, 3])
                        self.assertEqual(record["output_shape"], [1, 1000])
                        self.assertFalse(set(record["ops"]) & exporter.BANNED_OPS)
                        graph = onnx.load(str(output))
                        self.assertEqual([i.name for i in graph.graph.input], ["input"])
                        self.assertEqual([o.name for o in graph.graph.output], ["output"])
                        self.assertTrue(any(op.domain in ("", "ai.onnx") and op.version == 17
                                            for op in graph.opset_import))
                        self.assertEqual(exporter.identity_selections(graph), [])

    def test_identity_selection_is_detected(self):
        from onnx import TensorProto, helper, numpy_helper
        import numpy as np

        def graph(matrix):
            return helper.make_model(helper.make_graph(
                [helper.make_node("MatMul", ["P", "X"], ["Y"], name="/slim.0/select")], "selector",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 2])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, matrix.shape[0], 2])],
                [numpy_helper.from_array(matrix.astype(np.float32), name="P")]))

        self.assertEqual(exporter.identity_selections(graph(np.eye(3))), ["/slim.0/select"])
        for matrix in (np.eye(3)[[0, 2]], np.eye(3)[[1, 0, 2]]):
            self.assertEqual(exporter.identity_selections(graph(matrix)), [])


if __name__ == "__main__":
    unittest.main()
