"""CPU checks of the pipeline wiring: pinned settings, GPU assignment, resume protection."""
import argparse
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from common.files import claim_lock, lock_owner
from scripts import run_pipeline as runner


def load(config="configs/final.yaml", **overrides):
    options = dict(config=config, data_path=None, output_root=None, gpus=None, num_workers=None)
    options.update(overrides)
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("IMAGENET_DIR", None)
        return runner.load_config(argparse.Namespace(**options))


class FinalSettingsTests(unittest.TestCase):
    """The values below are the method; a change here is a different experiment."""

    def setUp(self):
        self.config = load()

    def test_structural_settings(self):
        expected = {"epochs": 30, "lr_schedule_epochs": 50, "lr": 5e-5, "weight_decay": 0.05,
                    "warmup_epochs": 5, "min_lr": 1e-6, "clip_grad": 1.0, "smoothing": 0.1,
                    "amp": True, "model_ema": True, "model_ema_decay": 0.9998,
                    "kd_alpha": 0.5, "kd_temperature": 4.0, "pruning_max_sparsity": 0.95,
                    "prune_refresh_steps": 100, "seed": 2022, "memory_penalty_weight": 1.0,
                    "memory_batch_size": 1, "memory_warmups": 10, "memory_runs": 50}
        batches = {"tiny30": 384, "tiny50": 384, "small30": 192, "small50": 192}
        for key, batch in batches.items():
            structural, _ = runner.structural_config(self.config, key)
            for name, value in expected.items():
                self.assertEqual(structural[name], value, name)
            self.assertEqual(structural["batch_size"], batch)
            self.assertEqual(structural["kd_teacher"], structural["model"])
            self.assertTrue(structural["memory_budget_json"].endswith(f"{key}/00_budget/budget.json"))
        self.assertEqual(self.config["structural"]["checkpoint_selection"], "terminal_epoch")
        self.assertEqual([m["target_compression"] for m in self.config["models"].values()],
                         [0.3, 0.5, 0.3, 0.5])

    def test_patch_slimming_settings(self):
        batches = {"tiny30": 512, "tiny50": 640, "small30": 256, "small50": 256}
        for key, batch in batches.items():
            cfg, _ = runner.patch_config(self.config, key)
            self.assertEqual(cfg["search"]["epsilon"], 0.02)
            self.assertEqual(cfg["search"]["error_metric"], "sample_relative_frobenius_sq_mean")
            self.assertEqual(cfg["search"]["search_step"], 10)
            self.assertEqual([(cfg[name]["num_samples"], cfg[name]["seed"]) for name in
                              ("score_data", "reconstruction_train", "reconstruction_eval")],
                             [(2048, 2022), (4096, 2023), (2048, 2024)])
            self.assertEqual(cfg["block_finetune"], {"optimizer": "adamw", "learning_rate": 1e-5,
                                                     "weight_decay": 0.0, "epochs": 3, "grad_clip_norm": 1.0})
            final = cfg["final_finetune"]
            self.assertEqual((final["epochs"], final["lr"], final["warmup_epochs"], final["weight_decay"]),
                             (30, 2e-5, 3, 0.05))
            self.assertEqual(final["batch_size"], batch)

    def test_calibration_and_compile_settings(self):
        self.assertEqual(self.config["calibration"]["num_images"], 1000)
        self.assertEqual(self.config["compile"]["inference_scheme"], "global8")
        self.assertEqual(self.config["evaluation"], {"val_images": 50000, "classes": 1000})


class MachineOptionTests(unittest.TestCase):
    def test_gpu_assignment_follows_available_gpus(self):
        self.assertEqual(list(runner.gpu_assignment(load()).values()), [0, 1, 2, 3])
        self.assertEqual(list(runner.gpu_assignment(load(gpus="5")).values()), [5, 5, 5, 5])
        self.assertEqual(list(runner.gpu_assignment(load(gpus="2,3")).values()), [2, 3, 2, 3])

    def test_data_path_precedence(self):
        self.assertEqual(load()["data_path"], str(runner.ROOT / "data/imagenet"))
        with patch.dict(os.environ, {"IMAGENET_DIR": "/env/imagenet"}):
            args = argparse.Namespace(config="configs/final.yaml", data_path=None, output_root=None,
                                      gpus=None, num_workers=None)
            self.assertEqual(runner.load_config(args)["data_path"], "/env/imagenet")
            args.data_path = "/cli/imagenet"
            self.assertEqual(runner.load_config(args)["data_path"], "/cli/imagenet")

    def test_only_final_method_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "other.yaml"
            path.write_text((runner.ROOT / "configs/final.yaml").read_text().replace(
                f"method: {runner.METHOD}", "method: other"))
            with self.assertRaises(ValueError):
                load(str(path))

    def test_smoke_config_is_a_reduced_copy_of_the_final_method(self):
        smoke, final = load("configs/smoke.yaml"), load()
        self.assertEqual(list(smoke["models"]), ["tiny30"])
        self.assertEqual(smoke["models"]["tiny30"], final["models"]["tiny30"])
        self.assertLess(smoke["structural"]["epochs"], final["structural"]["epochs"])


class ResumeProtectionTests(unittest.TestCase):
    def test_materialization_is_repeatable_but_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            config = load(output_root=folder)
            runner.materialize(config)
            runner.materialize(config)
            modified = copy.deepcopy(config)
            modified["structural"]["memory_penalty_weight"] = 2
            with self.assertRaisesRegex(ValueError, "output_root"):
                runner.materialize(modified)
            # GPU reassignment does not change results and must not block a resume.
            runner.materialize({**config, "gpus": [0]})

    def test_settings_of_later_stages_are_protected_as_well(self):
        with tempfile.TemporaryDirectory() as folder:
            config = load(output_root=folder)
            runner.materialize(config)
            for section, name, value in (("calibration", "seed", 1), ("compile", "inference_scheme", "all"),
                                         ("evaluation", "val_images", 100)):
                modified = copy.deepcopy(config)
                modified[section][name] = value
                with self.assertRaisesRegex(ValueError, "run_settings.yaml"):
                    runner.materialize(modified)

    def test_lock_names_its_live_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gpu0.lock"
            self.assertIsNone(lock_owner(path))
            with path.open("a") as lock:
                self.assertTrue(claim_lock(lock))
                self.assertEqual(lock_owner(path), os.getpid())
                with path.open("a") as second:
                    self.assertFalse(claim_lock(second))
            self.assertIsNone(lock_owner(path))  # released: a recorded PID alone means nothing

    def test_terminal_checkpoint_is_exact_and_repeatable(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            torch.save({"epoch": 29, "model": {"weight": torch.tensor([30.0])}}, directory / "checkpoint_last.pt")
            # A different best checkpoint must not affect selection.
            torch.save({"epoch": 26}, directory / "checkpoint_best.pt")
            final = runner.pin_terminal_checkpoint(directory, 30)
            self.assertEqual(final.name, "checkpoint_final.pt")
            self.assertEqual(torch.load(final, weights_only=False)["epoch"], 29)
            self.assertEqual(runner.pin_terminal_checkpoint(directory, 30), final)

    def test_terminal_checkpoint_rejects_under_and_overtrained(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for completed in (28, 31, 50):
                torch.save({"epoch": completed - 1}, directory / "checkpoint_last.pt")
                with self.assertRaisesRegex(ValueError, "Need exact epoch 30"):
                    runner.pin_terminal_checkpoint(directory, 30)

    def test_stored_lr_schedule_is_preserved(self):
        from compression.train import build_scheduler, validate_resume_schedule
        args = SimpleNamespace(epochs=30, lr_schedule_epochs=50, warmup_epochs=5, min_lr=1e-6)
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=5e-5)
        scheduler = build_scheduler(optimizer, args)
        self.assertEqual(scheduler.state_dict()["_schedulers"][1]["T_max"], 45)
        checkpoint = {"args": {"epochs": 50}, "epoch": 27}
        validate_resume_schedule(checkpoint, args)
        with self.assertRaisesRegex(ValueError, "LR horizon"):
            validate_resume_schedule(checkpoint, SimpleNamespace(epochs=30, lr_schedule_epochs=30))
        checkpoint["epoch"] = 30
        with self.assertRaisesRegex(ValueError, "beyond"):
            validate_resume_schedule(checkpoint, args)

    def test_finished_stage_is_not_rerun(self):
        with tempfile.TemporaryDirectory() as folder:
            config = load(output_root=folder)
            job = runner.materialize(config)[0]
            (Path(job["paths"]["budget"]) / "budget.json").write_text("{}")
            with patch.object(runner, "run", side_effect=AssertionError("must not rerun")):
                runner.budget_phase(config, job)


if __name__ == "__main__":
    unittest.main()
