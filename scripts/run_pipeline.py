#!/usr/bin/env python3
"""Resumable x86 pipeline for the models assigned to one GPU.

    budget -> structural (pruning + KD, reducing) -> patch (search, fine-tuning)
           -> export (ONNX) -> memory -> evaluate (FP32) -> compile (calibration, MXQ)

One process owns one physical GPU and runs its models one after another. Every
stage is skipped when its output already exists, so the same command resumes an
interrupted run. The settings of a run are stored below its output root; a run
whose settings changed is refused rather than silently mixed.

    python -m scripts.run_pipeline --config configs/final.yaml --gpu 0 --phase all

scripts/launch.py starts one such process per GPU.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from common.files import claim_lock, read_json, write_json
from common.naming import INFERENCE_SCHEME, onnx_name

ROOT = Path(__file__).resolve().parents[1]
METHOD = "baseline_target_peak_global_l2_kd_static_patch"
STAGES = ("baseline", "reduced", "patch_slimming")
PHASES = ("budget", "structural", "patch", "export", "memory", "evaluate", "compile")
FOLDERS = {"budget": "00_budget", "structural": "01_structural", "reduced": "02_reduced",
           "patch": "03_patch_slimming", "onnx": "04_onnx", "memory": "05_memory",
           "evaluation": "06_evaluation", "mxq": "07_mxq"}


# ── configuration ─────────────────────────────────────────────────────────────

def absolute(path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def add_machine_arguments(parser: argparse.ArgumentParser) -> None:
    """Options that describe the machine, not the experiment."""
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--data-path", help="ImageNet root with train/ and val/ (or env IMAGENET_DIR)")
    parser.add_argument("--output-root", help="override output_root of the config")
    parser.add_argument("--gpus", help="comma-separated physical GPU ids; models are assigned in order")
    parser.add_argument("--num-workers", type=int, help="data loader workers per process")


def machine_arguments(args) -> list[str]:
    """Re-create the machine options for a child process."""
    forwarded = ["--config", str(absolute(args.config))]
    for flag, value in (("--data-path", args.data_path), ("--output-root", args.output_root),
                        ("--gpus", args.gpus), ("--num-workers", args.num_workers)):
        if value is not None:
            forwarded += [flag, str(value)]
    return forwarded


def load_config(args) -> dict:
    config = yaml.safe_load(absolute(args.config).read_text())
    if config["method"] != METHOD:
        raise ValueError(f"Unsupported method {config['method']!r}; this code implements {METHOD}")
    data_path = args.data_path or os.environ.get("IMAGENET_DIR") or config["data_path"]
    config["data_path"] = str(absolute(data_path))
    config["output_root"] = str(absolute(args.output_root or config["output_root"]))
    if args.gpus:
        config["gpus"] = [int(value) for value in str(args.gpus).split(",")]
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if not config.get("gpus"):
        raise ValueError("At least one GPU id is required (config 'gpus' or --gpus)")
    return config


def gpu_assignment(config: dict) -> dict[str, int]:
    """Models take the listed GPUs in order; with fewer GPUs they queue up."""
    gpus = config["gpus"]
    return {key: int(gpus[index % len(gpus)]) for index, key in enumerate(config["models"])}


def paths(config: dict, key: str) -> dict[str, Path]:
    root = Path(config["output_root"]) / key
    return {"root": root, **{name: root / folder for name, folder in FOLDERS.items()}}


def structural_config(config: dict, key: str) -> tuple[dict, Path]:
    model = config["models"][key]
    common = config["structural"]
    p = paths(config, key)
    cfg = {
        "model": model["model"], "data_path": config["data_path"],
        "epochs": common["epochs"], "batch_size": model["structural_batch"],
        "lr_schedule_epochs": common["lr_schedule_epochs"],
        "lr": common["lr"], "weight_decay": common["weight_decay"],
        "clip_grad": common["clip_grad"], "warmup_epochs": common["warmup_epochs"],
        "min_lr": common["min_lr"], "smoothing": common["smoothing"],
        "amp": common["amp"], "num_workers": config["num_workers"],
        "model_ema": common["model_ema"], "model_ema_decay": common["model_ema_decay"],
        "target_compression": model["target_compression"],
        "pruning_max_sparsity": common["pruning_max_sparsity"],
        "prune_refresh_steps": common["prune_refresh_steps"],
        "seed": config["seed"],
        "initial_checkpoint": str(p["budget"] / "baseline.pt"),
        "memory_budget_json": str(p["budget"] / "budget.json"),
        "kd_alpha": common["kd_alpha"], "kd_temperature": common["kd_temperature"],
        "kd_teacher": model["teacher"], "output_dir": str(p["structural"]),
    }
    # Shared measurement protocol; the budget always comes from the preparation stage.
    cfg.update({k: v for k, v in common.items() if k.startswith("memory_")})
    return cfg, p["root"] / "configs/structural.yaml"


def patch_config(config: dict, key: str) -> tuple[dict, Path]:
    model = config["models"][key]
    ps = config["patch_slimming"]
    p = paths(config, key)
    workers = config["num_workers"]

    def subset(samples, seed):
        return {"num_samples": samples, "batch_size": model["search_batch"], "seed": seed,
                "split": "train", "num_workers": workers}

    cfg = {
        "method": "patch_slimming_static",
        "seed": config["seed"],
        "model": {"name": model["model"], "num_classes": 1000,
                  "reduced_checkpoint": str(p["reduced"] / "reduced.pt")},
        "data_path": config["data_path"], "output_dir": str(p["patch"]),
        # Three disjoint training subsets: scoring / block reconstruction / acceptance.
        "score_data": subset(ps["score_samples"], 2022),
        "reconstruction_train": subset(ps["reconstruction_train_samples"], 2023),
        "reconstruction_eval": subset(ps["reconstruction_eval_samples"], 2024),
        "score": {"mode": "paper_path_energy_dp"},
        "search": {"epsilon": ps["epsilon"], "error_metric": ps["error_metric"],
                   "search_step": ps["search_step"], "max_search_iters": 10000,
                   "score_max_batches": None},
        "block_finetune": {"optimizer": "adamw", "learning_rate": 1e-5,
                           "weight_decay": 0.0, "epochs": ps["block_finetune_epochs"],
                           "grad_clip_norm": 1.0},
        "final_finetune": {"epochs": ps["final_finetune_epochs"], "lr": 2e-5, "weight_decay": 0.05,
                           "batch_size": model["patch_finetune_batch"], "warmup_epochs": 3,
                           "num_workers": workers},
    }
    return cfg, p["root"] / "configs/patch_slimming.yaml"


def pin_settings(path: Path, settings: dict) -> None:
    """Store the settings on first use; afterwards they must stay the same."""
    if path.exists():
        if yaml.safe_load(path.read_text()) != settings:
            raise ValueError(
                f"Settings differ from the ones this run was created with: {path}\n"
                "Resume with the same config and machine options, or start a new run "
                "with a different output_root (--output-root).")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(yaml.safe_dump(settings, sort_keys=False))
    temporary.replace(path)


def materialize(config: dict) -> list[dict]:
    """Create folders and per-model configs; refuse an output root made with other settings."""
    assignment = gpu_assignment(config)
    # Everything except where the run is stored and which GPUs execute it.
    pin_settings(Path(config["output_root"]) / "run_settings.yaml",
                 {k: v for k, v in config.items() if k not in ("output_root", "gpus")})
    manifest = []
    for key in config["models"]:
        p = paths(config, key)
        for folder in p.values():
            folder.mkdir(parents=True, exist_ok=True)
        (p["root"] / "configs").mkdir(exist_ok=True)
        configs = {}
        for name, (cfg, path) in (("structural_config", structural_config(config, key)),
                                  ("patch_config", patch_config(config, key))):
            pin_settings(path, cfg)
            configs[name] = str(path)
        manifest.append({"key": key, "gpu": assignment[key],
                         "paths": {k: str(v) for k, v in p.items()}, **configs})
    write_json(Path(config["output_root"]) / "experiment_manifest.json", manifest)
    return manifest


# ── execution helpers ─────────────────────────────────────────────────────────

def run(module: str, arguments: list[str], gpu: int, log: Path) -> None:
    """Run `python -m module` on one GPU, appending its output to a log file."""
    cmd = [sys.executable, "-m", module, *map(str, arguments)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as stream:
        stream.write("\n$ " + " ".join(cmd) + "\n")
        stream.flush()
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{module} failed with exit code {result.returncode}; see {log}")


def training_complete(path: Path, epochs: int) -> bool:
    if not path.exists():
        return False
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", -1)) + 1 >= epochs


def pin_terminal_checkpoint(directory: Path, epochs: int) -> Path:
    """Pin the exact terminal epoch; never substitute a later or best epoch."""
    import torch
    source = directory / "checkpoint_last.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    completed = int(checkpoint.get("epoch", -1)) + 1
    if completed != epochs:
        raise ValueError(f"Need exact epoch {epochs}, got {completed}: {source}")
    del checkpoint
    target = directory / "checkpoint_final.pt"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("Pinned terminal checkpoint differs from checkpoint_last")
    else:
        temporary = target.with_suffix(".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    write_json(directory / "terminal_checkpoint.json", {
        "completed_epochs": completed, "selection": "terminal_epoch",
        "source": str(source), "checkpoint": str(target), "sha256": digest})
    return target


def job_paths(job: dict) -> dict[str, Path]:
    return {k: Path(v) for k, v in job["paths"].items()}


# ── phases ────────────────────────────────────────────────────────────────────

def budget_phase(config: dict, job: dict) -> None:
    p = job_paths(job)
    if (p["budget"] / "budget.json").exists():
        return
    entry, settings = config["models"][job["key"]], config["structural"]
    run("compression.prepare_budget",
        ["--model", entry["model"], "--target-compression", entry["target_compression"],
         "--max-sparsity", settings["pruning_max_sparsity"], "--method", config["method"],
         "--seed", config["seed"], "--memory-batch-size", settings["memory_batch_size"],
         "--memory-warmups", settings["memory_warmups"], "--memory-runs", settings["memory_runs"],
         "--output-dir", p["budget"]],
        job["gpu"], p["budget"] / "prepare.log")


def structural_phase(config: dict, job: dict) -> None:
    p = job_paths(job)
    cfg = yaml.safe_load(Path(job["structural_config"]).read_text())
    epochs = int(cfg["epochs"])
    last = p["structural"] / "checkpoint_last.pt"
    if config["structural"]["checkpoint_selection"] != "terminal_epoch":
        raise ValueError("Structural compression uses the terminal epoch checkpoint")
    if not training_complete(last, epochs):
        arguments = ["--config", job["structural_config"]]
        if last.exists():
            arguments += ["--resume", last]
        run("compression.train", arguments, job["gpu"], p["structural"] / "train.log")
    terminal = pin_terminal_checkpoint(p["structural"], epochs)
    reduced = p["reduced"] / "reduced.pt"
    if not reduced.exists():
        run("compression.reduce",
            ["--model", cfg["model"], "--checkpoint", terminal, "--output", reduced],
            job["gpu"], p["reduced"] / "reduce.log")


def patch_phase(config: dict, job: dict) -> None:
    p = job_paths(job)
    reduced = p["reduced"] / "reduced.pt"
    if not reduced.exists():
        raise FileNotFoundError(reduced)
    if not all((p["patch"] / name).exists() for name in ("architecture.json", "searched_student.pt")):
        run("patch_slimming.run_search", ["--config", job["patch_config"], "--gpu", "0"],
            job["gpu"], p["patch"] / "search.log")
    last = p["patch"] / "compact_finetuned_last.pt"
    epochs = int(config["patch_slimming"]["final_finetune_epochs"])
    if not training_complete(last, epochs):
        arguments = ["--config", job["patch_config"]]
        if last.exists():
            arguments.append("--resume")
        run("patch_slimming.run_finetune", arguments, job["gpu"], p["patch"] / "finetune.log")


def export_phase(config: dict, job: dict) -> None:
    key, p = job["key"], job_paths(job)
    model = config["models"][key]["model"]
    commands = [
        ("baseline", ["--kind", "baseline", "--model", model,
                      "--baseline-checkpoint", p["budget"] / "baseline.pt"]),
        ("reduced", ["--kind", "reduced", "--reduced", p["reduced"] / "reduced.pt"]),
        ("patch_slimming", ["--kind", "patch", "--arch", p["patch"] / "architecture.json",
                            "--weights", p["patch"] / "compact_finetuned_best.pt"]),
    ]
    for stage, arguments in commands:
        output = p["onnx"] / onnx_name(key, stage)
        marker = output.with_suffix(".verified.json")
        if output.exists() and marker.exists():
            print(f"[skip] ONNX exists: {output}")
            continue
        run("export.export_onnx", [*arguments, "--output", output, "--npu-safe"],
            job["gpu"], p["onnx"] / f"export_{stage}.log")


def memory_phase(config: dict, job: dict) -> None:
    key, p = job["key"], job_paths(job)
    settings = config["structural"]
    commands = [
        ("baseline", ["--checkpoint", p["budget"] / "baseline.pt"]),
        ("reduced", ["--checkpoint", p["reduced"] / "reduced.pt"]),
        ("patch_slimming", ["--arch", p["patch"] / "architecture.json",
                            "--weights", p["patch"] / "compact_finetuned_best.pt"]),
    ]
    for stage, arguments in commands:
        output = p["memory"] / f"{key}__{stage}__b1_fp32.json"
        if output.exists():
            print(f"[skip] memory profile exists: {output}")
            continue
        run("evaluation.profile_memory",
            [*arguments, "--device", "cuda:0", "--batch-size", "1", "--details",
             "--warmups", settings["memory_warmups"], "--runs", settings["memory_runs"],
             "--output", output],
            job["gpu"], p["memory"] / f"profile_{stage}.log")


def evaluate_phase(config: dict, job: dict) -> None:
    key, p = job["key"], job_paths(job)
    evaluation = config.get("evaluation") or {}
    sources = {
        "baseline": ["--checkpoint", p["budget"] / "baseline.pt"],
        "pruned": ["--checkpoint", p["structural"] / "checkpoint_final.pt",
                   "--terminal-epochs", config["structural"]["epochs"]],
        "reduced": ["--checkpoint", p["reduced"] / "reduced.pt"],
        "patch_slimming": ["--checkpoint", p["patch"] / "compact_finetuned_best.pt",
                           "--arch", p["patch"] / "architecture.json"],
    }
    for kind, arguments in sources.items():
        output = p["evaluation"] / f"{kind}.json"
        if output.exists():
            continue
        run("evaluation.evaluate",
            ["--kind", kind, "--model", config["models"][key]["model"], *arguments,
             "--data-path", config["data_path"], "--label", key,
             "--expected-images", evaluation.get("val_images", 50000),
             "--expected-classes", evaluation.get("classes", 1000),
             "--num-workers", config["num_workers"], "--output", output],
            job["gpu"], output.with_suffix(".log"))


def calibration_directory(config: dict, gpu: int) -> Path:
    """Shared calibration tensors, generated once per run by whichever worker arrives first."""
    root = Path(config["output_root"])
    settings = config.get("calibration") or {}
    directory = root / "calibration"
    with (root / "calibration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (directory / "preprocess_profile.json").exists():
            if directory.exists():
                shutil.rmtree(directory)  # incomplete earlier attempt
            image_root = absolute(settings["image_root"]) if settings.get("image_root") \
                else Path(config["data_path"]) / "train"
            arguments = ["--image-root", image_root, "--output-dir", directory,
                         "--num-images", settings.get("num_images", 1000)]
            if settings.get("image_list"):
                arguments += ["--image-list", absolute(settings["image_list"])]
            else:
                arguments += ["--seed", settings.get("seed", config["seed"])]
            run("compile.calibration", arguments, gpu, root / "calibration.log")
    return directory


def compile_phase(config: dict, job: dict) -> None:
    p = job_paths(job)
    settings = config.get("compile") or {}
    scheme = settings.get("inference_scheme", INFERENCE_SCHEME)
    count = (config.get("calibration") or {}).get("num_images", 1000)
    calibration = calibration_directory(config, job["gpu"])
    run("compile.compile_mxq",
        ["--onnx-dir", p["onnx"], "--calib-dir", calibration, "--output-dir", p["mxq"],
         "--scheme", scheme, "--num-calibration", count],
        job["gpu"], p["mxq"] / "compile.log")


PHASE_FUNCTIONS = {"budget": budget_phase, "structural": structural_phase, "patch": patch_phase,
                   "export": export_phase, "memory": memory_phase, "evaluate": evaluate_phase,
                   "compile": compile_phase}


def record_time(job: dict, phase: str, seconds: float) -> None:
    """Accumulate wall time per phase across resumed invocations."""
    path = Path(job["paths"]["root"]) / "timings.json"
    timings = read_json(path) if path.exists() else {}
    entry = timings.setdefault(phase, {"seconds": 0.0, "invocations": 0})
    entry["seconds"] += seconds
    entry["invocations"] += 1
    entry["last_finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, timings)


class Terminated(SystemExit):
    """Raised by SIGTERM so that the status file records a deliberate stop."""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_machine_arguments(parser)
    parser.add_argument("--gpu", type=int, help="run the models assigned to this physical GPU")
    parser.add_argument("--phase", required=True, choices=["materialize", *PHASES, "all"])
    args = parser.parse_args()
    config = load_config(args)
    jobs = materialize(config)
    root = Path(config["output_root"])
    if args.phase == "materialize":
        for job in jobs:
            print(f"{job['key']}: GPU {job['gpu']}  {job['paths']['root']}")
        return
    if args.gpu is None:
        parser.error("--gpu is required for every phase except materialize")
    jobs = [job for job in jobs if job["gpu"] == args.gpu]
    if not jobs:
        parser.error(f"No model is assigned to GPU {args.gpu}; assignment follows 'gpus': {config['gpus']}")
    with (root / f"gpu{args.gpu}.lock").open("a") as lock:
        if not claim_lock(lock):
            raise SystemExit(f"Another pipeline process already owns GPU {args.gpu} in {root}")
        status_path = root / f"gpu{args.gpu}_status.json"
        status = {"gpu": args.gpu, "pid": os.getpid(), "jobs": [j["key"] for j in jobs],
                  "completed": [], "state": "running"}

        def terminated(signum, frame):
            raise Terminated(f"terminated by signal {signum}")

        signal.signal(signal.SIGTERM, terminated)  # so the status file records the stop
        try:
            for job in jobs:
                for phase in PHASES:
                    if args.phase not in (phase, "all"):
                        continue
                    status.update(key=job["key"], phase=phase,
                                  updated_utc=datetime.now(timezone.utc).isoformat())
                    write_json(status_path, status)
                    print(f"[queue] gpu={args.gpu} job={job['key']} phase={phase}", flush=True)
                    started = time.perf_counter()
                    try:
                        PHASE_FUNCTIONS[phase](config, job)
                    finally:  # interrupted attempts count towards the elapsed time as well
                        record_time(job, phase, time.perf_counter() - started)
                    status["completed"].append(f"{job['key']}/{phase}")
                    write_json(status_path, status)
            status["state"] = "complete"
        except BaseException as exc:
            stopped = isinstance(exc, (Terminated, KeyboardInterrupt))
            status.update(state="stopped" if stopped else "failed", error=str(exc))
            raise
        finally:
            status["updated_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(status_path, status)
    report = subprocess.run([sys.executable, "-m", "scripts.build_report", *machine_arguments(args)], cwd=ROOT)
    if report.returncode:  # the stages are done; the report can be rebuilt on its own
        print("[warning] scripts.build_report failed; run it again after fixing the cause", flush=True)


if __name__ == "__main__":
    main()
