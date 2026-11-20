"""Run the whole x86 pipeline: one worker process per GPU, each with its queue of models.

    python -m scripts.launch --config configs/final.yaml --data-path /data/imagenet --detach

Before starting, the launcher records the environment and the hashes of the
source files below the output root, and runs the unit tests. Re-running the same
command resumes the run. If the source or the settings changed since the run was
started, the launcher refuses to continue; use a new output root for the changed code.

    python -m scripts.launch --config configs/final.yaml --stop     # stop the launcher and its workers

--stop returns after every process of the run has exited, so the run can be
resumed right away.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from common.files import claim_lock, lock_owner, read_json, sha256_file, write_json
from scripts.run_pipeline import ROOT, absolute, add_machine_arguments, gpu_assignment, \
    load_config, machine_arguments, materialize

SOURCE_FOLDERS = ("common", "compression", "patch_slimming", "export", "evaluation", "compile", "scripts")
PACKAGES = ("torch", "torchvision", "timm", "numpy", "onnx", "onnxruntime", "opencv-python",
            "PyYAML", "psutil", "qbcompiler")


def source_manifest(config_path: Path) -> dict[str, str]:
    """Hashes of the code that produces results and of the experiment config."""
    files = [path for folder in SOURCE_FOLDERS for path in sorted((ROOT / folder).rglob("*.py"))]
    manifest = {str(path.relative_to(ROOT)): sha256_file(path) for path in files}
    manifest["<config>"] = sha256_file(config_path)
    return manifest


def command_output(command) -> str | None:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def environment() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"recorded_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "glibc": " ".join(platform.libc_ver()),
            "hostname": platform.node(), "packages": packages,
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "gpus": command_output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                                    "--format=csv"])}


STOP_GRACE_SECONDS = 60


class StopRequested(Exception):
    """SIGTERM or SIGINT arrived at the launcher."""


def stop_group(pid: int, signum=signal.SIGTERM) -> None:
    """Signal a worker together with every process it started."""
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:  # not a group leader (started by hand) or already gone
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def stop_workers(children: dict) -> None:
    """Terminate the worker groups and return only when all of them have exited."""
    for child in children.values():
        if child.poll() is None:
            stop_group(child.pid)
    deadline = time.monotonic() + STOP_GRACE_SECONDS
    for child in children.values():
        try:
            child.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            stop_group(child.pid, signal.SIGKILL)
            child.wait()


def stop_run(root: Path) -> None:
    """Stop the launcher of this run, or the workers a dead launcher left behind.

    Only processes that hold a lock of this run are signalled, so a PID that was
    reused by an unrelated process is never touched.
    """
    locks = [root / "launcher.lock", *sorted(root.glob("gpu*.lock"))]
    launcher = lock_owner(locks[0])
    if launcher:
        os.kill(launcher, signal.SIGTERM)  # a live launcher stops its own workers
        targets = [f"launcher {launcher}"]
    else:
        owners = [owner for owner in map(lock_owner, locks[1:]) if owner]
        for owner in owners:
            stop_group(owner)
        targets = [f"worker {owner}" for owner in owners]
    if not targets:
        print("Nothing is running")
        return
    deadline = time.monotonic() + 2 * STOP_GRACE_SECONDS
    while any(lock_owner(lock) for lock in locks):
        if time.monotonic() > deadline:
            raise SystemExit(f"Still running after {2 * STOP_GRACE_SECONDS} s: {root}")
        time.sleep(1)
    status_path = root / "launcher_status.json"
    if not launcher and status_path.exists():
        write_json(status_path, {**read_json(status_path), "state": "stopped",
                                 "updated_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Stopped {', '.join(targets)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_machine_arguments(ap)
    ap.add_argument("--detach", action="store_true", help="run in the background; follow launcher.log")
    ap.add_argument("--stop", action="store_true", help="stop the running launcher and its workers")
    ap.add_argument("--skip-tests", action="store_true", help="do not run the unit tests first")
    args = ap.parse_args()
    config = load_config(args)
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    if args.stop:
        stop_run(root)
        return
    forwarded = machine_arguments(args)
    if args.detach:
        command = [sys.executable, "-m", "scripts.launch", *forwarded]
        if args.skip_tests:
            command.append("--skip-tests")
        with (root / "launcher.log").open("a") as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Launcher PID={child.pid}; log={root / 'launcher.log'}")
        return
    with (root / "launcher.lock").open("a") as lock:
        if not claim_lock(lock):
            raise SystemExit(f"A launcher is already running for {root}")
        current = source_manifest(absolute(args.config))
        manifest_path = root / "source_manifest.json"
        if manifest_path.exists():
            recorded = read_json(manifest_path)
            changed = sorted(name for name in set(current) | set(recorded)
                             if current.get(name) != recorded.get(name))
            if changed:
                raise SystemExit(
                    f"Source files changed since this run was started in {root}:\n  "
                    + "\n  ".join(changed[:20])
                    + "\nRestore them to resume, or start a new run with --output-root.")
        else:
            write_json(manifest_path, current)
        environment_path = root / "environment.json"  # one entry per start or resume
        history = read_json(environment_path) if environment_path.exists() else []
        write_json(environment_path, [*history, environment()])
        jobs = materialize(config)
        if not args.skip_tests:
            with (root / "tests.log").open("w") as stream:
                tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                                       cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                                       stdout=stream, stderr=subprocess.STDOUT)
            if tests.returncode:
                raise SystemExit(f"Unit tests failed; see {root / 'tests.log'}")
        assignment = gpu_assignment(config)
        print(f"{len(jobs)} models -> GPUs {assignment}", flush=True)
        children, streams = {}, []
        status_path = root / "launcher_status.json"

        def write_status(state: str) -> None:
            write_json(status_path, {
                "pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(), "state": state,
                "workers": {gpu: {"pid": child.pid, "exit_code": child.poll()}
                            for gpu, child in children.items()}})

        def request_stop(signum, frame):
            raise StopRequested(signum)

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        try:
            for gpu in sorted(set(assignment.values())):
                stream = (root / f"gpu{gpu}_pipeline.log").open("a")
                streams.append(stream)
                children[gpu] = subprocess.Popen(
                    [sys.executable, "-m", "scripts.run_pipeline", *forwarded, "--gpu", str(gpu), "--phase", "all"],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                codes = [child.poll() for child in children.values()]
                state = "running" if any(code is None for code in codes) else \
                    ("complete" if all(code == 0 for code in codes) else "failed")
                write_status(state)
                if state != "running":
                    break
                time.sleep(30)
        except StopRequested as request:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            stop_workers(children)
            write_status("stopped")
            raise SystemExit(f"Stopped by signal {request}; run the same command again to resume")
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            for stream in streams:
                stream.close()
        report = subprocess.run([sys.executable, "-m", "scripts.build_report", *forwarded], cwd=ROOT)
        if state == "complete" and (report.returncode or
                                    read_json(root / "reports/RESULTS.json")["state"] != "complete"):
            state = "failed"
            write_json(status_path, {**read_json(status_path), "state": state,
                                     "error": "Required artifacts are missing or not verified"})
        print(f"Run {state}: {root}", flush=True)
        raise SystemExit(0 if state == "complete" else 1)


if __name__ == "__main__":
    main()
