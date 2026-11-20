#!/usr/bin/env python3
"""Compile NHWC ONNX models to MXQ with Mobilint qbcompiler.

Run inside the container that has qbcompiler and a CUDA GPU. The compiler is
always called with the vision_transformer preset, CPU offload enabled and no
manual quantization option. Each model is compiled in its own process and
leaves three files in the output folder:

    <onnx_stem>_vt_<scheme>.mxq            compiled model
    <onnx_stem>_vt_<scheme>.compile.log    compiler stdout/stderr
    <onnx_stem>_vt_<scheme>.compile.json   input/output hashes, compiler version, calibration

A model whose .mxq and .compile.json both exist is skipped; if that MXQ was
compiled from a different ONNX file the run stops instead. --dry-run checks the
inputs without importing qbcompiler.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time

from common.files import read_json, sha256_file, write_json
from common.naming import INFERENCE_SCHEME, INFERENCE_SCHEMES, ONNX_TAG, mxq_name
from compile.calibration import IMAGE_TABLE, PROFILE_FILE, validate

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PRESET = "vision_transformer"
CPU_OFFLOAD = True
TENSOR_LIST = "calibration_tensors.txt"


def compile_one(onnx_path, calib_txt, mxq_path, scheme=INFERENCE_SCHEME) -> None:
    """The only qbcompiler call in this repository. Do not add quantization options."""
    from qbcompiler import mxq_compile

    mxq_compile(
        model=str(onnx_path),
        calib_data_path=str(calib_txt),
        save_path=str(mxq_path),
        backend="onnx",
        inference_scheme=scheme,
        cpu_offload=CPU_OFFLOAD,
        config_preset=CONFIG_PRESET,
    )


def compiler_environment() -> dict:
    import qbcompiler
    import torch

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "qbcompiler": getattr(qbcompiler, "__version__", None),
        "qbcompiler_distribution": version("qbcompiler"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": version("numpy"),
        "onnx": version("onnx"),
    }


def worker(args) -> None:
    onnx_path, mxq_path, calib_dir = Path(args.worker_onnx), Path(args.worker_mxq), Path(args.calib_dir)
    temporary = mxq_path.with_name(mxq_path.stem + ".partial.mxq")
    temporary.unlink(missing_ok=True)
    started = time.perf_counter()
    compile_one(onnx_path, Path(args.worker_calib_txt), temporary, args.scheme)
    seconds = time.perf_counter() - started
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"qbcompiler produced no output for {onnx_path}")
    # The record is written first: an MXQ on disk always has its record.
    write_json(mxq_path.with_suffix(".compile.json"), {
        "onnx": str(onnx_path), "onnx_sha256": sha256_file(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "mxq": str(mxq_path), "mxq_sha256": sha256_file(temporary),
        "mxq_bytes": temporary.stat().st_size,
        "arguments": {"backend": "onnx", "inference_scheme": args.scheme,
                      "cpu_offload": CPU_OFFLOAD, "config_preset": CONFIG_PRESET},
        "calibration": {"directory": str(calib_dir),
                        "profile": read_json(calib_dir / PROFILE_FILE),
                        "image_table_sha256": sha256_file(calib_dir / IMAGE_TABLE)
                        if (calib_dir / IMAGE_TABLE).is_file() else None},
        "environment": compiler_environment(),
        "compile_seconds": seconds,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "note": "MXQ files are not byte-reproducible; compare compiled models by NPU accuracy",
    })
    temporary.replace(mxq_path)
    print(f"Saved: {mxq_path} ({seconds:.0f} s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-dir", help=f"compile every *{ONNX_TAG}.onnx in this folder")
    parser.add_argument("--onnx", action="append", default=[], help="compile this ONNX file; repeatable")
    parser.add_argument("--calib-dir", required=True, help="output folder of compile/calibration.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scheme", choices=INFERENCE_SCHEMES, default=INFERENCE_SCHEME)
    parser.add_argument("--num-calibration", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-onnx", help=argparse.SUPPRESS)
    parser.add_argument("--worker-mxq", help=argparse.SUPPRESS)
    parser.add_argument("--worker-calib-txt", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_onnx:
        worker(args)
        return

    models = [Path(p).resolve() for p in args.onnx]
    if args.onnx_dir:
        models += sorted(p.resolve() for p in Path(args.onnx_dir).glob(f"*{ONNX_TAG}.onnx"))
    if not models:
        parser.error("no ONNX model selected; use --onnx-dir or --onnx")
    for model in models:
        if not model.is_file():
            parser.error(f"ONNX file not found: {model}")
    calib_dir = Path(args.calib_dir).resolve()
    tensors = validate(calib_dir, args.num_calibration)
    output_dir = Path(args.output_dir).resolve()
    jobs = [(model, output_dir / mxq_name(model.stem, args.scheme)) for model in models]
    print(f"{len(tensors)} calibration tensors; {len(jobs)} ONNX models")
    print(f"preset={CONFIG_PRESET}, scheme={args.scheme}, cpu_offload={CPU_OFFLOAD}")
    pending = []
    for model, mxq in jobs:
        record = mxq.with_suffix(".compile.json")
        if mxq.exists() and record.exists():
            if read_json(record)["onnx_sha256"] != sha256_file(model):
                raise RuntimeError(f"{mxq} was compiled from a different {model.name}; "
                                   "remove it and its .compile.json to compile again")
            print(f"[skip] {mxq.name}")
            continue
        if mxq.exists():
            raise FileExistsError(f"{mxq} exists without its compile record; remove it or use a new output folder")
        print(f"[compile] {model.name} -> {mxq}")
        pending.append((model, mxq))
    if args.dry_run or not pending:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    calib_txt = output_dir / TENSOR_LIST
    listing = "\n".join(str(p.resolve()) for p in tensors) + "\n"
    if not calib_txt.exists() or calib_txt.read_text() != listing:
        temporary = calib_txt.with_name(f".{calib_txt.name}.{os.getpid()}.tmp")
        temporary.write_text(listing)
        temporary.replace(calib_txt)  # atomic: other compile processes may share this folder
    for model, mxq in pending:
        log = mxq.with_suffix(".compile.log")
        command = [sys.executable, "-m", "compile.compile_mxq", "--worker-onnx", str(model),
                   "--worker-mxq", str(mxq), "--worker-calib-txt", str(calib_txt),
                   "--calib-dir", str(calib_dir), "--output-dir", str(output_dir), "--scheme", args.scheme]
        started = time.perf_counter()
        with log.open("w") as stream:
            code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        if code:
            raise RuntimeError(f"Compilation failed for {model.name}; see {log}")
        print(f"[done] {mxq.name} in {time.perf_counter() - started:.0f} s", flush=True)
    print("Compiled models still need accuracy evaluation on the NPU.")


if __name__ == "__main__":
    main()
