"""Collect the results of a run into tables and a deployable model folder.

Reads only the files this run produced; missing values stay pending. Writes

    <output_root>/reports/RESULTS.md, RESULTS.json, *.csv
    <output_root>/deploy/onnx, deploy/mxq, deploy/SHA256SUMS   (verified models only)

deploy/ is the folder to copy to the NPU host for the benchmark.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil

from common.files import read_json, sha256_file
from common.naming import INFERENCE_SCHEME, mxq_name, onnx_name
from scripts.run_pipeline import PHASES, STAGES, add_machine_arguments, load_config, paths


def read(path):
    return read_json(path) if Path(path).exists() else None


def place(source: Path, target: Path) -> None:
    """Hard-link (or copy) a finished artifact into the deploy folder."""
    if target.exists() and target.stat().st_size == source.stat().st_size \
            and target.stat().st_mtime >= source.stat().st_mtime:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(target)


def cell(value) -> str:
    if value is None:
        return "대기"
    if isinstance(value, float):
        return f"{value:.2e}" if 0 < abs(value) < 1e-3 else f"{value:.4f}"
    return str(value)


def table(title, rows, report_dir, name):
    lines = [f"## {title}", ""]
    if not rows:
        return lines + ["측정/생성 대기 중.", ""]
    fields = list(rows[0])
    with (report_dir / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        lines.append("| " + " | ".join(cell(row[k]) for k in fields) + " |")
    return lines + [""]


def build(config: dict) -> None:
    root = Path(config["output_root"])
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    deploy = root / "deploy"
    scheme = (config.get("compile") or {}).get("inference_scheme", INFERENCE_SCHEME)
    result = {"method": config["method"],
              "structural_epochs": config["structural"]["epochs"],
              "lr_schedule_epochs": config["structural"]["lr_schedule_epochs"],
              "updated_utc": datetime.now(timezone.utc).isoformat(), "models": {}}
    accuracy_rows, memory_rows, budget_rows, onnx_rows, mxq_rows, time_rows = [], [], [], [], [], []
    deployed = []
    complete = True
    for key in config["models"]:
        p = paths(config, key)
        budget = read(p["budget"] / "budget.json")
        evaluation = {stage: read(p["evaluation"] / f"{stage}.json")
                      for stage in ("baseline", "pruned", "reduced", "patch_slimming")}
        memory = {stage: read(p["memory"] / f"{key}__{stage}__b1_fp32.json") for stage in STAGES}
        arch = read(p["patch"] / "architecture.json")
        terminal = read(p["structural"] / "terminal_checkpoint.json")
        timings = read(p["root"] / "timings.json") or {}
        onnx, mxq = {}, {}
        for stage in STAGES:
            path = p["onnx"] / onnx_name(key, stage)
            verification = read(path.with_suffix(".verified.json"))
            valid = bool(path.exists() and verification and
                         sha256_file(path) == verification["onnx_sha256"])
            onnx[stage] = {"path": str(path), "verified": valid, "verification": verification}
            if valid:
                onnx_rows.append({"model": key, "stage": stage, "file": path.name,
                                  "size_mib": path.stat().st_size / 2**20,
                                  "pytorch_ort_max_abs": verification["pytorch_ort_max_abs"],
                                  "sha256": verification["onnx_sha256"]})
            compiled = p["mxq"] / mxq_name(path.stem, scheme)
            record = read(compiled.with_suffix(".compile.json"))
            built = bool(valid and compiled.exists() and record
                         and record["onnx_sha256"] == verification["onnx_sha256"]
                         and sha256_file(compiled) == record["mxq_sha256"])
            mxq[stage] = {"path": str(compiled), "compiled": built, "record": record}
            if built:
                mxq_rows.append({"model": key, "stage": stage, "file": compiled.name,
                                 "size_mib": compiled.stat().st_size / 2**20,
                                 "compile_seconds": record["compile_seconds"],
                                 "qbcompiler": record["environment"]["qbcompiler"],
                                 "sha256": record["mxq_sha256"]})
                for source, folder in ((path, "onnx"), (path.with_suffix(".verified.json"), "onnx"),
                                       (compiled, "mxq"), (compiled.with_suffix(".compile.json"), "mxq")):
                    place(source, deploy / folder / source.name)
                    deployed.append(deploy / folder / source.name)
        training_memory = [read(f) for f in sorted((p["structural"] / "memory_budget").glob("epoch_*.json"))]
        if budget:
            budget_rows.append({"model": key,
                                "baseline_peak_mib": budget["baseline_profile"]["peak_allocated_bytes"] / 2**20,
                                "target_peak_mib": budget["max_mib"],
                                "target_trained_steps": budget["trained_steps"],
                                "target_parameters": budget["target_profile"]["parameters"],
                                "candidate_measurements": len(training_memory),
                                "candidate_max_mib": max((m["peak_allocated_bytes"] / 2**20
                                                          for m in training_memory), default=None)})
        for stage, data in evaluation.items():
            if data:
                accuracy_rows.append({"model": key, "stage": stage, "parameters": data["parameters"],
                                      "weight_mib": data["parameters"] * 4 / 2**20,
                                      "top1": data["metrics"]["acc1"], "top5": data["metrics"]["acc5"],
                                      "samples": data["samples"]})
        for stage, data in memory.items():
            if data:
                memory_rows.append({"model": key, "stage": stage,
                                    "weight_mib": data["parameter_bytes"] / 2**20,
                                    "peak_allocated_mib": data["peak_allocated_bytes"] / 2**20,
                                    "transient_peak_mib": data["transient_peak_bytes"] / 2**20,
                                    "module_output_sum_mib": data["activation"]["module_output_sum_bytes"] / 2**20,
                                    "latency_mean_ms": data["latency_mean_ms"]})
        if timings:
            time_rows.append({"model": key, **{f"{phase}_hours": timings[phase]["seconds"] / 3600
                                               if phase in timings else None for phase in PHASES}})
        ready = bool(budget and terminal
                     and terminal["completed_epochs"] == config["structural"]["epochs"]
                     and all(evaluation.values()) and all(memory.values())
                     and all(v["verified"] for v in onnx.values())
                     and all(v["compiled"] for v in mxq.values()))
        complete &= ready
        result["models"][key] = {"complete": ready, "budget": budget, "evaluation": evaluation,
                                 "memory": memory, "architecture": arch, "onnx": onnx, "mxq": mxq,
                                 "structural_terminal": terminal, "timings": timings}
    if deployed:
        sums = [f"{sha256_file(path)}  {path.relative_to(deploy).as_posix()}" for path in sorted(set(deployed))]
        (deploy / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    result["state"] = "complete" if complete else "in_progress"
    lines = ["# 실행 결과", "", f"상태: `{result['state']}`", "",
             "이 문서는 이 실행이 만든 JSON에서 자동 생성된다. 끝나지 않은 항목은 결과로 싣지 않는다.", "",
             f"Structural 학습 종료: {config['structural']['epochs']} epoch "
             f"(LR schedule 기준 길이 {config['structural']['lr_schedule_epochs']} epoch). "
             "이후 단계는 정확히 그 epoch의 `checkpoint_final.pt`를 쓴다.", ""]
    for name, rows, title in (("budget", budget_rows, "메모리 예산과 후보 측정"),
                              ("accuracy", accuracy_rows, "서버 FP32 ImageNet 정확도"),
                              ("memory", memory_rows, "서버 메모리"),
                              ("onnx", onnx_rows, "ONNX"), ("mxq", mxq_rows, "MXQ"),
                              ("timings", time_rows, "단계별 소요 시간")):
        lines += table(title, rows, report_dir, name)
    lines += ["## 읽는 법", "",
              "- 정확도는 ImageNet validation 전체를 FP32(AMP off)로 다시 평가한 값이다. 학습 중 기록되는 AMP/EMA 지표와 다르다.",
              "- Pruned는 dense shape에 zero mask를 적용한 모델, reduced는 그 채널을 물리적으로 제거한 모델이다.",
              "- 메모리는 batch 1, FP32, random input, warmup 뒤 반복 forward 중 CUDA allocator 최대값이다.",
              "- Module output 합은 별도 forward 1회의 출력 텐서 용량 합이며 동시에 살아 있는 메모리의 peak가 아니다.",
              "- ONNX 검증은 FP32 수치 일치 확인이다. MXQ의 정확도는 NPU 벤치마크로만 확인된다.",
              "- MXQ는 byte 단위로 재현되지 않는다. 컴파일 기록은 각 MXQ 옆의 `.compile.json`, `.compile.log`에 있다.", ""]
    for path, content in ((report_dir / "RESULTS.json", json.dumps(result, indent=2) + "\n"),
                          (report_dir / "RESULTS.md", "\n".join(lines))):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)
    print(f"{result['state']}: {report_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_machine_arguments(ap)
    config = load_config(ap.parse_args())
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "report.lock").open("a") as lock:  # the GPU workers of a run report concurrently
        fcntl.flock(lock, fcntl.LOCK_EX)
        build(config)


if __name__ == "__main__":
    main()
