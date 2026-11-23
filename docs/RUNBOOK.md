# 실행 안내

모든 명령은 저장소 루트에서 `python -m <모듈>` 형태로 실행한다. 1–6절은 x86 GPU 서버, 7절은 NPU 장치 호스트에서 수행한다.

## 1. x86 환경

필요한 것: CUDA GPU, Python 3.10, PyTorch(CUDA 빌드), Mobilint `qbcompiler`.

`qbcompiler`는 설치된 PyTorch의 CUDA 라이브러리에 직접 링크되어 있다. 그래서 다음을 지킨다.

- 컴파일러가 설치된 환경의 **PyTorch를 바꾸지 않는다.** 버전을 바꾸면 컴파일러가 로드되지 않을 수 있다.
- `timm`은 PyTorch를 끌어올리지 않도록 `--no-deps`로 설치한다.
- `qbcompiler 1.1.0`은 Python 3.10, x86_64, glibc 2.34 이상(Ubuntu 22.04 이상)이 필요하다.
- Mobilint SDK(wheel, 매뉴얼)는 Mobilint에서 받아 설치한다. 이 저장소에 넣지 않는다.

```bash
pip install --no-deps timm==1.0.25
pip install -r requirements/compress.txt     # 이미 있는 torch/torchvision은 그대로 둔다
pip install -r requirements/compile.txt
python -c "import qbcompiler, torch, timm; print(qbcompiler.__version__, torch.__version__, timm.__version__)"
```

`scripts.launch`는 항상 MXQ 컴파일까지 실행하므로 컴파일러가 있는 환경에서 쓴다. 컴파일러가 없는 서버에서 압축만 하려면 `scripts.run_pipeline`으로 `budget`부터 `evaluate`까지 단계를 하나씩 실행하고(5절), `04_onnx/`의 ONNX를 컴파일러가 있는 서버로 옮겨 `compile.compile_mxq`를 실행한다(6절). 단위 테스트는 두 requirements 파일의 패키지가 모두 있어야 돈다.

검증에 쓴 환경:

| 항목 | 값 |
|---|---|
| OS / glibc / kernel | Ubuntu 22.04.5 LTS / 2.35 / 5.4.0-216 (Docker 컨테이너) |
| GPU / driver | NVIDIA RTX A5000 × 4 / 570.133.07 |
| Python | 3.10.12 |
| PyTorch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| timm | 1.0.25 |
| NumPy / OpenCV | 1.26.0 / 4.11.0.86 |
| ONNX / ONNX Runtime | 1.16.2 / 1.23.2 |
| qbcompiler | 1.1.0 |

실행을 시작하거나 재개할 때마다 같은 정보가 `<output_root>/environment.json`에 차례로 추가된다. 참고 실행 `final_20261006`의 압축 단계는 PyTorch 2.6.0+cu118, RTX 2080 Ti에서 수행했다.

## 2. 테스트

```bash
python -m unittest discover -s tests -t .
```

GPU, ImageNet, SDK 없이 CPU에서 1분 안팎으로 끝난다. 설정 값 고정, 재개 보호, 메모리 항, pruning·reducing 등가성, token 탐색의 불변식, ONNX export 계약, calibration·컴파일러 호출 인자, 전처리, NPU 보고서 생성을 검사한다.

## 3. 데이터

[DATASET.md](DATASET.md)에 따라 ImageNet을 준비하고 확인한다.

```bash
python -m dataset.prepare_imagenet check --root /data/imagenet
```

## 4. 축소 실행

전체 실행(RTX A5000 4장에서 약 23시간) 전에, 모델 하나(tiny30)와 작은 부분집합으로 모든 단계가 이어지는지 확인한다. 수 분의 학습과 MXQ 3개 컴파일(약 7분)로 끝난다. 정확도는 보지 않는다.

```bash
python -m dataset.prepare_imagenet subset --source /data/imagenet --output data/imagenet_smoke
python -m scripts.launch --config configs/smoke.yaml
```

`output/smoke/reports/RESULTS.md`의 상태가 `complete`이면 통과다. 도중에 중단하고 같은 명령을 다시 실행해 재개도 확인할 수 있다.

## 5. 전체 실행

```bash
python -m scripts.launch --config configs/final.yaml --data-path /data/imagenet --detach
tail -f output/final/launcher.log
```

실행기는 시작할 때 환경 정보와 소스 파일 해시를 기록하고 단위 테스트를 돌린 뒤, GPU마다 작업 프로세스를 하나씩 띄운다. 각 프로세스는 배정된 모델을 예산 → 학습 → reducing → Patch Slimming → ONNX → 메모리 → 평가 → 컴파일 순서로 처리한다.

### 장비에 따라 바꾸는 옵션

| 옵션 | 설정 키 | 설명 |
|---|---|---|
| `--data-path` | `data_path` | ImageNet 위치. 환경 변수 `IMAGENET_DIR`도 쓸 수 있다 |
| `--output-root` | `output_root` | 실행 결과 폴더 |
| `--gpus 0,1` | `gpus` | 쓸 GPU. 모델을 순서대로 배정하고, GPU가 모자라면 같은 GPU에서 차례로 실행한다 |
| `--num-workers` | `num_workers` | 프로세스당 data loader worker 수. CPU 코어 수에 맞춘다 |

`--data-path`와 `--num-workers`는 실행 설정에 기록되므로 재개할 때도 처음과 같은 값을 준다. `--gpus`는 재개할 때 바꿔도 된다.

`configs/final.yaml`의 나머지 값(epoch, lr, batch 크기 등)은 방법 자체다. 바꾸면 다른 실험이 된다([METHODOLOGY.md](METHODOLOGY.md)). GPU 메모리가 커도 batch를 키우지 않는다.

### 진행 상황

```text
output/final/
├── launcher.log, launcher_status.json    전체 상태
├── gpu<N>_pipeline.log, gpu<N>_status.json   GPU별 현재 모델·단계
├── environment.json, source_manifest.json    환경과 소스 해시
├── run_settings.yaml                     이 실행의 설정 (재개할 때 대조)
├── calibration/                          calibration tensor, images.tsv
├── <tiny30|tiny50|small30|small50>/
│   ├── 00_budget/                        baseline.pt, target_untrained.pt, budget.json
│   ├── 01_structural/                    train.log, metrics.jsonl, checkpoint_final.pt
│   ├── 02_reduced/reduced.pt
│   ├── 03_patch_slimming/                architecture.json, search.log, finetune.log, compact_finetuned_best.pt
│   ├── 04_onnx/                          ONNX 3개 + .verified.json
│   ├── 05_memory/, 06_evaluation/        서버 메모리, FP32 정확도
│   ├── 07_mxq/                           MXQ 3개 + .compile.json + .compile.log
│   ├── configs/                          이 모델에 실제로 쓴 설정
│   └── timings.json                      단계별 소요 시간
├── reports/RESULTS.md, RESULTS.json, *.csv
└── deploy/onnx, deploy/mxq, SHA256SUMS   NPU 장치로 옮길 폴더
```

### 중간 확인

설정이 잘못되었는지 일찍 알 수 있는 지점이다. 기대값은 참고 실행 기준이다.

| 시점 | 확인 | 기대 |
|---|---|---|
| 예산 단계 직후 | `budget.json`의 목표 구조 파라미터 수 | Tiny30 4,003,396 / Tiny50 2,862,256 / Small30 15,434,188 / Small50 11,023,204 (정확히 일치) |
| 학습 5 epoch | `metrics.jsonl`의 validation Top-1 | Tiny30 63.6 / Tiny50 39.3 / Small30 74.6 / Small50 59.7 부근 |
| 학습 10 epoch | 같음 | 69.8 / 55.9 / 77.9 / 68.9 부근 |
| 매 epoch | `train.log`의 `[Memory]` 줄 | 후보 peak가 예산 이하, penalty 0 |
| Reducing 직후 | pruned와 reduced의 FP32 정확도 | 서로 같음 |
| 탐색 직후 | `architecture.json`의 token 수 | 뒤 block으로 갈수록 같거나 줄고 마지막은 1 |
| Export 직후 | `.verified.json` | 금지 연산 없음, 오차 1e-4 미만 |

참고 실행의 epoch별 기록은 `reference/final_20261006/server_run/<실험>/structural_metrics.jsonl`에 있다.

### 재개와 보호 장치

- 같은 명령을 다시 실행하면 끝난 단계는 건너뛰고, 학습은 마지막으로 저장된 epoch부터 이어진다.
- 실행 설정(`run_settings.yaml`과 `<모델>/configs/*.yaml`: 학습·탐색·평가·calibration·컴파일 설정, 데이터 경로, worker 수)이 처음 실행 때와 다르면 거부한다. 설정을 바꾼 실험은 `--output-root`로 새 폴더에서 시작한다. GPU 배정은 바꿔도 된다.
- `scripts.launch`는 결과를 만드는 코드(`common`, `compression`, `patch_slimming`, `export`, `evaluation`, `compile`, `scripts`의 `.py`)나 사용한 설정 파일이 실행 시작 때와 달라졌으면 재개를 거부하고 바뀐 파일을 알려 준다. 파일을 되돌리거나 새 출력 폴더를 쓴다.
- 학습은 정확히 30 epoch에서 끝나야 한다. 더 학습된 checkpoint나 best checkpoint로 대체하지 않는다.
- 실행을 멈출 때는 같은 옵션에 `--stop`을 붙인다. 실행기와 작업 프로세스가 모두 끝난 뒤에 돌아오므로 바로 재개할 수 있다. 상태 파일에는 `stopped`로 남는다.
- ONNX를 다시 만들면 그 ONNX로 컴파일한 MXQ가 아닌 기존 MXQ는 재사용하지 않고 오류로 알린다. 해당 `.mxq`와 `.compile.json`을 지우고 다시 실행한다.

### 단계별로 실행하기

GPU 하나의 작업을 직접 실행하거나 특정 단계만 실행할 수 있다.

```bash
python -m scripts.run_pipeline --config configs/final.yaml --data-path /data/imagenet --gpu 0 --phase structural
python -m scripts.build_report --config configs/final.yaml --data-path /data/imagenet
```

`--phase`는 `budget`, `structural`, `patch`, `export`, `memory`, `evaluate`, `compile`, `all` 중 하나다. 각 단계는 앞 단계의 산출물이 있어야 한다.

## 6. Calibration과 컴파일만 따로 실행하기

이미 있는 ONNX를 컴파일할 때 쓴다. 파이프라인의 `compile` 단계가 하는 일과 같다.

```bash
python -m compile.calibration --image-root /data/imagenet/train --seed 2022 --output-dir output/calibration
python -m compile.compile_mxq --onnx-dir <onnx 폴더> --calib-dir output/calibration --output-dir <mxq 폴더>
```

- Calibration 출력 폴더는 새 폴더여야 한다. 컴파일러 진입점은 전처리 설정·개수·shape·dtype이 맞지 않는 tensor를 거부한다.
- RTX A5000에서 모델당 2~9분 걸렸다. MXQ와 기록 파일이 모두 있는 모델은 건너뛴다.
- `CUDA_VISIBLE_DEVICES`로 컴파일에 쓸 GPU를 고른다. 이것은 NPU 장치 번호와 무관하다.

## 7. NPU 장치에서 벤치마크

### 환경

필요한 것: Mobilint Aries가 장착된 호스트, 장치에 맞는 Mobilint SDK(`qbruntime`, `mobilint-cli`, 드라이버), Python 3.10 이상.

```bash
git lfs install && git lfs pull            # models/final의 ONNX·MXQ를 받는다
pip install -r requirements/device.txt
mobilint-cli status
python3 -c "import qbruntime; from mblt_tracker import NPUDeviceTracker; print('runtime OK')"
python3 -m unittest discover -s tests -t . -p "test_device*.py"
```

장치 쪽 코드는 `numpy`, `opencv`, `onnx`, `psutil`, `qbruntime`, `mblt_tracker`만 쓴다. `torch`, `timm`은 필요 없다. 참고 측정 환경은 Linux 5.15.185-tegra aarch64, Python 3.10.12, NumPy 2.2.6, OpenCV 4.13.0, qbruntime 1.1.0, mobilint-cli 1.2.0, Aries driver 1.11.0, firmware 1.2.5다.

### 모델과 데이터 옮기기

저장소의 최종 모델은 `models/final/`에 있다([README](../models/final/README.md)). 직접 실행한 모델을 평가하려면 x86 서버의 `output/final/deploy/`를 장치로 복사한다. 어느 쪽이든 해시를 먼저 확인한다.

ImageNet validation도 장치에 있어야 한다(class별 폴더, [DATASET.md](DATASET.md)). **서버에서 FP32 평가에 쓴 것과 같은 validation 사본을 쓴다.** 사본이 다르면 baseline 정확도부터 0.8~0.9%p 달라져([RESULTS.md](RESULTS.md) 1.5절) 서버와 NPU의 차이를 해석할 수 없다. 서버의 `val/` 폴더(약 6.4 GB)를 그대로 복사하는 것이 가장 확실하다.

```bash
(cd models/final && sha256sum -c SHA256SUMS)          # 저장소의 최종 모델
rsync -a <x86>:<repo>/output/final/deploy/ deploy/    # 직접 실행한 모델
(cd deploy && sha256sum -c SHA256SUMS)
rsync -a <x86>:/data/imagenet/val/ /data/imagenet/val/
```

### 실행 순서

```bash
MODELS=models/final            # 직접 실행한 모델이면 deploy
RUN=results/npu_benchmark/run1

# 1) 파일·데이터 확인 (NPU를 쓰지 않는다)
python3 -m benchmark.eval_imagenet --models-dir $MODELS/mxq --val-dir /data/imagenet/val --dry-run

# 2) 모델마다 1장: 로드와 추론이 되는지 확인
python3 -m benchmark.eval_imagenet --models-dir $MODELS/mxq --val-dir /data/imagenet/val \
  --limit 1 --warmup-runs 2 --output-dir results/npu_benchmark/smoke --fail-fast

# 3) 독립 프로세스 메모리 측정 (모델당 약 20초)
python3 -m benchmark.profile_memory --models-dir $MODELS/mxq \
  --warmups 10 --runs 100 --memory-seconds 10 --memory-sample-ms 20 --npu-sample-ms 500 \
  --output-dir $RUN/isolated_memory

# 4) 전체 평가: 12개 모델 × 50,000장 (약 2시간)
python3 -m benchmark.eval_imagenet --models-dir $MODELS/mxq --val-dir /data/imagenet/val \
  --classes 1000 --images-per-class 50 --core-mode global8 --device-id 0 \
  --warmup-runs 10 --memory-sample-ms 20 --npu-sample-ms 500 \
  --output-dir $RUN --fail-fast
```

- 측정 중에는 같은 NPU에서 다른 작업을 실행하지 않는다.
- 출력 폴더는 새 폴더여야 한다(미리 만든 `isolated_memory/`만 예외). 재실행할 때는 다른 이름을 쓴다.
- 결과: `summary.md`(표), `summary.csv`, `benchmark.json`(전체 지표), `samples.tsv`(평가 이미지와 정답), `predictions/*.npy`(모델별 Top-1 예측), `run_metadata.json`(환경).
- `--core-mode`는 컴파일의 inference scheme(`global8`)과 같아야 한다.

단일 이미지로 MXQ와 ONNX의 예측을 비교해 볼 수도 있다.

```bash
python3 -m benchmark.run_image --mxq $MODELS/mxq/tiny30__reduced__nhwc_b1_npusafe_vt_global8.mxq --image example.jpg
python3 -m benchmark.onnx_reference --model $MODELS/onnx/tiny30__reduced__nhwc_b1_npusafe.onnx --image example.jpg
```

### 결과 판단

새로 컴파일한 MXQ는 해시가 아니라 정확도로 비교한다. 참고값과 다시 학습했을 때의 기대 범위는 [RESULTS.md](RESULTS.md)에 있다.
