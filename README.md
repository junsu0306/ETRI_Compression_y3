# ViT 압축과 Mobilint NPU 배포

**IITP (ETRI Project) - On_Device_AI**

ImageNet pretrained ViT-Tiny / ViT-Small을 압축해 Mobilint NPU(Aries)에서 실행하기까지의 전체 과정을 담고 있습니다. 압축 학습부터 NPU 벤치마크까지 이 폴더의 코드만으로 처음부터 실행할 수 있습니다.

```text
Pretrained ViT (timm, augreg_in21k_ft_in1k)
  1. 메모리 예산       목표 구조를 추가 학습 없이 만들어 추론 peak 측정        compression/
  2. 구조적 압축       global L2 FFN pruning + logit KD 30 epoch → reducing    compression/
  3. Patch Slimming    정적 token 구조 탐색 → compact 모델 fine-tuning 30 epoch patch_slimming/
  4. ONNX export       NHWC batch 1 고정 그래프, PyTorch ↔ ONNX Runtime 검증    export/
  5. 서버 평가         FP32 ImageNet 정확도, CUDA 메모리                        evaluation/
  6. MXQ 컴파일        calibration 1,000장 → qbcompiler                         compile/
  ─────────────────── 여기까지 x86 GPU 서버, 아래는 NPU 장치 ───────────────────
  7. NPU 벤치마크      정확도·latency·메모리·전력                               benchmark/
```

대상은 Tiny / Small × 전체 파라미터 30% / 50% 감소, 4개 실험입니다. 실험마다 baseline, reduced(FFN 채널 제거), patch slimming(reduced에서 token까지 축소) 세 모델이 나와 ONNX 12개, MXQ 12개가 됩니다.

---

## 폴더 구조

```
ETRI_Compression_y3/
├── configs/                            # 실험 설정 (final.yaml: 최종 실험, smoke.yaml: 축소 실행)
├── dataset/                            # ImageNet 폴더 구조 준비·검사, 부분집합
├── compression/
│   ├── prepare_budget.py               # 메모리 예산 M_max 생성
│   ├── vit_pruning.py, vit_reducing.py # Pruning/Reducing 함수
│   ├── train.py, engine.py             # 압축 학습 메인 (KD + Pruning)
│   ├── memory_budget.py                # 메모리 항 (예산 초과분)
│   └── reduce.py                       # 학습이 끝난 모델의 채널 제거
├── patch_slimming/                     # token 중요도·탐색·복원(psvit/), 탐색과 fine-tuning 실행
├── export/                             # ONNX export와 검증
├── evaluation/                         # 서버 FP32 정확도, CUDA 메모리 측정
├── compile/                            # calibration tensor 생성, MXQ 컴파일
├── benchmark/                          # NPU 장치: ImageNet 평가, 메모리, 단일 이미지 확인
├── common/                             # 서버·장치가 함께 쓰는 전처리, 파일 이름 규칙
├── scripts/                            # 실행기(launch, run_pipeline), 결과 보고서
├── tests/                              # 단위 테스트 (test_device_*: NPU 호스트에서도 실행)
├── requirements/                       # 환경별 패키지 (compress, compile, device)
├── docs/                               # 방법론, 실행 안내, 데이터 준비, 결과
├── models/final/                       # 최종 ONNX·MXQ 각 12개 (Git LFS)
└── reference/final_20261006/           # 참고 실행의 측정 기록
```

---

## 방법

각 단계의 고정 설정과 측정 조건은 [docs/METHODOLOGY.md](docs/METHODOLOGY.md)에 있습니다.

### 1. 메모리 예산 (Memory Budget)

Pretrained 모델의 사본을 목표 비율로 pruning·reducing해 목표 구조를 먼저 만들고(추가 학습 없음), 그 구조의 FP32 추론 peak를 예산 M_max로 정합니다.

```text
M_max = 목표 구조의 CUDA peak allocated (MiB)
측정: FP32, batch 1, 224×224 random 입력, warmup 10회 뒤 forward 50회의 최대값
```

코드: `compression/prepare_budget.py`

### 2. Pruning (Global L2 FFN Pruning)

FFN(`D → M → D`)의 hidden channel마다 첫 Linear의 행 norm `‖W_fc1[j,:]‖₂`를 점수로 쓰고, 모든 block의 점수를 함께 비교해 낮은 채널부터 제거합니다. 채널 하나를 제거하면 FC1의 행과 bias, FC2의 대응 열이 함께 없어집니다. Attention, embedding 차원, classifier는 그대로 둡니다.

- 목표 30% / 50%는 FFN이 아니라 전체 모델 파라미터 수 대비 비율입니다.
- 학습 중에는 채널을 0으로 마스킹한 dense 모델을 유지하고, 제거할 채널의 순위는 100 step마다 갱신합니다.

코드: `compression/vit_pruning.py`

### 3. Knowledge Distillation과 메모리 항

Teacher는 같은 구조의 pretrained baseline(고정)이고 logit KL distillation을 씁니다. 매 epoch 현재 모델을 reducing한 사본의 추론 peak를 별도 프로세스에서 측정해, 예산을 넘은 만큼을 loss에 더합니다.

```text
L_task = (1−α)·CE(z_s, y) + α·T²·KL(softmax(z_t/T) ‖ softmax(z_s/T))     α 0.5, T 4.0
L_mem  = λ·max(0, M_candidate − M_max)                                    λ 1.0
L      = L_task + L_mem
```

메모리 항은 측정값에서 온 상수이므로 가중치에 대한 gradient가 없습니다. 학습은 30 epoch(AdamW, lr 5e-5)입니다.

코드: `compression/train.py`, `compression/engine.py`, `compression/memory_budget.py`

### 4. Reducing

마스킹된 FFN 채널을 물리적으로 제거해 작은 dense 모델(reduced)을 만듭니다. 살아남은 채널의 값은 EMA 가중치이고, block별 FFN 폭(`mlp_dims`)이 checkpoint에 함께 저장됩니다.

코드: `compression/reduce.py`, `compression/vit_reducing.py`

### 5. Patch Slimming

Reduced 모델에서 block마다 남길 token을 고정으로 정합니다. 입력마다 token을 고르는 동적 방식이 아닙니다. 마지막 block부터 앞으로 진행하며 중요도가 높은 patch를 10개씩 추가하고, 그 block만 3 epoch 복원 학습한 뒤 다음 block 출력의 상대 오차가 0.02 이하가 될 때까지 늘립니다. 이렇게 token 수가 줄어든 compact 모델을 CE로 30 epoch fine-tuning합니다.

참고 논문: Tang et al., *Patch Slimming for Efficient Vision Transformers*, CVPR 2022.

코드: `patch_slimming/`

### 6. ONNX export, MXQ 컴파일, NPU 측정

세 모델(baseline / reduced / patch slimming)을 NHWC, batch 1 고정 ONNX로 export하고 PyTorch와 ONNX Runtime의 출력을 비교해 검증합니다. ImageNet train 이미지 1,000장으로 calibration한 뒤 Mobilint `qbcompiler`로 MXQ(global8)를 만들고, NPU 장치에서 정확도·latency·메모리·전력을 측정합니다.

코드: `export/`, `compile/`, `benchmark/`

---

## 결과

전체 표는 [docs/RESULTS.md](docs/RESULTS.md)에 있습니다. 아래는 최종 모델([models/final](models/final/README.md))의 공식 ImageNet validation 50,000장 Top-1입니다. 서버는 RTX A5000, NPU는 Mobilint Aries에서 측정했습니다.

| 실험 | 파라미터 (baseline → 압축) | 서버 FP32: baseline / reduced / patch | NPU: baseline / reduced / patch | NPU latency ms: baseline / reduced / patch |
|---|---|---|---|---|
| Tiny30 | 5.72M → 4.00M | 75.458 / 72.468 / 68.270 | 74.126 / 71.812 / 52.476 | 3.020 / 2.886 / 2.320 |
| Tiny50 | 5.72M → 2.86M | 75.458 / 63.530 / 60.202 | 74.126 / 62.856 / 34.568 | 3.002 / 2.811 / 2.329 |
| Small30 | 22.05M → 15.43M | 81.380 / 79.964 / 78.240 | 81.356 / 79.520 / 69.626 | 5.714 / 5.435 / 4.474 |
| Small50 | 22.05M → 11.02M | 81.380 / 74.298 / 70.936 | 81.356 / 73.578 / 61.750 | 5.755 / 5.245 / 3.985 |

- FFN reducing은 NPU에서 정확도를 서버 FP32와 비슷하게 유지하면서(0.4~0.7%p 차이) latency와 메모리를 줄입니다.
- Patch Slimming은 NPU에서 가장 빠르고 메모리도 가장 적지만, MXQ로 변환한 뒤 정확도가 서버 FP32보다 8.6~25.6%p 더 떨어집니다. 참고 실행(7.9~18.1%p)에 이어 다시 실행해도 나타났으며, MXQ 변환의 양자화에서 생기는 것으로 판단합니다.

참고 실행 `final_20261006`의 결과는 [docs/RESULTS.md](docs/RESULTS.md) 1절에, 측정 기록은 [reference/final_20261006](reference/final_20261006/README.md)에 있습니다. 그 실행의 모델 파일은 저장소에 두지 않습니다.

---

## 실행 환경

코드는 하나이고 실행 환경은 둘입니다.

| 환경 | 단계 | 필요한 것 |
|---|---|---|
| x86 GPU 서버 | 1–6 | CUDA GPU, Python 3.10, PyTorch, timm 1.0.25, Mobilint `qbcompiler` |
| NPU 장치 호스트 | 7 | Mobilint Aries, `qbruntime`, NumPy, OpenCV. 학습용 패키지는 필요 없습니다 |

Mobilint SDK(컴파일러, 런타임)는 Mobilint에서 받아 설치하며 이 저장소에 포함하지 않습니다. 설치 세부 사항은 [docs/RUNBOOK.md](docs/RUNBOOK.md)에 있습니다.

## 빠른 시작

#### Step 1: 코드와 모델 받기

모델 파일(`.onnx`, `.mxq`, 합계 약 618 MiB)은 Git LFS로 관리합니다. clone하기 전에 [Git LFS](https://git-lfs.com)를 설치합니다.

```bash
git lfs install                                   # 처음 한 번
git clone -b 2026 https://github.com/ku-air/On_Device_AI.git
cd On_Device_AI/ETRI_Compression_y3
(cd models/final && sha256sum -c SHA256SUMS)      # macOS: shasum -a 256 -c SHA256SUMS
```

Git LFS 없이 clone하면 모델 파일이 130바이트 남짓한 포인터 파일로만 받아지거나, clone이 도중에 멈춰 일부 파일이 빠질 수 있습니다. 이때는 Git LFS를 설치하고 `git lfs install`을 실행한 뒤 다시 clone합니다. 포인터 파일만 받은 경우에는 받은 폴더에서 `git lfs pull`을 실행해도 됩니다.

이후의 모든 명령은 이 폴더(`ETRI_Compression_y3/`)에서 `python -m <모듈>` 형태로 실행합니다.

#### Step 2: 환경 설치 (x86 GPU 서버)

```bash
pip install --no-deps timm==1.0.25
pip install -r requirements/compress.txt         # 이미 설치된 torch/torchvision은 그대로 둡니다
pip install -r requirements/compile.txt
python -m unittest discover -s tests -t .        # 단위 테스트 (GPU·데이터 불필요)
```

`qbcompiler`는 설치된 PyTorch의 CUDA 라이브러리에 링크되어 있으므로, 컴파일러가 설치된 환경의 PyTorch는 바꾸지 않습니다.

#### Step 3: 데이터 준비와 축소 실행

ImageNet을 [docs/DATASET.md](docs/DATASET.md)에 따라 준비한 뒤, 모델 하나(tiny30)와 작은 부분집합으로 1–6단계가 이어지는지 확인합니다. 정확도는 의미가 없습니다.

```bash
python -m dataset.prepare_imagenet check --root /data/imagenet
python -m dataset.prepare_imagenet subset --source /data/imagenet --output data/imagenet_smoke
python -m scripts.launch --config configs/smoke.yaml
```

#### Step 4: 전체 실행

4개 실험을 GPU에 하나씩 배정합니다(RTX A5000 4장에서 약 23시간).

```bash
python -m scripts.launch --config configs/final.yaml --data-path /data/imagenet --detach
tail -f output/final/launcher.log
```

같은 명령을 다시 실행하면 끝난 단계는 건너뛰고 이어서 진행합니다. 결과는 `output/final/`에 쌓이고, 끝나면 `output/final/reports/RESULTS.md`와 장치로 옮길 `output/final/deploy/`가 만들어집니다.

#### Step 5: NPU 벤치마크 (NPU 장치 호스트)

MXQ 폴더를 지정해 벤치마크를 실행합니다. 아래는 저장소에 들어 있는 최종 모델(`models/final/`)을 쓰는 예입니다. 직접 실행한 모델은 `deploy/` 폴더를 장치로 옮겨 그 안의 `mxq/`를 지정합니다.

```bash
pip install -r requirements/device.txt           # qbruntime, mobilint-cli는 Mobilint SDK로 설치
python3 -m benchmark.profile_memory --models-dir models/final/mxq --output-dir results/npu_benchmark/run1/isolated_memory
python3 -m benchmark.eval_imagenet  --models-dir models/final/mxq --val-dir /data/imagenet/val \
  --output-dir results/npu_benchmark/run1
```

측정 옵션과 결과 파일은 [docs/RUNBOOK.md](docs/RUNBOOK.md) 7절에 있습니다.

---

## 문서

| 문서 | 내용 |
|---|---|
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | 각 단계의 방법, 고정 설정, ONNX 계약, 전처리, 측정 기준 |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | 환경 설치, 단계별 명령, 재개, 장치로 옮겨 벤치마크하기 |
| [docs/DATASET.md](docs/DATASET.md) | ImageNet 준비와 calibration 이미지 |
| [docs/RESULTS.md](docs/RESULTS.md) | 실행 결과와 참고 결과 |
| [models/final/README.md](models/final/README.md) | 최종 모델 목록, 무결성 확인, NPU 측정 원본 |
