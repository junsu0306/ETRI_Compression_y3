# ViT 압축과 Mobilint NPU 배포

ImageNet pretrained ViT-Tiny / ViT-Small을 압축해 Mobilint NPU에서 실행하기까지의 전체 과정을 담은 저장소다. 압축 학습부터 NPU 벤치마크까지 이 저장소의 코드만으로 처음부터 실행할 수 있다.

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

대상은 Tiny / Small × 전체 파라미터 30% / 50% 감소 4개 실험이다. 실험마다 baseline, reduced(FFN 채널 제거), patch slimming(reduced에서 token까지 축소) 세 모델이 나와 ONNX 12개, MXQ 12개가 된다.

## 실행 환경

코드는 한 저장소이고 실행 환경은 둘이다.

| 환경 | 단계 | 필요한 것 |
|---|---|---|
| x86 GPU 서버 | 1–6 | CUDA GPU, Python 3.10, PyTorch, timm 1.0.25, Mobilint `qbcompiler` |
| NPU 장치 호스트 | 7 | Mobilint Aries, `qbruntime`, NumPy, OpenCV. 학습용 패키지는 필요 없다 |

설치와 환경 세부 사항은 [docs/RUNBOOK.md](docs/RUNBOOK.md)에 있다. Mobilint SDK(컴파일러, 런타임)는 Mobilint에서 받아 설치하며 이 저장소에 포함하지 않는다.

## 빠른 시작

ImageNet을 [docs/DATASET.md](docs/DATASET.md)에 따라 준비한 뒤, x86 서버의 저장소 루트에서 실행한다.

```bash
# 단위 테스트 (GPU·데이터 불필요)
python -m unittest discover -s tests -t .

# 축소 실행: 모델 하나, 작은 부분집합으로 1–6단계를 한 번에 확인 (정확도는 의미 없음)
python -m dataset.prepare_imagenet subset --source /data/imagenet --output data/imagenet_smoke
python -m scripts.launch --config configs/smoke.yaml

# 전체 실행: 4개 실험을 GPU에 하나씩 배정
python -m scripts.launch --config configs/final.yaml --data-path /data/imagenet --detach
```

같은 명령을 다시 실행하면 끝난 단계는 건너뛰고 이어서 진행한다. 결과는 `output/final/`에 쌓이고, 끝나면 `output/final/reports/RESULTS.md`와 장치로 옮길 `output/final/deploy/`가 만들어진다.

NPU 장치에서는 `deploy/` 폴더를 받아 벤치마크를 실행한다.

```bash
python3 -m benchmark.profile_memory --models-dir deploy/mxq --output-dir results/npu_benchmark/run1/isolated_memory
python3 -m benchmark.eval_imagenet  --models-dir deploy/mxq --val-dir /data/imagenet/val \
  --output-dir results/npu_benchmark/run1
```

## 결과

전체 표는 [docs/RESULTS.md](docs/RESULTS.md)에 있다. 아래는 참고 실행 `final_20261006`의 ImageNet validation 50,000장 Top-1이다. 이 저장소의 코드로 처음부터 다시 실행한 서버 결과(공식 ImageNet, RTX A5000)는 같은 문서의 2절에 따로 있다.

| 실험 | 파라미터 (baseline → 압축) | 서버 FP32: baseline / reduced / patch | NPU: baseline / reduced / patch | NPU latency ms: baseline / reduced / patch |
|---|---|---|---|---|
| Tiny30 | 5.72M → 4.00M | 74.660 / 72.212 / 67.902 | 74.226 / 70.880 / 53.528 | 3.192 / 3.065 / 2.408 |
| Tiny50 | 5.72M → 2.86M | 74.660 / 63.230 / 59.252 | 74.226 / 62.442 / 41.164 | 3.170 / 2.976 / 2.388 |
| Small30 | 22.05M → 15.43M | 80.502 / 79.624 / 77.970 | 80.486 / 79.324 / 70.094 | 5.884 / 5.569 / 4.585 |
| Small50 | 22.05M → 11.02M | 80.502 / 74.002 / 70.736 | 80.486 / 73.616 / 61.082 | 5.821 / 5.310 / 4.204 |

- FFN reducing은 NPU에서 정확도를 서버 FP32와 비슷하게 유지하면서 latency와 메모리를 줄인다.
- Patch Slimming은 NPU에서 가장 빠르고 메모리도 가장 적지만, MXQ로 변환한 뒤 정확도가 서버 FP32보다 7.9~18.1%p 더 떨어진다. MXQ 변환의 양자화에서 생기는 것으로 판단하며, 다시 실행해도 나타날 것으로 예상하는 현상이다.

참고 실행의 ONNX 12개와 MXQ 12개, 측정 기록은 [reference/final_20261006](reference/final_20261006/README.md)에 있다(Git LFS). 새로 실행한 산출물과 섞이지 않도록 실행기의 출력 폴더와 분리되어 있다.

## 저장소 구조

```text
configs/final.yaml            최종 실험 설정 (configs/smoke.yaml: 축소 실행)
dataset/                      ImageNet 폴더 구조 준비·검사, 부분집합
compression/                  메모리 예산, global L2 pruning, KD 학습, reducing
patch_slimming/               token 중요도·탐색·복원(psvit/), 탐색과 fine-tuning 실행
export/                       ONNX export와 검증
evaluation/                   서버 FP32 정확도, CUDA 메모리 프로파일
compile/                      calibration tensor 생성, MXQ 컴파일
benchmark/                    NPU 장치: ImageNet 평가, 메모리, 단일 이미지 확인
common/                       양쪽이 함께 쓰는 전처리·파일 이름 규칙
scripts/                      실행기(launch, run_pipeline), 결과 보고서
tests/                        단위 테스트 (test_device_*: NPU 호스트에서도 실행)
docs/                         방법론, 실행 안내, 데이터 준비, 결과
reference/final_20261006/     참고 실행의 ONNX·MXQ와 측정 기록
```

## 문서

| 문서 | 내용 |
|---|---|
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | 각 단계의 방법, 고정 설정, ONNX 계약, 전처리, 측정 기준 |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | 환경 설치, 단계별 명령, 재개, 장치로 옮겨 벤치마크하기 |
| [docs/DATASET.md](docs/DATASET.md) | ImageNet 준비와 calibration 이미지 |
| [docs/RESULTS.md](docs/RESULTS.md) | 실행 결과와 참고 결과 |
