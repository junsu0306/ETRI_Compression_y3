# 결과

모든 정확도는 ImageNet validation 50,000장 기준이다. 서버 값은 PyTorch FP32(AMP off), NPU 값은 MXQ를 Mobilint Aries에서 실행한 값이다. 서로 다른 실행의 결과는 한 표에 섞지 않는다.

## 1. 참고 실행 final_20261006

원본 기록은 [reference/final_20261006](../reference/final_20261006/README.md)에 있다. 서버 단계는 RTX 2080 Ti, 공식 배포본과 다른 ImageNet 사본([DATASET.md](DATASET.md))으로 수행했다.

### 1.1 정확도 (Top-1 / Top-5, %)

| 실험 | 모델 | 파라미터 | 서버 FP32 | NPU | NPU − 서버 Top-1 |
|---|---|---:|---|---|---:|
| Tiny30 | Baseline | 5,717,416 | 74.660 / 92.494 | 74.226 / 92.270 | −0.434 |
| Tiny30 | Reduced | 4,003,396 | 72.212 / 91.168 | 70.880 / 90.744 | −1.332 |
| Tiny30 | Patch Slimming | 4,003,396 | 67.902 / 88.130 | 53.528 / 78.384 | −14.374 |
| Tiny50 | Baseline | 5,717,416 | 74.660 / 92.494 | 74.226 / 92.270 | −0.434 |
| Tiny50 | Reduced | 2,862,256 | 63.230 / 85.832 | 62.442 / 85.314 | −0.788 |
| Tiny50 | Patch Slimming | 2,862,256 | 59.252 / 82.270 | 41.164 / 66.934 | −18.088 |
| Small30 | Baseline | 22,050,664 | 80.502 / 95.862 | 80.486 / 95.852 | −0.016 |
| Small30 | Reduced | 15,434,188 | 79.624 / 95.162 | 79.324 / 95.102 | −0.300 |
| Small30 | Patch Slimming | 15,434,188 | 77.970 / 94.190 | 70.094 / 90.184 | −7.876 |
| Small50 | Baseline | 22,050,664 | 80.502 / 95.862 | 80.486 / 95.852 | −0.016 |
| Small50 | Reduced | 11,023,204 | 74.002 / 92.234 | 73.616 / 92.124 | −0.386 |
| Small50 | Patch Slimming | 11,023,204 | 70.736 / 90.124 | 61.082 / 84.134 | −9.654 |

Pruned(zero mask를 적용한 dense 모델)의 서버 정확도는 reduced와 같다. Patch Slimming의 best checkpoint는 30 epoch 중 Tiny30 28번째, Tiny50 27번째, Small30 29번째, Small50 29번째 epoch였다.

### 1.2 NPU 성능

`global8`, batch 1, 모델당 warmup 10회. Latency는 `model.infer()` 구간의 평균이고, NPU 메모리는 장치 전체 telemetry의 관측 최대값(MB)이다.

| 실험 | 모델 | Latency ms | Baseline 대비 | NPU peak MB | MXQ MB | GFLOPs |
|---|---|---:|---:|---:|---:|---:|
| Tiny30 | Baseline | 3.192 | 1.00x | 12 | 7.6 | 2.507 |
| Tiny30 | Reduced | 3.065 | 1.04x | 11 | 6.0 | 1.834 |
| Tiny30 | Patch Slimming | 2.408 | 1.33x | 7 | 5.7 | 0.990 |
| Tiny50 | Baseline | 3.170 | 1.00x | 12 | 7.6 | 2.507 |
| Tiny50 | Reduced | 2.976 | 1.07x | 10 | 4.9 | 1.385 |
| Tiny50 | Patch Slimming | 2.388 | 1.33x | 5 | 4.6 | 0.750 |
| Small30 | Baseline | 5.884 | 1.00x | 35 | 24.5 | 9.198 |
| Small30 | Reduced | 5.569 | 1.06x | 28 | 18.3 | 6.594 |
| Small30 | Patch Slimming | 4.585 | 1.28x | 23 | 18.1 | 4.245 |
| Small50 | Baseline | 5.821 | 1.00x | 35 | 24.5 | 9.198 |
| Small50 | Reduced | 5.310 | 1.10x | 24 | 13.9 | 4.859 |
| Small50 | Patch Slimming | 4.204 | 1.38x | 18 | 13.8 | 2.998 |

12개 모델 × 50,000장에서 오류 이미지는 없었고, 모델마다 새 프로세스로 다시 잰 독립 메모리 측정도 같은 peak를 보였다. Latency 분포, 전력, 온도, host 메모리는 [summary.md](../reference/final_20261006/npu_benchmark/summary.md)에 있다.

### 1.3 서버 메모리

RTX 2080 Ti, FP32, batch 1, warmup 10회 뒤 forward 50회의 CUDA allocator 최대값(MiB).

| 실험 | 예산 M_max | Baseline peak | Reduced peak | Patch Slimming peak | Module 출력 합: reduced → patch |
|---|---:|---:|---:|---:|---|
| Tiny30 | 26.1523 | 32.8311 | 26.1523 | 27.3125 | 42.8932 → 19.3636 |
| Tiny50 | 21.8018 | 32.8311 | 21.8018 | 22.8862 | 33.9834 → 14.5954 |
| Small30 | 71.5552 | 97.1577 | 71.5552 | 74.0703 | 86.6843 → 46.8540 |
| Small50 | 54.3721 | 97.1577 | 54.3721 | 56.6802 | 69.4420 → 32.6188 |

학습 중 매 epoch 측정한 후보의 peak는 모두 예산과 같았고 메모리 penalty는 0이었다.

### 1.4 해석

- FFN reducing은 NPU에서 서버 FP32와 거의 같은 정확도를 유지하면서(0.3~1.4%p 차이) latency를 4~9%, NPU 메모리를 8~31% 줄였다.
- Patch Slimming은 NPU에서 baseline보다 22~28% 빠르고 메모리가 34~58% 적다. 대신 서버 FP32에서 reduced보다 1.7~4.3%p 낮고, MXQ로 변환한 뒤에는 서버 FP32보다 7.9~18.1%p 더 낮아진다.
- 서버 CUDA peak는 patch slimming이 reduced보다 조금 크다. 서버 메모리 수치로 NPU 메모리를 추정할 수 없다.
- 정확도를 우선하면 Small30 Reduced(NPU 79.324%)가 압축 모델 중 가장 높다.

### 1.5 같은 가중치를 공식 ImageNet 배포본으로 평가한 값

참고 실행의 가중치를 그대로 두고 validation 데이터만 공식 배포본(Hugging Face `ILSVRC/imagenet-1k`, 50,000장)으로 바꿔 서버 FP32 정확도를 다시 잰 값이다(RTX A5000, PyTorch 2.7.1). 데이터 사본에 따른 차이만 보여 준다.

| 실험 | 모델 | 참고 사본 Top-1 / Top-5 | 공식 배포본 Top-1 / Top-5 | Top-1 차이 |
|---|---|---|---|---:|
| Tiny30 | Baseline | 74.660 / 92.494 | 75.458 / 92.850 | +0.798 |
| Tiny30 | Reduced | 72.212 / 91.168 | 72.300 / 91.392 | +0.088 |
| Tiny30 | Patch Slimming | 67.902 / 88.130 | 67.590 / 88.126 | −0.312 |
| Tiny50 | Baseline | 74.660 / 92.494 | 75.458 / 92.850 | +0.798 |
| Tiny50 | Reduced | 63.230 / 85.832 | 63.440 / 86.022 | +0.210 |
| Tiny50 | Patch Slimming | 59.252 / 82.270 | 59.076 / 82.246 | −0.176 |
| Small30 | Baseline | 80.502 / 95.862 | 81.380 / 96.136 | +0.878 |
| Small30 | Reduced | 79.624 / 95.162 | 79.702 / 95.244 | +0.078 |
| Small30 | Patch Slimming | 77.970 / 94.190 | 77.324 / 93.910 | −0.646 |
| Small50 | Baseline | 80.502 / 95.862 | 81.380 / 96.136 | +0.878 |
| Small50 | Reduced | 74.002 / 92.234 | 74.082 / 92.278 | +0.080 |
| Small50 | Patch Slimming | 70.736 / 90.124 | 70.488 / 89.936 | −0.248 |

Pretrained baseline은 공식 배포본에서 0.8~0.9%p 높다(timm이 공개한 수치와 같다). 참고 사본으로 학습한 압축 모델은 차이가 그보다 작다. 공식 배포본으로 새로 실행한 결과(2절)는 이 표의 오른쪽 열과 비교한다.

## 2. 재현 실행 (공식 ImageNet, RTX A5000)

이 저장소의 코드와 `configs/final.yaml`로 예산 측정부터 MXQ 컴파일까지 처음부터 다시 실행한 결과다. 데이터는 공식 배포본(Hugging Face `ILSVRC/imagenet-1k`), 장비는 RTX A5000 4장(모델당 1장), PyTorch 2.7.1+cu128, qbcompiler 1.1.0이다. 이 실행의 ONNX와 MXQ는 [models/final](../models/final/README.md)에 있다. MXQ는 아직 NPU에서 평가하지 않았다. 아래는 모두 서버 값이다.

### 2.1 서버 FP32 정확도 (Top-1 / Top-5, %)

비교 대상은 같은 데이터로 평가한 참고 가중치(1.5절의 공식 배포본 열)다.

| 실험 | 모델 | 파라미터 | 재현 실행 | 참고 가중치 | Top-1 차이 |
|---|---|---:|---|---|---:|
| Tiny30 | Baseline | 5,717,416 | 75.458 / 92.850 | 75.458 / 92.850 | 0.000 |
| Tiny30 | Reduced | 4,003,396 | 72.468 / 91.330 | 72.300 / 91.392 | +0.168 |
| Tiny30 | Patch Slimming | 4,003,396 | 68.270 / 88.334 | 67.590 / 88.126 | +0.680 |
| Tiny50 | Baseline | 5,717,416 | 75.458 / 92.850 | 75.458 / 92.850 | 0.000 |
| Tiny50 | Reduced | 2,862,256 | 63.530 / 86.074 | 63.440 / 86.022 | +0.090 |
| Tiny50 | Patch Slimming | 2,862,256 | 60.202 / 82.888 | 59.076 / 82.246 | +1.126 |
| Small30 | Baseline | 22,050,664 | 81.380 / 96.136 | 81.380 / 96.136 | 0.000 |
| Small30 | Reduced | 15,434,188 | 79.964 / 95.326 | 79.702 / 95.244 | +0.262 |
| Small30 | Patch Slimming | 15,434,188 | 78.240 / 94.210 | 77.324 / 93.910 | +0.916 |
| Small50 | Baseline | 22,050,664 | 81.380 / 96.136 | 81.380 / 96.136 | 0.000 |
| Small50 | Reduced | 11,023,204 | 74.298 / 92.314 | 74.082 / 92.278 | +0.216 |
| Small50 | Patch Slimming | 11,023,204 | 70.936 / 90.176 | 70.488 / 89.936 | +0.448 |

Pruned의 정확도는 네 실험 모두 reduced와 같았다. Patch Slimming의 best checkpoint는 30 epoch 중 Tiny30 27번째, Tiny50 30번째, Small30 28번째, Small50 30번째 epoch였다.

### 2.2 구조

Block별 FFN 폭은 네 실험 모두 참고 실행과 같았다. Token 수는 Tiny30과 Small30이 참고 실행과 같고 Tiny50과 Small50이 달랐다.

| 실험 | Block 0~11의 출력 token 수 | 참고 실행 |
|---|---|---|
| Tiny30 | 121 ×5, 111 ×6, 1 | 같음 |
| Tiny50 | 121 ×11, 1 | 121 ×5, 111 ×6, 1 |
| Small30 | 197 ×3, 141, 111 ×7, 1 | 같음 |
| Small50 | 171, 161, 141, 121, 101 ×7, 1 | 171, 151, 131, 121, 111 ×7, 1 |

### 2.3 서버 메모리

RTX A5000, FP32, batch 1, warmup 10회 뒤 forward 50회의 CUDA allocator 최대값(MiB).

| 실험 | 예산 M_max | Baseline peak | Reduced peak | Patch Slimming peak | Module 출력 합: reduced → patch |
|---|---:|---:|---:|---:|---|
| Tiny30 | 26.1523 | 32.8311 | 26.1523 | 27.3125 | 42.8932 → 19.3636 |
| Tiny50 | 21.8018 | 32.8311 | 21.8018 | 22.9351 | 33.9834 → 15.2964 |
| Small30 | 71.5552 | 97.1577 | 71.5552 | 74.0703 | 86.6843 → 46.8540 |
| Small50 | 54.3721 | 97.1577 | 54.3721 | 56.6489 | 69.4420 → 31.3360 |

예산과 baseline·reduced peak는 참고 실행(RTX 2080 Ti)과 같다. 학습 중 매 epoch 측정한 후보의 peak는 30회 모두 예산과 같았고 메모리 penalty는 0이었다.

### 2.4 ONNX와 MXQ

| 실험 | 모델 | ONNX MiB | PyTorch↔ORT 최대 오차 | MXQ MiB | 컴파일 초 |
|---|---|---:|---:|---:|---:|
| Tiny30 | Baseline | 21.89 | 1.14e-05 | 7.56 | 154 |
| Tiny30 | Reduced | 15.35 | 8.29e-06 | 6.01 | 158 |
| Tiny30 | Patch Slimming | 15.62 | 6.85e-06 | 5.74 | 144 |
| Tiny50 | Baseline | 21.89 | 1.14e-05 | 7.56 | 141 |
| Tiny50 | Reduced | 11.00 | 7.75e-06 | 4.86 | 139 |
| Tiny50 | Patch Slimming | 11.17 | 7.09e-06 | 4.58 | 122 |
| Small30 | Baseline | 84.20 | 6.20e-06 | 24.48 | 129 |
| Small30 | Reduced | 58.96 | 4.29e-06 | 18.31 | 130 |
| Small30 | Patch Slimming | 59.27 | 3.47e-06 | 18.15 | 121 |
| Small50 | Baseline | 84.20 | 6.20e-06 | 24.48 | 139 |
| Small50 | Reduced | 42.13 | 3.46e-06 | 13.90 | 133 |
| Small50 | Patch Slimming | 42.98 | 4.53e-06 | 13.83 | 129 |

12개 ONNX 모두 금지 연산이 없고 입력 `[1, 224, 224, 3]`, 출력 `[1, 1000]`이다. Calibration은 공식 train에서 class당 1장을 seed 2022로 뽑은 1,000장을 썼다.

### 2.5 소요 시간 (시간)

네 모델을 GPU 한 장씩에 동시에 실행했고(CPU 64 core 공유, data loader worker 프로세스당 16개) 시작부터 끝까지 22.8시간 걸렸다.

| 실험 | 예산 | 학습 + reducing | Patch Slimming | ONNX | 메모리 | FP32 평가 | 컴파일 | 합계 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny30 | 0.01 | 6.12 | 4.15 | 0.01 | 0.01 | 0.05 | 0.13 | 10.47 |
| Tiny50 | 0.01 | 6.39 | 4.18 | 0.01 | 0.01 | 0.04 | 0.11 | 10.74 |
| Small30 | 0.01 | 13.04 | 9.57 | 0.01 | 0.01 | 0.06 | 0.11 | 22.80 |
| Small50 | 0.01 | 13.01 | 7.51 | 0.01 | 0.01 | 0.06 | 0.11 | 20.71 |

Patch Slimming 시간의 대부분은 30 epoch fine-tuning이고, 탐색은 모델당 5~12분(후보 23~31개)이다.

## 3. 다시 실행했을 때의 기대 범위

같은 설정으로 처음부터 다시 학습하면 수치가 참고값과 조금 다르다. 아래 범위 안이면 정상이다. 2절의 재현 실행도 이 범위 안이었다.

| 항목 | 기대 |
|---|---|
| 목표 구조 파라미터 수, baseline 가중치 | 정확히 같다 (pretrained 가중치로 정해진다) |
| Baseline 정확도 | 같은 ImageNet 사본이면 같다. 데이터가 다르면 달라진다 |
| Block별 FFN 폭 | 같거나 거의 같다 (재현 실행에서는 모두 같았다) |
| Reduced 정확도 | 소수점 첫째 자리 수준의 차이 (재현 실행 +0.09~+0.26%p) |
| Patch Slimming token 구조 | 달라질 수 있다 (오차 0.02 경계에서 10개 단위로 정해진다) |
| Patch Slimming 정확도 | 1%p 안팎 차이 가능 (재현 실행 +0.45~+1.13%p) |
| 서버 메모리 수치 | GPU·CUDA에 따라 달라질 수 있다 |
| NPU baseline 정확도 | 참고값과 거의 같다 |
| NPU reduced 정확도 | 서버 FP32보다 0.3~1.4%p 낮은 수준 |
| NPU patch slimming 정확도 | 서버 FP32보다 크게 낮다 (참고 실행 7.9~18.1%p) |

Reduced 정확도가 참고값과 2%p 이상 다르면 설정이 달라졌을 가능성이 크다. [METHODOLOGY.md](METHODOLOGY.md)의 고정 설정부터 대조한다.
