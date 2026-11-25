# 최종 NPU 벤치마크 결과

## 평가 개요

| 항목 | 결과 |
|---|---|
| 최종 상태 | **PASS** (`complete`) |
| 평가 모델 | 12개 중 12개 정상, 오류 이미지 0장 |
| 데이터셋 | ImageNet validation 50000장/모델 (`/media/airlab_compression/nvme_storage/ETRI_Compression_y3/data/imagenet/val`) |
| 총 추론 수 | 600000회 |
| 최종 모델 폴더 | `/media/airlab_compression/nvme_storage/ETRI_Compression_y3/models/final/mxq` |
| 전처리 / 코어 | `opencv_bicubic249_crop224` / `global8`, device 0 |
| Warmup | 모델당 10회 |
| NPU telemetry | 12/12 모델 정상 수집 |
| 평가 시간 합계 | 2.14시간 |
| 실행 시작 시각 / 시작 HEAD | `2026-11-23T16:14:10.169167+09:00` / `9004410a86ee` |
| 실행 시 worktree | `clean` |
| Runtime / CLI | qbruntime `v1.1.0` / mobilint-cli `v1.2.0` |
| Aries driver / firmware | `1.11.0` / `1.2.5` |
| 샘플 SHA-256 | `f89c7708072a82afc870ddc899b340a97188227737dbf7ebd9bf20484db5ddfd` |

## 핵심 비교

압축 모델 중 정확도가 가장 높은 모델, 가장 빠른 모델, NPU 메모리가 가장 적은 모델을 그룹별로 요약했다.

| Group | Baseline Top-1 | 최고 압축 정확도 | 가장 빠른 압축 모델 | 최저 NPU 메모리 |
|---|---:|---|---|---|
| tiny30 | 74.126% | Reduced: 71.812% (-2.314 pp) | Patch Slimming: 2.320 ms (1.30x) | Patch Slimming: 7.0/7.0 MB (41.7%) |
| tiny50 | 74.126% | Reduced: 62.856% (-11.270 pp) | Patch Slimming: 2.329 ms (1.29x) | Patch Slimming: 5.0/5.0 MB (58.3%) |
| small30 | 81.356% | Reduced: 79.520% (-1.836 pp) | Patch Slimming: 4.474 ms (1.28x) | Patch Slimming: 23.0/23.0 MB (34.3%) |
| small50 | 81.356% | Reduced: 73.578% (-7.778 pp) | Patch Slimming: 3.985 ms (1.44x) | Patch Slimming: 18.0/18.0 MB (48.6%) |

## 결과 해석

- **tiny30**: Reduced는 Top-1 `71.812%` (`-2.314 pp`), latency `4.4%` 감소, NPU 메모리 `8.3%` 감소. Patch Slimming은 Top-1 `52.476%` (`-21.650 pp`), latency `23.2%` 감소, NPU 메모리 `41.7%` 감소.
- **tiny50**: Reduced는 Top-1 `62.856%` (`-11.270 pp`), latency `6.4%` 감소, NPU 메모리 `16.7%` 감소. Patch Slimming은 Top-1 `34.568%` (`-39.558 pp`), latency `22.4%` 감소, NPU 메모리 `58.3%` 감소.
- **small30**: Reduced는 Top-1 `79.520%` (`-1.836 pp`), latency `4.9%` 감소, NPU 메모리 `20.0%` 감소. Patch Slimming은 Top-1 `69.626%` (`-11.730 pp`), latency `21.7%` 감소, NPU 메모리 `34.3%` 감소.
- **small50**: Reduced는 Top-1 `73.578%` (`-7.778 pp`), latency `8.9%` 감소, NPU 메모리 `31.4%` 감소. Patch Slimming은 Top-1 `61.750%` (`-19.606 pp`), latency `30.8%` 감소, NPU 메모리 `48.6%` 감소.
- Reduced 구성 중 baseline 대비 Top-1 손실이 가장 작은 구성은 **small30 Reduced**로, Top-1 `79.520%`와 baseline 대비 `-1.836 pp`를 기록했다.
- 실행 상태 PASS는 정확도 보존을 뜻하지 않는다. 정확도·지연시간·메모리를 함께 비교하며 독립 메모리 재검증은 해당 결과가 있을 때만 판단한다.

## 정확도 및 모델 압축

| Group | Model | 상태 | Top-1 | Δ Top-1 | Top-5 | 예측 일치율 | Params M | Params 감소 | GFLOPs* | FLOPs 감소 | MXQ MB | MXQ 감소 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny30 | Baseline | ok | 74.126% | 0.000 pp | 92.378% | 100.000% | 5.717 | 0.0% | 2.507 | 0.0% | 7.6 | 0.0% |
| tiny30 | Patch Slimming | ok | 52.476% | -21.650 pp | 77.482% | 56.902% | 4.003 | 30.0% | 0.990 | 60.5% | 5.7 | 24.0% |
| tiny30 | Reduced | ok | 71.812% | -2.314 pp | 91.090% | 81.268% | 4.003 | 30.0% | 1.834 | 26.9% | 6.0 | 20.6% |
| tiny50 | Baseline | ok | 74.126% | 0.000 pp | 92.378% | 100.000% | 5.717 | 0.0% | 2.507 | 0.0% | 7.6 | 0.0% |
| tiny50 | Patch Slimming | ok | 34.568% | -39.558 pp | 59.082% | 37.162% | 2.862 | 49.9% | 0.773 | 69.2% | 4.6 | 39.5% |
| tiny50 | Reduced | ok | 62.856% | -11.270 pp | 85.492% | 69.882% | 2.862 | 49.9% | 1.385 | 44.7% | 4.9 | 35.7% |
| small30 | Baseline | ok | 81.356% | 0.000 pp | 96.108% | 100.000% | 22.051 | 0.0% | 9.198 | 0.0% | 24.5 | 0.0% |
| small30 | Patch Slimming | ok | 69.626% | -11.730 pp | 89.946% | 74.860% | 15.434 | 30.0% | 4.245 | 53.9% | 18.1 | 25.9% |
| small30 | Reduced | ok | 79.520% | -1.836 pp | 95.152% | 88.102% | 15.434 | 30.0% | 6.594 | 28.3% | 18.3 | 25.2% |
| small50 | Baseline | ok | 81.356% | 0.000 pp | 96.108% | 100.000% | 22.051 | 0.0% | 9.198 | 0.0% | 24.5 | 0.0% |
| small50 | Patch Slimming | ok | 61.750% | -19.606 pp | 84.590% | 66.342% | 11.023 | 50.0% | 2.897 | 68.5% | 13.8 | 43.5% |
| small50 | Reduced | ok | 73.578% | -7.778 pp | 92.064% | 80.526% | 11.023 | 50.0% | 4.859 | 47.2% | 13.9 | 43.2% |

## 추론 성능

`Inference`는 `model.infer()`만, `E2E`는 이미지 로딩과 전처리를 포함한다.

| Group | Model | 평균 ms | P50 | P95 | P99 | Inference FPS | Baseline 대비 | E2E FPS | 전처리 평균 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny30 | Baseline | 3.020 | 3.004 | 3.148 | 3.285 | 331.12 | 1.00x | 91.68 | 7.759 |
| tiny30 | Patch Slimming | 2.320 | 2.305 | 2.430 | 2.559 | 431.10 | 1.30x | 101.68 | 7.391 |
| tiny30 | Reduced | 2.886 | 2.874 | 3.001 | 3.131 | 346.45 | 1.05x | 94.27 | 7.598 |
| tiny50 | Baseline | 3.002 | 2.987 | 3.125 | 3.264 | 333.06 | 1.00x | 92.20 | 7.718 |
| tiny50 | Patch Slimming | 2.329 | 2.315 | 2.440 | 2.573 | 429.40 | 1.29x | 101.79 | 7.376 |
| tiny50 | Reduced | 2.811 | 2.796 | 2.930 | 3.075 | 355.71 | 1.07x | 95.47 | 7.541 |
| small30 | Baseline | 5.714 | 5.695 | 5.909 | 6.148 | 175.01 | 1.00x | 60.77 | 10.573 |
| small30 | Patch Slimming | 4.474 | 4.455 | 4.644 | 4.840 | 223.50 | 1.28x | 70.73 | 9.508 |
| small30 | Reduced | 5.435 | 5.415 | 5.629 | 5.869 | 183.98 | 1.05x | 63.17 | 10.230 |
| small50 | Baseline | 5.755 | 5.737 | 5.954 | 6.177 | 173.75 | 1.00x | 60.55 | 10.590 |
| small50 | Patch Slimming | 3.985 | 3.966 | 4.144 | 4.336 | 250.94 | 1.44x | 76.39 | 8.961 |
| small50 | Reduced | 5.245 | 5.224 | 5.435 | 5.670 | 190.64 | 1.10x | 64.04 | 10.206 |

## NPU 메모리 및 사용률

| Group | Model | 메모리 평균 MB | P99 MB | 피크 MB | 피크 감소 | 사용률 평균 | P99 | 최대 | Samples |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny30 | Baseline | 12.0 | 12.0 | 12.0 | 0.0% | 22.2% | 24.2% | 25.3% | 1092 |
| tiny30 | Patch Slimming | 7.0 | 7.0 | 7.0 | 41.7% | 17.7% | 19.8% | 21.4% | 985 |
| tiny30 | Reduced | 11.0 | 11.0 | 11.0 | 8.3% | 21.8% | 23.6% | 24.3% | 1062 |
| tiny50 | Baseline | 12.0 | 12.0 | 12.0 | 0.0% | 22.4% | 24.2% | 25.2% | 1086 |
| tiny50 | Patch Slimming | 5.0 | 5.0 | 5.0 | 58.3% | 17.8% | 20.0% | 21.7% | 984 |
| tiny50 | Reduced | 10.0 | 10.0 | 10.0 | 16.7% | 21.4% | 23.2% | 24.4% | 1049 |
| small30 | Baseline | 35.0 | 35.0 | 35.0 | 0.0% | 30.2% | 34.8% | 37.3% | 1647 |
| small30 | Patch Slimming | 23.0 | 23.0 | 23.0 | 34.3% | 26.8% | 30.0% | 31.4% | 1415 |
| small30 | Reduced | 28.0 | 28.0 | 28.0 | 20.0% | 29.8% | 34.0% | 35.8% | 1584 |
| small50 | Baseline | 35.0 | 35.0 | 35.0 | 0.0% | 30.1% | 34.8% | 37.5% | 1653 |
| small50 | Patch Slimming | 18.0 | 18.0 | 18.0 | 48.6% | 25.5% | 28.5% | 30.8% | 1311 |
| small50 | Reduced | 24.0 | 24.0 | 24.0 | 31.4% | 28.9% | 33.2% | 34.4% | 1563 |

## NPU 전력 및 온도

Total power는 보드 전체, NPU core power는 NPU 코어 전력이다.

| Group | Model | Total W 평균 | P99 | 최대 | NPU core W 평균 | 최대 | 온도 °C 평균 | P99 | 최대 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny30 | Baseline | 16.75 | 17.01 | 17.07 | 6.94 | 7.04 | 43.2 | 46.0 | 47.0 |
| tiny30 | Patch Slimming | 16.50 | 16.59 | 16.82 | 6.76 | 6.96 | 41.9 | 43.0 | 43.0 |
| tiny30 | Reduced | 16.68 | 16.78 | 16.82 | 6.88 | 6.96 | 41.9 | 43.0 | 43.0 |
| tiny50 | Baseline | 16.74 | 16.85 | 16.90 | 6.92 | 7.01 | 41.7 | 42.0 | 42.0 |
| tiny50 | Patch Slimming | 16.47 | 16.55 | 16.66 | 6.74 | 6.81 | 41.5 | 42.0 | 42.0 |
| tiny50 | Reduced | 16.64 | 16.73 | 16.78 | 6.85 | 6.93 | 41.6 | 42.0 | 42.0 |
| small30 | Baseline | 17.21 | 17.47 | 17.61 | 7.26 | 7.45 | 42.3 | 43.0 | 43.0 |
| small30 | Patch Slimming | 16.83 | 16.99 | 17.14 | 6.94 | 7.23 | 41.8 | 42.0 | 43.0 |
| small30 | Reduced | 16.99 | 17.18 | 17.34 | 7.09 | 7.25 | 42.0 | 43.0 | 43.0 |
| small50 | Baseline | 17.20 | 17.43 | 17.63 | 7.25 | 7.45 | 42.2 | 43.0 | 43.0 |
| small50 | Patch Slimming | 16.75 | 16.91 | 17.31 | 6.88 | 7.24 | 41.6 | 42.0 | 43.0 |
| small50 | Reduced | 16.89 | 17.08 | 17.20 | 7.01 | 7.19 | 41.9 | 43.0 | 43.0 |

## 호스트 프로세스 메모리

| Group | Model | RSS 시작 MB | RSS 피크 MB | RSS 증가 MB | USS 피크 MB | PSS 피크 MB |
|---|---|---:|---:|---:|---:|---:|
| tiny30 | Baseline | 244.9 | 390.5 | 145.6 | 346.0 | 347.1 |
| tiny30 | Patch Slimming | 289.1 | 348.0 | 58.9 | 346.6 | 347.7 |
| tiny30 | Reduced | 290.0 | 362.8 | 72.8 | 354.4 | 355.5 |
| tiny50 | Baseline | 290.2 | 391.0 | 100.8 | 381.2 | 382.3 |
| tiny50 | Patch Slimming | 290.6 | 383.5 | 92.9 | 376.7 | 377.8 |
| tiny50 | Reduced | 290.9 | 395.3 | 104.4 | 382.7 | 383.8 |
| small30 | Baseline | 290.7 | 343.2 | 52.5 | 332.0 | 333.1 |
| small30 | Patch Slimming | 290.9 | 375.3 | 84.4 | 374.8 | 375.9 |
| small30 | Reduced | 291.0 | 343.1 | 52.1 | 330.6 | 331.7 |
| small50 | Baseline | 291.1 | 351.7 | 60.6 | 345.3 | 346.4 |
| small50 | Patch Slimming | 291.4 | 368.0 | 76.6 | 361.5 | 362.5 |
| small50 | Reduced | 291.1 | 394.2 | 103.0 | 383.2 | 384.3 |

## 아티팩트 무결성

| Group | Model | MXQ bytes | MXQ SHA-256 | ONNX bytes | ONNX SHA-256 |
|---|---|---:|---|---:|---|
| tiny30 | Baseline | 7929396 | `5cffb4c9ea359d79a8e7f1a2ed6eeb6233bc6ce747a643e91f5089b59225135c` | 22952270 | `07393963039c1631caadcb315a8b31cfc104af8d4665609dd3b8df04f6ff694f` |
| tiny30 | Patch Slimming | 6022812 | `2cb455ebe5ac752504857568244f235d99c2f96a0540253d4189e8edc2120494` | 16379897 | `5895cc864d1529b3854d8ab2769e29ef9130a211627a7f20e4075dc4ac7889de` |
| tiny30 | Reduced | 6297268 | `432d34b8adf8f71349bf7f018b857f4f001bd084fcce20467b7c2ed4ec1141ac` | 16096190 | `24172da88a57ae9450ea714f23b6ac2b477a238e26b55b9d92e13b7ff97a343b` |
| tiny50 | Baseline | 7929396 | `ef974178f887e081f8a92c866bf3f95db53d57b1a64e8dd7a1ff13c432495c39` | 22952270 | `07393963039c1631caadcb315a8b31cfc104af8d4665609dd3b8df04f6ff694f` |
| tiny50 | Patch Slimming | 4799184 | `75e6919ec03f020f96d9b88cbaae3d04f73405f88df4afc471554c86b6386314` | 11707467 | `ae64eddb2eff9d2aa92511ab85c321d167f1159ef84b7de7493534362ae1bc1e` |
| tiny50 | Reduced | 5100498 | `cc4e31b4e67a1875c48f63257e917557d49828b3c6348eacb3342315d1c94449` | 11531612 | `be79c40169156db088eb2fd3cccb985a0078189e99377e173c02403e5258df4c` |
| small30 | Baseline | 25669056 | `a85fd72624ceeb4a5cf7a769e7298fb1578f15f57fc2d72bfb0f81d26fe6e8a3` | 88285310 | `a392b1cb3fc8e59364928b9aaa0f906fc67f447ac43aca49e6ac83e008846778` |
| small30 | Patch Slimming | 19027674 | `476e776bde710c9da01e099bb3d8436bd7aca0d3bbf1e4855a4350eaf06de388` | 62152349 | `049fc760690830c0103672d703576086e5ed0b06ea9859dd30f0db889b2d501e` |
| small30 | Reduced | 19197292 | `e8aac483b15dbd78f1894e01e1d39865357c8ab81411d32f88ef84786d756503` | 61819358 | `22a2ff0d6c48f5280a9c55317b94cce71604f0dd4ef2faf5cea98889e667a433` |
| small50 | Baseline | 25669056 | `7e01d49c957c132dd0fd73022ab403fd23fae3656d5df73d5c46d617a34726ad` | 88285310 | `a392b1cb3fc8e59364928b9aaa0f906fc67f447ac43aca49e6ac83e008846778` |
| small50 | Patch Slimming | 14501761 | `83314af0c8b75a857f0bd67794fcefaa9651baa6165b119a610da674bdcfa13e` | 45067817 | `30b274c76d7cd543c9a045ce3f04dbaa6a42fa214166a11f6ab81fb98715ed3e` |
| small50 | Reduced | 14579718 | `2c3d00e5700caea5f9c2a4e92fb5675e507a260d1b2f1c136a879134f5f91946` | 44175422 | `bb0179179540c922b2e9b93ab5988dfa07da115dad7665853ed5dff4e58d190e` |

## 독립 프로세스 메모리 재검증

각 모델을 새 프로세스에서 고정 random FP32 입력으로 실행했다. Idle/Active는 장치 전체에서 `mobilint-cli`가 보고한 MB이며, host 값은 프로세스 MiB이다.

| Group | Model | 상태 | Idle MB | Active peak MB | Δ NPU MB | Host RSS peak MiB | Host USS peak MiB | 평균 ms | Telemetry runs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| tiny30 | Baseline | ok | 0.0 | 12.0 | 12.0 | 141.8 | 122.2 | 3.158 | 3194 |
| tiny30 | Reduced | ok | 0.0 | 11.0 | 11.0 | 141.1 | 122.2 | 3.000 | 3352 |
| tiny30 | Patch Slimming | ok | 0.0 | 7.0 | 7.0 | 137.8 | 118.4 | 2.454 | 4123 |
| tiny50 | Baseline | ok | 0.0 | 12.0 | 12.0 | 136.4 | 116.8 | 3.104 | 3234 |
| tiny50 | Reduced | ok | 0.0 | 10.0 | 10.0 | 141.1 | 122.2 | 2.918 | 3403 |
| tiny50 | Patch Slimming | ok | 0.0 | 5.0 | 5.0 | 138.6 | 119.6 | 2.474 | 4050 |
| small30 | Baseline | ok | 0.0 | 35.0 | 35.0 | 142.0 | 122.8 | 5.736 | 1753 |
| small30 | Reduced | ok | 0.0 | 28.0 | 28.0 | 142.4 | 122.8 | 5.477 | 1840 |
| small30 | Patch Slimming | ok | 0.0 | 23.0 | 23.0 | 138.8 | 119.9 | 4.509 | 2209 |
| small50 | Baseline | ok | 0.0 | 35.0 | 35.0 | 139.4 | 119.8 | 5.729 | 1746 |
| small50 | Reduced | ok | 0.0 | 24.0 | 24.0 | 144.3 | 124.7 | 5.197 | 1908 |
| small50 | Patch Slimming | ok | 0.0 | 18.0 | 18.0 | 139.3 | 120.0 | 4.116 | 2496 |

통합 평가와 비교 가능한 독립 측정 12개 중 peak 일치 12개, idle 0 MB 12개. 이 비교는 관측 메모리 값의 일치이며 내부 activation peak의 보장이 아니다.

## 측정 기준

- `GFLOPs*`는 ONNX Conv/MatMul/Gemm 정적 shape로 계산했으며 `1 MAC = 2 FLOPs`를 적용했다. Elementwise, normalization, activation, softmax와 컴파일러 fusion/tiling은 포함하지 않는다.
- NPU 메모리·전력·사용률·온도는 `mblt_tracker`가 `mobilint-cli status`를 주기적으로 관측한 값이다. 메모리 피크는 관측 구간의 최대 할당량이며 연산 중 초단기 내부 activation peak와는 다르다.
- 정확도 비교는 동일한 이미지 목록, label, `opencv_bicubic249_crop224` 전처리, `global8` 코어 조건을 사용했다.
- 상세 원본 지표와 오류 정보는 같은 폴더의 `benchmark.json`, 분석용 평탄화 지표는 `summary.csv`에 저장된다.
