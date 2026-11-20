# 방법론

이 문서는 각 단계가 무엇을 하는지와, 결과에 직접 영향을 주어 바꾸면 안 되는 설정을 적는다. 설정 값의 원본은 `configs/final.yaml`이다.

## 1. 대상과 전체 흐름

대상은 timm `vit_tiny_patch16_224`, `vit_small_patch16_224`(pretrained tag `augreg_in21k_ft_in1k`)이다. RGB 224×224 입력을 16×16 patch 196개와 CLS 1개, 모두 197개 token으로 처리한다. 두 모델 모두 12 block이고 embedding 차원은 Tiny 192, Small 384다. 실험은 전체 파라미터 30%, 50% 감소를 목표로 한 4개다.

```text
Pretrained baseline 저장, FP32 추론 메모리 측정
  → 사본을 목표 비율로 pruning·reducing (추가 학습 0 step) → 추론 peak = M_max
  → baseline에서 global L2 FFN pruning + logit KD + 메모리 항, 30 epoch
  → 30 epoch 가중치를 물리적으로 reducing                      [reduced]
  → 정적 token 구조 탐색 (block별 복원 학습 3 epoch)
  → token이 줄어든 compact 모델을 CE로 30 epoch fine-tuning       [patch slimming]
  → baseline / reduced / patch slimming ONNX export와 검증
  → 서버 FP32 정확도·메모리 평가
  → calibration 1,000장으로 MXQ 컴파일
  → NPU 장치에서 정확도·latency·메모리·전력 측정
```

## 2. 메모리 예산 M_max

코드: `compression/prepare_budget.py`

Pretrained baseline을 `00_budget/baseline.pt`로 저장한다. 그 사본을 목표 비율로 global L2 pruning하고 채널을 물리적으로 제거해 `target_untrained.pt`를 만든다. 이 과정에는 학습이 없다(`trained_steps = 0`). 이 구조의 FP32 추론 peak가 예산이다.

```text
M_max = target 구조의 CUDA peak allocated (MiB)
측정: FP32, batch 1, 224×224 random 입력, warmup 10회 뒤 forward 50회의 최대값
```

예산용 target은 최종 모델이 아니고 목표 구조의 메모리를 미리 재는 용도다. `budget.json`에 baseline·target의 경로와 SHA-256, 측정 조건, M_max가 저장되고 학습 단계가 이를 검증한다. 한번 만든 예산은 다시 계산하지 않는다.

목표 구조는 pretrained 가중치의 L2 norm으로 정해지므로 환경이 달라도 파라미터 수가 같다: Tiny30 4,003,396 / Tiny50 2,862,256 / Small30 15,434,188 / Small50 11,023,204. 메모리 측정값 자체는 GPU와 CUDA 버전에 따라 달라질 수 있다.

## 3. Global L2 FFN pruning

코드: `compression/vit_pruning.py`

FFN은 `D → M → D` 구조다. Hidden channel j의 점수는 첫 Linear의 행 norm `‖W_fc1[j,:]‖₂`이고, 모든 block의 점수를 함께 비교해 낮은 채널부터 제거한다. 한 채널을 제거하면 FC1의 행과 bias, FC2의 대응 열이 함께 없어진다(채널당 `2D+1`개). Attention, embedding 차원, classifier는 그대로 둔다.

- 목표 30% / 50%는 FFN만이 아니라 **전체 모델 파라미터 수** 대비 비율이다.
- Block별 최대 sparsity 0.95, 최소 생존 채널 4.
- 학습 중에는 채널을 0으로 마스킹한 dense 모델(**pruned**)을 유지한다. Optimizer update마다 mask를 다시 적용하고, 어떤 채널을 자를지의 순위는 100 step마다 갱신한다.

## 4. KD 학습과 메모리 항

코드: `compression/train.py`, `compression/engine.py`, `compression/memory_budget.py`

Teacher는 같은 구조의 pretrained baseline(고정)이고 logit KL distillation을 쓴다.

```text
L_task = (1−α)·CE(z_s, y) + α·T²·KL(softmax(z_t/T) ‖ softmax(z_s/T))
L_mem  = λ·max(0, M_candidate − M_max)
L      = L_task + L_mem
```

| 설정 | 값 |
|---|---|
| 학습 종료 | 30 epoch |
| LR schedule 기준 길이 | 50 epoch (warmup 뒤 cosine의 T_max = 45) |
| Optimizer | AdamW, lr 5e-5, weight decay 0.05 |
| Warmup / 최소 lr | 5 epoch / 1e-6 |
| Grad clip / label smoothing | 1.0 / 0.1 |
| AMP / EMA | 사용 / decay 0.9998 |
| KD | α 0.5, T 4.0 |
| 메모리 항 | λ 1.0, 매 epoch 측정 |
| Batch | Tiny 384, Small 192 |
| Seed | 2022 |
| 증강 | RandAugment `rand-m9-mstd0.5-inc1`, color jitter 0.4, random erasing 0.25 |

- **학습 종료 30 epoch과 LR schedule 길이 50 epoch은 서로 다른 설정이다.** 30 epoch에 lr이 최소값까지 내려가는 recipe가 아니다.
- **Batch 크기는 방법의 일부다.** GPU 메모리가 커도 키우지 않는다. Update 횟수가 달라져 결과가 바뀐다.
- 메모리 항: 매 epoch 현재 모델의 사본(raw mask를 EMA에 이식한 뒤 reducing)을 별도 프로세스에서 2절과 같은 조건으로 측정하고, 예산 초과분을 다음 batch부터 loss에 더한다. 측정값에서 온 상수이므로 **가중치에 대한 gradient가 없고**, 예산을 강제로 만족시키는 기능도 아니다. 참고 실행에서는 초과가 없어 항상 0이었다.
- 후속 단계는 best가 아니라 **정확히 30 epoch이 끝난 checkpoint**(`checkpoint_final.pt`)를 쓴다.

## 5. Reducing

코드: `compression/reduce.py`, `compression/vit_reducing.py`

Raw 모델에서 FC1 행 norm이 정확히 0인 채널을 찾아, EMA 가중치에 그 zero 패턴을 이식한 뒤 채널을 물리적으로 제거한다. 살아남은 채널의 값은 EMA 가중치다. 결과 `reduced.pt`에는 block별 FFN 폭(`mlp_dims`)이 함께 저장된다. Pruned와 reduced는 같은 함수이므로 FP32 정확도가 같다.

## 6. Patch Slimming

코드: `patch_slimming/psvit/`, `patch_slimming/run_search.py`, `patch_slimming/run_finetune.py`
참고 논문: Tang et al., *Patch Slimming for Efficient Vision Transformers*, CVPR 2022.

Reduced 모델에서 block마다 남길 token ID를 **고정**으로 정한다. 입력마다 token을 고르는 동적 방식이 아니다.

**데이터.** ImageNet train에서 서로 겹치지 않는 부분집합 3개를 쓴다.

| 용도 | 장수 | seed |
|---|---:|---:|
| 중요도 점수 | 2,048 | 2022 |
| Block 복원 학습 | 4,096 | 2023 |
| 후보 판정 | 2,048 | 2024 |

**중요도.** 현재 block의 attention·feature energy와 뒤쪽 block들의 attention 경로 영향을 결합한 점수를 이미지별로 구해 평균한다. Head 경로 집계는 `G ← mean_h(P_hᵀ G P_h)`의 역방향 동적 계획법으로 계산한다.

**탐색.** 마지막 block의 출력은 CLS 하나로 두고 뒤에서 앞으로(block 10 → 0) 진행한다. 각 block은 다음 block이 쓰는 token 집합에서 시작해 중요도가 높은 patch를 10개씩 추가한다. 후보마다 그 block만 3 epoch 학습(AdamW lr 1e-5, MSE)한 뒤 판정용 부분집합에서 다음 block 출력의 오차를 잰다.

```text
E = mean_b( ‖pred_b − target_b‖_F² / max(‖target_b‖_F², 1e-12) )     채택 조건: E ≤ 0.02
```

Teacher는 Patch Slimming 전의 reduced 모델이다. CLS는 항상 남기고, 뒤 block의 token 집합은 앞 block의 집합에 포함된다. 채택 지표 이름은 `sample_relative_frobenius_sq_mean`이다.

**Compact 모델.** Block마다 key/value는 입력 token 전체에서, query와 FFN은 남길 token에서만 계산한다. Token 텐서가 실제로 `[B, N_in, D] → [B, N_out, D]`로 작아진다.

**Fine-tuning.** 고정 token 구조의 compact 모델을 CE로 30 epoch 학습한다(AdamW lr 2e-5, weight decay 0.05, warmup 3, label smoothing 0.1, AMP, KD 없음). Batch는 Tiny30 512, Tiny50 640, Small 256이다. 학습 중 validation Top-1이 가장 높은 checkpoint(`compact_finetuned_best.pt`)를 쓴다.

Token 구조는 오차 0.02 경계에서 10개 단위로 정해지므로, 다시 학습하면 구조가 조금 달라질 수 있다.

## 7. ONNX 계약

코드: `export/export_onnx.py`

| 항목 | 값 |
|---|---|
| 입력 | `input`, float32, NHWC `[1, 224, 224, 3]` |
| 출력 | `output`, `[1, 1000]` logits |
| Batch / opset | 1 고정 / 17 |
| 정규화 | 그래프 밖에서 수행 (mean = std = 0.5) |
| 파일 이름 | `<실험>__<baseline\|reduced\|patch_slimming>__nhwc_b1_npusafe.onnx` |

- Token 선택은 고정 0/1 행렬 `P[N_out, N_in]`의 MatMul로 표현한다. 모든 token을 그대로 두는 선택(identity)은 그래프에 넣지 않고, CLS만 남기는 선택은 static Slice/Squeeze로 처리한다.
- `Gather`, `GatherElements`, `GatherND`, `TopK`, `Equal`, `Where`, `Expand`가 그래프에 있으면 export는 실패한다.
- Random 입력 3개에서 PyTorch와 ONNX Runtime 출력의 최대 절대오차가 1e-4 미만이어야 하고, 통과하면 `<이름>.verified.json`을 쓴다. 이 파일이 있어야 다음 단계로 넘어간다.
- 이름의 `npusafe`는 이 고정 그래프 규칙을 뜻한다. 컴파일 성공이나 NPU 정확도를 보증하는 뜻이 아니다. ONNX 검증은 FP32 수치 일치만 확인한다.

## 8. 전처리

구현이 둘이고, 통일하지 않는다.

| 사용처 | 구현 | 내용 |
|---|---|---|
| 서버 학습·FP32 평가 | timm / PIL | bicubic, `crop_pct 0.9`, center crop 224, mean = std = 0.5 |
| Calibration, NPU 평가 | OpenCV (`common/preprocess.py`) | `imread` → RGB → 짧은 변 249 → `INTER_CUBIC` → center crop 224 → `/255` → `(x−0.5)/0.5` → HWC float32 |

두 구현은 같은 설정을 뜻하지만 픽셀 단위로 같지는 않다. NPU 호스트에는 학습용 패키지를 설치하지 않으므로 장치 쪽은 OpenCV 구현을 쓴다.

**Calibration과 NPU 평가의 전처리는 완전히 같아야 한다.** 두 쪽 모두 `common.preprocess.preprocess` 한 함수를 쓰고, 테스트(`tests/test_device_preprocess.py`)가 이를 확인한다.

## 9. Calibration과 MXQ 컴파일

코드: `compile/calibration.py`, `compile/compile_mxq.py`

ImageNet train에서 뽑은 1,000장을 8절의 OpenCV 전처리로 HWC float32 `.npy`로 저장한다([DATASET.md](DATASET.md)). 컴파일러 호출은 아래 하나뿐이며 다른 양자화 옵션을 넘기지 않는다.

```python
mxq_compile(model=onnx_path, calib_data_path=calib_txt, save_path=mxq_path,
            backend="onnx", inference_scheme="global8",
            cpu_offload=True, config_preset="vision_transformer")
```

- 출력 이름은 `<onnx 이름>_vt_global8.mxq`다. 벤치마크 코드가 이 이름으로 모델을 찾으므로 규칙은 `common/naming.py` 한곳에 있다.
- Preset 안에서 정해지는 bit 폭과 scale은 설치된 SDK에 따른다.
- **MXQ는 byte 단위로 재현되지 않는다.** 같은 ONNX를 두 번 컴파일하면 파일 해시가 다르다. 컴파일된 모델은 해시가 아니라 NPU 정확도로 비교한다.
- 컴파일할 때마다 MXQ 옆에 컴파일러 출력(`.compile.log`)과 기록(`.compile.json`: ONNX·MXQ의 SHA-256, 컴파일러 버전, calibration 정보, 소요 시간)을 남긴다.

## 10. 서버 측정

- **정확도** (`evaluation/evaluate.py`): ImageNet validation 50,000장, FP32, AMP off. Baseline, pruned, reduced, patch slimming 네 모델을 평가한다. 학습 로그의 validation 값(AMP)과는 다른 측정이다.
- **메모리** (`evaluation/profile_memory.py`): 새 프로세스, FP32, batch 1, random 입력(seed 2022), warmup 10회 뒤 forward 50회 동안의 `torch.cuda.max_memory_allocated()`. 평균이 아니라 최대값이다. 함께 기록하는 module 출력 합은 별도 forward 1회의 출력 텐서 크기 합이며 동시에 살아 있는 메모리의 peak가 아니다.

## 11. NPU 측정

코드: `benchmark/eval_imagenet.py`, `benchmark/profile_memory.py`

| 항목 | 값 |
|---|---|
| 데이터 | ImageNet validation, class당 50장(seed 42) = 50,000장. 모든 모델에 같은 이미지·순서 |
| 입력 | HWC float32, batch 1 |
| Core mode | `global8` (컴파일의 inference scheme과 같아야 한다) |
| Warmup | 모델당 10회 |
| Latency | `model.infer()` 구간만. 이미지 읽기와 전처리를 포함한 값은 end-to-end FPS로 따로 기록 |
| NPU telemetry | `mblt_tracker`, 500 ms 주기 (메모리, 사용률, 전력, 온도) |
| 독립 메모리 측정 | 모델마다 새 프로세스, 고정 random 입력(seed 2022), warmup 10회, latency 100회, telemetry 10초 |

- NPU 메모리는 장치 전체에서 관측한 MB 값이다. CUDA allocator peak나 layer별 activation peak와 같은 지표가 아니며, 다른 프로세스가 같은 NPU를 쓰면 값에 섞인다. 측정 중에는 NPU를 단독으로 쓴다.
- GFLOPs는 ONNX의 Conv/MatMul/Gemm을 정적 shape로 계산한 이론값이다(1 MAC = 2 FLOPs).
- 보고서의 `PASS`는 실행이 오류 없이 끝났다는 뜻이며 정확도 보존을 뜻하지 않는다.

## 12. 알려진 현상

- **Patch Slimming 모델은 NPU에서 정확도가 더 떨어진다.** 참고 실행에서 서버 FP32 대비 7.9~18.1%p 낮았다. Baseline과 reduced는 0.02~1.4%p 차이에 그친다. MXQ 변환의 양자화에서 생기는 것으로 판단하며 원인을 더 분리하지는 않았다. 이 저장소로 다시 실행해도 같은 현상이 나타날 것으로 예상한다.
- **서버 CUDA peak는 patch slimming이 reduced보다 조금 크다.** Token은 줄지만 선택 행렬과 명시적인 직사각형 attention 경로의 비용이 더해진다. NPU에서는 patch slimming이 메모리와 latency 모두 가장 작다. 서버 수치로 NPU 수치를 추정하지 않는다.
