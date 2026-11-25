# 최종 모델

이 저장소의 코드와 `configs/final.yaml`로 처음부터 실행해 얻은 ONNX 12개와 MXQ 12개다. 공식 ImageNet, RTX A5000 4장, PyTorch 2.7.1+cu128, qbcompiler 1.1.0으로 만들었고, 서버 결과는 [docs/RESULTS.md](../../docs/RESULTS.md) 2절에 있다. 실행기가 만든 `output/final/deploy/` 폴더를 그대로 옮긴 것이다.

| 경로 | 내용 |
|---|---|
| `onnx/` | export한 ONNX 12개와 검증 기록(`.verified.json`) |
| `mxq/` | 위 ONNX를 컴파일한 MXQ 12개와 컴파일 기록(`.compile.json`) |
| `SHA256SUMS` | 위 파일 전체의 SHA-256 |
| `npu_benchmark/` | NPU 측정 원본: `summary.md`, `benchmark.json`, 평가 이미지 목록, 모델별 예측, 독립 메모리 측정 |

ONNX와 MXQ는 이름으로 대응한다. `<이름>.onnx`를 컴파일한 것이 `<이름>_vt_global8.mxq`다.

```text
onnx/tiny30__reduced__nhwc_b1_npusafe.onnx
mxq/tiny30__reduced__nhwc_b1_npusafe_vt_global8.mxq
```

`.onnx`와 `.mxq`는 Git LFS로 관리한다(합계 약 619 MiB). 받은 뒤 무결성을 확인한다.

```bash
git lfs pull
(cd models/final && sha256sum -c SHA256SUMS)
```

알아둘 점:

- Tiny30과 Tiny50의 baseline, Small30과 Small50의 baseline은 같은 pretrained 모델이다. ONNX는 같고 MXQ는 따로 컴파일한 것이다.
- 기록 파일에는 만든 서버와 측정한 장치의 절대 경로가 그대로 들어 있다.
- NPU 결과는 [docs/RESULTS.md](../../docs/RESULTS.md) 2.6~2.7절에 있다. 측정 방법은 [docs/RUNBOOK.md](../../docs/RUNBOOK.md) 7절에 있다. 모델별 측정 결과는 `npu_benchmark/models/<MXQ 이름>.json`이다.
