# 참고 실행 final_20261006

이 저장소의 방법으로 얻은 기존 최종 결과다. 새로 실행한 결과의 비교 대상이며, 실행기의 출력 폴더로 쓰지 않는다. 서버 쪽은 2026-10-06 ~ 10-08에 RTX 2080 Ti × 4로, NPU 측정은 2026-10-15에 Mobilint Aries로 수행했다.

| 경로 | 내용 |
|---|---|
| `onnx/` | 서버에서 export한 ONNX 12개와 검증 기록(`.verified.json`) |
| `mxq/` | 위 ONNX를 컴파일해 NPU에서 실제로 평가한 MXQ 12개 |
| `server_run/` | 서버 결과표(`FINAL_RESULTS.md`, CSV, JSON), 실험별 epoch 기록과 token 구조 |
| `npu_benchmark/` | NPU 측정 원본: `summary.md`, `benchmark.json`, 평가 이미지 목록, 모델별 예측, 독립 메모리 측정 |
| `calibration_images.txt` | MXQ 컴파일에 쓴 calibration 이미지 1,000장의 목록(`<class>/<파일>`) |
| `SHA256SUMS` | ONNX·MXQ의 SHA-256 |

ONNX와 MXQ는 이름으로 대응한다. `<이름>.onnx`를 컴파일한 것이 `<이름>_vt_global8.mxq`이고, 그 MXQ의 측정 결과가 `npu_benchmark/models/<이름>_vt_global8.json`이다.

```text
onnx/tiny30__reduced__nhwc_b1_npusafe.onnx
mxq/tiny30__reduced__nhwc_b1_npusafe_vt_global8.mxq
npu_benchmark/models/tiny30__reduced__nhwc_b1_npusafe_vt_global8.json
```

`.onnx`와 `.mxq`는 Git LFS로 관리한다(합계 약 618 MiB). 받은 뒤 무결성을 확인한다.

```bash
git lfs pull
(cd reference/final_20261006 && sha256sum -c SHA256SUMS)
```

알아둘 점:

- 이 실행의 ImageNet은 공식 배포본과 다른 사본이다(`docs/DATASET.md`). `calibration_images.txt`와 `npu_benchmark/samples.tsv`의 파일 이름은 그 사본의 것이다.
- 측정 파일에는 당시 서버의 절대 경로가 그대로 들어 있다. 기록이므로 고치지 않았다.
- 참고 MXQ의 컴파일 로그와 컴파일러 버전은 남아 있지 않다. 새로 컴파일하면 MXQ 옆에 기록이 남는다.
- `npu_benchmark/run_metadata.json`의 전처리 이름 `timm`은 이 저장소의 `common/preprocess.py`와 같은 OpenCV 전처리를 가리킨다.
