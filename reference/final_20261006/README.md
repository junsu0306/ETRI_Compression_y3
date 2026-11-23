# 참고 실행 final_20261006

이 저장소의 방법으로 얻은 기존 최종 결과다. 새로 실행한 결과의 비교 대상이며, 실행기의 출력 폴더로 쓰지 않는다. 서버 쪽은 2026-10-06 ~ 10-08에 RTX 2080 Ti × 4로, NPU 측정은 2026-10-15에 Mobilint Aries로 수행했다.

| 경로 | 내용 |
|---|---|
| `server_run/` | 서버 결과표(`FINAL_RESULTS.md`, CSV, JSON), 실험별 epoch 기록과 token 구조 |
| `npu_benchmark/` | NPU 측정 원본: `summary.md`, `benchmark.json`, 평가 이미지 목록, 모델별 예측, 독립 메모리 측정 |
| `calibration_images.txt` | MXQ 컴파일에 쓴 calibration 이미지 1,000장의 목록(`<class>/<파일>`) |

이 실행의 ONNX와 MXQ 파일은 저장소에 두지 않고 측정 기록만 둔다. 모델별 NPU 측정 결과는 `npu_benchmark/models/<MXQ 이름>.json`에 있다. 이 저장소의 코드로 다시 만든 모델은 [models/final](../../models/final/README.md)에 있다.

알아둘 점:

- 이 실행의 ImageNet은 공식 배포본과 다른 사본이다(`docs/DATASET.md`). `calibration_images.txt`와 `npu_benchmark/samples.tsv`의 파일 이름은 그 사본의 것이다.
- 측정 파일에는 당시 서버의 절대 경로가 그대로 들어 있다. 기록이므로 고치지 않았다.
- 참고 MXQ의 컴파일 로그와 컴파일러 버전은 남아 있지 않다.
- `npu_benchmark/run_metadata.json`의 전처리 이름 `timm`은 이 저장소의 `common/preprocess.py`와 같은 OpenCV 전처리를 가리킨다.
