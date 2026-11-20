# 데이터 준비 (ImageNet-1k)

이 저장소는 ImageNet 이미지를 포함하지 않는다. ImageNet 이용 약관에 따라 직접 받아서 아래 구조로 준비한다.

## 필요한 폴더 구조

```text
<imagenet>/
├── train/<class>/<image>     # 공식 배포본 1,281,167장
└── val/<class>/<image>       # 50,000장, class당 50장
```

`<class>` 폴더 이름은 두 가지 모두 쓸 수 있다.

| 폴더 이름 | 예 | label |
|---|---|---|
| WordNet ID | `n01440764`, `n01443537`, … | 정렬한 순서의 index |
| class index | `00000`, `00001`, … | 폴더 이름의 숫자 |

두 경우 모두 정렬한 폴더 순서가 표준 ImageNet-1k class index(0 = tench, 1 = goldfish, …)와 같다. 서버 쪽 코드(`torchvision.datasets.ImageFolder`)와 NPU 벤치마크(`benchmark/eval_imagenet.py`)가 같은 label을 쓴다. 공식 배포본의 `val/`은 폴더 구분 없이 50,000장이 한 폴더에 있으므로 class 폴더로 나누어야 한다.

데이터 위치는 `--data-path`, 환경 변수 `IMAGENET_DIR`, 설정 파일의 `data_path` 순으로 정한다.

## 준비 방법

### Hugging Face `ILSVRC/imagenet-1k`에서 받는 경우

검증 실행에 쓴 방법이다. 계정으로 로그인해 데이터셋 페이지에서 약관에 동의한 뒤 받는다. 받는 데 `huggingface_hub`, 추출에 `pyarrow`가 필요하다.

```bash
pip install huggingface_hub pyarrow
huggingface-cli login
huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset \
  --include "data/train-*.parquet" "data/validation-*.parquet" --local-dir /data/imagenet_hf

python -m dataset.prepare_imagenet from-hf \
  --parquet-dir /data/imagenet_hf/data --output /data/imagenet
```

Parquet 308개(train 294개 136 GiB, validation 14개 6.2 GiB)에 들어 있는 원본 JPEG 바이트를 다시 인코딩하지 않고 그대로 파일로 쓴다. 파일 이름은 공식 이름(`n01440764_10026.JPEG`, `ILSVRC2012_val_00000293.JPEG`)이고 폴더는 WordNet ID다. 추출이 끝나면 parquet은 지워도 된다.

### 공식 tar 배포본을 이미 가지고 있는 경우

`train/`은 푼 그대로 쓴다. `val/`만 class 폴더로 나눈다. Kaggle 배포본의 `LOC_val_solution.csv`, 또는 `<이미지 이름> <WordNet ID>` 형식의 텍스트 파일을 label로 받는다.

```bash
python -m dataset.prepare_imagenet val-folders \
  --val-dir /data/ILSVRC2012/val_flat --labels LOC_val_solution.csv --output /data/imagenet/val
```

기본은 심볼릭 링크를 만들고 `--move`를 주면 파일을 옮긴다.

### 확인

```bash
python -m dataset.prepare_imagenet check --root /data/imagenet
```

Class 수(1,000), split별 이미지 수, class당 최소·최대 장수, train과 val의 폴더 이름 일치를 검사한다. Validation이 50,000장(class당 50장)이 아니면 실패한다.

### 축소 실행용 부분집합

`configs/smoke.yaml`은 작은 부분집합으로 전체 단계를 빠르게 확인한다. class당 train 8장, val 2장을 심볼릭 링크로 만든다.

```bash
python -m dataset.prepare_imagenet subset --source /data/imagenet --output data/imagenet_smoke
```

## Calibration 이미지

MXQ 컴파일에는 train에서 뽑은 1,000장을 쓴다. 파이프라인은 class 폴더마다 1장을 seed(`calibration.seed`, 기본 2022)로 뽑는다. 같은 데이터와 같은 seed면 항상 같은 이미지가 뽑힌다. 실제로 쓴 이미지의 상대 경로와 SHA-256은 `<output_root>/calibration/images.tsv`에 남는다.

정해진 목록을 쓰려면 설정의 `calibration.image_list`에 `<data_path>/train` 기준 상대 경로가 한 줄에 하나씩 있는 파일을 지정한다.

## 데이터에 따른 수치 차이

참고 결과(`reference/final_20261006`)는 공식 배포본이 아닌 다른 사본으로 얻은 것이다. 그 사본은 class 폴더가 `00000`~`00999`, 파일 이름이 숫자이고, 이미지가 다시 인코딩되어 있으며(train 약 50 GB) train이 1,280,297장으로 공식본보다 870장 적다. 출처와 만든 방법은 기록되어 있지 않다. 그 사본에서 뽑아 참고 MXQ 컴파일에 쓴 calibration 1,000장의 목록이 `reference/final_20261006/calibration_images.txt`다(seed 없이 뽑은 것이라 다시 만들 수 없다).

데이터가 다르면 수치도 달라진다. Pretrained baseline의 FP32 정확도부터 사본마다 조금 다르므로, 다른 데이터로 얻은 결과를 참고값과 비교할 때는 이 차이를 감안한다. 두 데이터의 baseline 정확도는 [RESULTS.md](RESULTS.md)에 나란히 있다.

NPU 평가는 class당 50장을 seed 42로 뽑는데 validation은 class당 정확히 50장이므로 50,000장 전체를 쓰게 된다. 파일 이름이 달라도 평가 대상 집합은 같다.
