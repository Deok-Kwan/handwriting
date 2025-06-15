# MIL (Multiple Instance Learning) 데이터셋 관리 도구

다중 인스턴스 학습(MIL)을 위한 필기체 데이터 생성, 구조화 및 로드를 위한 도구 모음입니다.

## 프로젝트 개요

이 프로젝트는 다음과 같은 기능을 제공합니다:

1. **데이터 생성**: 여러 작성자의 필기체 라인을 조합하여 합성 문서 생성
2. **파일 구조화**: 생성된 데이터를 효율적인 디렉토리 구조로 정리
3. **데이터 로더**: 머신러닝 모델 학습을 위한 PyTorch 데이터셋 및 데이터 로더

## 디렉토리 구조

```
MIL/
├── Data/
│   └── Data2.py            # 합성 문서 생성 스크립트
├── organize_mil_dataset.py # 데이터셋 구조화 도구
├── create_mil_dataloader.py # PyTorch 데이터 로더
├── synth_output_mil_mixed/ # 원본 생성 데이터
│   ├── *.png               # 이미지 파일
│   └── logs/               # 메타데이터 로그
├── mil_dataset/            # 구조화된 데이터셋
│   ├── single_author/      # 단일 작성자 문서
│   ├── multi_author/       # 다중 작성자 문서
│   ├── splits/             # 데이터셋 분할
│   └── index/              # 메타데이터 인덱스
└── line_segments/          # 원본 필기체 라인 이미지
```

## 사용 방법

### 1. 데이터 생성 (Data2.py)

필기체 라인 이미지를 이용하여 합성 문서를 생성합니다:

```bash
cd /path/to/MIL
python Data/Data2.py
```

생성 매개변수는 스크립트 상단에서 설정할 수 있습니다.

### 2. 데이터셋 구조화 (organize_mil_dataset.py)

생성된 문서를 체계적인 디렉토리 구조로 정리합니다:

```bash
python organize_mil_dataset.py --source_dir ./synth_output_mil_mixed/ --target_dir ./mil_dataset/ --split --create_index
```

옵션:
- `--source_dir`: 원본 데이터 디렉토리
- `--target_dir`: 구조화된 데이터를 저장할 디렉토리
- `--split`: 데이터를 학습/검증/테스트 세트로 분할
- `--train_ratio`: 학습 데이터 비율 (기본값: 0.7)
- `--val_ratio`: 검증 데이터 비율 (기본값: 0.15)
- `--create_index`: 검색 가능한 메타데이터 인덱스 생성

### 3. 데이터 로더 사용 (create_mil_dataloader.py)

모델 학습을 위한 PyTorch 데이터 로더를 생성합니다:

```python
from create_mil_dataloader import get_dataloader

# 분류 모델용 데이터 로더
train_loader = get_dataloader(
    root_dir='./mil_dataset',
    batch_size=8,
    split='train',
    mode='classification'  # 'classification', 'segmentation', 'writer_identification'
)

# 데이터 로더 사용
for batch in train_loader:
    images = batch['image']    # 형태: [B, 3, H, W]
    labels = batch['label']    # 형태: [B]
    # 모델 학습 코드...
```

## 지원하는 작업 유형

이 데이터셋은 다음과 같은 머신러닝 작업을 지원합니다:

1. **문서 분류**: 단일 작성자 vs 다중 작성자 문서 분류
2. **세그멘테이션**: 작성자별 필기 영역 분할
3. **작성자 식별**: 문서 또는 필기 영역에서 작성자 식별

## 데이터셋 통계 (샘플)

- 총 문서 수: X개
- 단일 작성자 문서: Y개 
- 다중 작성자 문서: Z개
- 고유 작성자 수: W명

## 요구 사항

- Python 3.6+
- PyTorch 1.7+
- OpenCV
- NumPy
- Albumentations
- Pandas
- SQLite3

## 작성자

이 프로젝트는 MIL(Multiple Instance Learning) 데이터셋 관리를 위해 개발되었습니다. 