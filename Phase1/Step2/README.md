# CRAFT 텍스트 감지 기반 MIL 패치 추출기

## 개요

이 프로젝트는 문서 이미지에서 CRAFT 텍스트 감지 모델을 사용하여 텍스트 영역을 감지하고, 이를 기반으로 고정 크기의 패치를 추출하는 파이썬 라이브러리입니다. 추출된 패치는 다중 인스턴스 학습(Multiple Instance Learning, MIL) 형식으로 구성되어 다중 저자 문서 판별과 같은 작업에 활용할 수 있습니다.

## 주요 기능

1. **텍스트 영역 감지**: CRAFT 모델을 사용하여 문서 이미지에서 텍스트 영역을 정확하게 감지합니다.
2. **패치 추출**: 감지된 텍스트 영역을 중심으로 고정 크기(224x224)의 패치를 추출합니다.
3. **MIL 데이터셋**: 추출된 패치를 MIL 형식으로 구성하여, 각 이미지에서 추출된 패치 그룹을 '가방(bag)'으로 취급합니다.
4. **패치 중복 제거**: 비최대 억제(Non-Maximum Suppression) 알고리즘을 사용하여 중복되는 패치를 제거합니다.
5. **시각화 도구**: 텍스트 감지 결과와 추출된 패치를 시각화하는 기능을 제공합니다.

## 설치 방법

필요한 라이브러리를 설치하기 위해 다음 명령어를 실행하세요:

```bash
pip install torch torchvision opencv-python numpy matplotlib craft-text-detector tqdm
```

## 파일 구조

- `craft_utils.py`: CRAFT 텍스트 감지 모델을 로드하고 텍스트 영역을 감지하는 유틸리티 클래스
- `craft_mil_dataset.py`: CRAFT를 사용하여 패치를 추출하는 PyTorch 데이터셋 클래스
- `test_craft_detection.py`: 텍스트 감지 및 패치 추출 기능을 테스트하고 시각화하는 스크립트

## 사용 방법

### 1. 텍스트 감지 테스트

단일 이미지에서 텍스트 영역을 감지하고 시각화하려면:

```bash
python test_craft_detection.py --test_mode detection --sample_image path/to/your/image.jpg
```

### 2. 패치 추출 테스트

단일 이미지에서 텍스트 영역 기반 패치를 추출하고 시각화하려면:

```bash
python test_craft_detection.py --test_mode patches --sample_image path/to/your/image.jpg
```

### 3. MIL 데이터셋 테스트

전체 데이터셋에서 무작위로 선택된 샘플에 대해 MIL 패치 추출을 테스트하려면:

```bash
python test_craft_detection.py --test_mode dataset --root_dir path/to/mil_dataset --num_samples 5
```

### 4. 데이터 로더 테스트

미니 배치 형식의 MIL 패치 추출을 테스트하려면:

```bash
python test_craft_detection.py --test_mode dataloader --root_dir path/to/mil_dataset
```

### 5. 모든 테스트 실행

모든 테스트 모드를 한 번에 실행하려면:

```bash
python test_craft_detection.py --test_mode all --root_dir path/to/mil_dataset
```

## CraftMILDataset 클래스 사용 예시

```python
from craft_utils import CraftTextDetector
from craft_mil_dataset import CraftMILDataset, get_craft_mil_dataloader

# CRAFT 텍스트 감지기 초기화
craft_detector = CraftTextDetector()

# 데이터셋 초기화
dataset = CraftMILDataset(
    root_dir='mil_dataset',
    split='train',
    patch_size=(224, 224),
    max_patches_per_bag=50,
    craft_detector=craft_detector,
    precompute_patches=True,
    cache_dir='cache'
)

# 데이터 로더 초기화
dataloader = get_craft_mil_dataloader(
    root_dir='mil_dataset',
    batch_size=4,
    split='train',
    craft_detector=craft_detector,
    num_workers=2
)

# 데이터셋에서 단일 샘플 가져오기
sample = dataset[0]
patches = sample['patches']  # 텍스트 영역 기반 패치 (N, C, H, W)
bag_label = sample['bag_label']  # 다중 저자(1) 또는 단일 저자(0)

# 데이터 로더에서 미니 배치 가져오기
for batch in dataloader:
    patches = batch['patches']  # 패치 리스트 (배치 크기)
    patch_counts = batch['patch_counts']  # 각 샘플별 패치 수
    bag_labels = batch['bag_labels']  # 가방 레이블 (배치 크기)
```

## 참고 자료

- [CRAFT 공식 리포지토리](https://github.com/clovaai/CRAFT-pytorch)
- [craft-text-detector PyPI 패키지](https://github.com/fcakyon/craft-text-detector) 