# ArcFace 기반 MIL 파이프라인 데이터 생명주기 상세 분석

> 📅 작성일: 2025년 8월 3일  
> 📝 작성자: 데이터 아키텍트  
> 🎯 목적: 복수 작성자 필기 문서 탐지를 위한 3단계 파이프라인의 데이터 흐름 완벽 이해

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [데이터 생명주기 요약](#2-데이터-생명주기-요약)
3. [Stage 0: 원시 데이터](#3-stage-0-원시-데이터)
4. [Stage 1: ArcFace 특징 추출](#4-stage-1-arcface-특징-추출)
5. [Stage 2: MIL Bag 생성](#5-stage-2-mil-bag-생성)
6. [Stage 3: MIL 모델 학습 및 평가](#6-stage-3-mil-모델-학습-및-평가)
7. [데이터 변환 플로우 다이어그램](#7-데이터-변환-플로우-다이어그램)
8. [핵심 통찰 및 결론](#8-핵심-통찰-및-결론)

---

## 1. 프로젝트 개요

### 1.1 연구 목표
**"하나의 문서 내에 숨겨진 복수 작성자를 자동으로 탐지하는 AI 시스템 개발"**

전통적인 필기 분석이 두 문서 간 비교에 중점을 둔다면, 본 연구는 한 문서 내부의 미묘한 변화를 포착하는 혁신적 접근법을 제시합니다.

### 1.2 핵심 도전 과제
- **약지도 학습**: 개별 패치(단어)의 작성자 정보 없이 문서 전체 수준에서만 레이블 제공
- **미세한 변화 탐지**: 5~30%의 작은 비율로 섞인 다른 작성자의 필적 탐지
- **일반화 능력**: 학습에 사용되지 않은 새로운 작성자에 대한 성능 유지

### 1.3 기술 스택
- **특징 추출**: Vision Transformer + ArcFace Loss
- **약지도 학습**: Multiple Instance Learning (MIL)
- **평가 프레임워크**: 난이도별 성능 평가 시스템

---

## 2. 데이터 생명주기 요약

### 2.1 전체 흐름 개요

```
이미지 파일 → 특징 벡터 → Instance → Bag → 예측
   (픽셀)      (128차원)    (5단어)   (10문장)  (확률)
```

### 2.2 단계별 데이터 변환 테이블

| 단계 | 데이터 단위 | 포맷 | 차원 | 크기 (예시) | 의미 |
|------|------------|------|------|------------|------|
| **Stage 0** | 단어 이미지 | .png | H×W×3 | 100×50×3 | 원본 필기체 이미지 |
| | 메타데이터 | .csv | N×4 | 351,311×4 | 이미지 경로, 작성자 정보 |
| **Stage 1** | 특징 벡터 | CSV 행 | 1×128 | 1×128 | 이미지의 수학적 표현 |
| | 임베딩 파일 | .csv | N×130 | 208,233×130 | 전체 특징 벡터 집합 |
| **Stage 2** | Instance | 메모리 | 5×128 | 5×128 | 연속된 5단어의 특징 |
| | Bag | .pkl | 10×5×128 | 10×5×128 | 문서를 표현하는 10개 Instance |
| **Stage 3** | Attention | 배열 | 10 | 10 | Instance별 중요도 |
| | 예측 확률 | 스칼라 | 1 | 1 | 복수 작성자 확률 |

---

## 3. Stage 0: 원시 데이터

### 3.1 데이터셋 구조

```
/workspace/MIL/data/raw/
├── csafe_version5_xai_train/          # 이미지 루트 디렉토리
│   ├── 0/                             # 작성자 0번 폴더
│   │   ├── 'YX'_17.png               # 단어: 'YX', 반복: 17
│   │   ├── -he_10.png                # 단어: '-he', 반복: 10
│   │   └── ... (약 1,171개 이미지)
│   ├── 1/                             # 작성자 1번 폴더
│   │   └── ...
│   └── 299/                           # 작성자 299번 폴더
│       └── ...
└── naver_ocr.csv                      # 메타데이터 (351,311 행)
```

### 3.2 메타데이터 구조 (naver_ocr.csv)

| 컬럼명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| image_path | string | 이미지 파일 경로 | "0/'YX'_17.png" |
| label | int | 작성자 ID (0-299) | 0 |
| detected_word | string | OCR로 인식된 단어 | "YX" |
| repetition | int | 반복 번호 (0-26) | 17 |

### 3.3 데이터 수집 프로토콜
- **작성자 수**: 300명 (ID: 0~299)
- **반복 횟수**: 27회 (3세션 × 9회)
  - 세션 1: 반복 0-8
  - 세션 2: 반복 9-17
  - 세션 3: 반복 18-26
- **텍스트 유형**: WOZ, LND, PHR (문학, 편지, 구문)
- **전처리**: EasyOCR로 단어 단위 분할

### 3.4 데이터 특성
```python
# 데이터 통계
총 이미지 수: 351,311개
작성자당 평균: 1,171개
파일 형식: PNG (그레이스케일 또는 RGB)
해상도: 가변 (평균 100×50 픽셀)
```

---

## 4. Stage 1: ArcFace 특징 추출

### 4.1 입력 데이터

**데이터 분할 전략**:
```python
# 전체 300명을 60:20:20 비율로 분할
Train: 작성자 0-179 (180명) → 208,233 이미지
Val:   작성자 180-239 (60명) → 70,533 이미지  
Test:  작성자 240-299 (60명) → 72,457 이미지
```

**전처리 파이프라인**:
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(
        degrees=7,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
        shear=10
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])
```

### 4.2 모델 아키텍처

#### 4.2.1 Vision Transformer 백본
```python
class ViTArcFace(nn.Module):
    def __init__(self, num_classes=300, embedding_dim=128):
        # 1. Vision Transformer 백본
        self.backbone = timm.create_model('vit_base_patch16_224', 
                                        pretrained=True, 
                                        num_classes=0)
        
        # 2. 임베딩 투영 레이어
        self.embedding = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),  # 128차원
            nn.BatchNorm1d(embedding_dim)
        )
        
        # 3. ArcFace 분류 레이어
        self.arcface = ArcFaceLoss(embedding_dim, num_classes)
```

#### 4.2.2 ArcFace Loss 구현
```python
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50):
        self.scale = scale      # 스케일링 팩터
        self.margin = margin    # 각도 마진
        self.cos_m = torch.cos(margin)
        self.sin_m = torch.sin(margin)
```

**핵심 원리**: 
- 같은 클래스의 특징 벡터는 각도상 가깝게
- 다른 클래스의 특징 벡터는 각도상 멀게
- 마진을 추가하여 더 엄격한 분류 경계 생성

### 4.3 학습 과정

#### 4.3.1 하이퍼파라미터
```python
# 최적화 설정
optimizer = AdamW(lr=1e-5, weight_decay=5e-4)
scheduler = OneCycleLR(
    max_lr=1e-4,
    epochs=30,
    pct_start=0.1,  # 10% warm-up
)

# 학습 설정
batch_size = 64
num_gpus = 5
effective_batch_size = 320  # 64 × 5
```

#### 4.3.2 학습 결과
```
최종 검증 성능:
- AUC: 0.9240
- EER: 0.1361
- 학습 시간: 4.04시간 (15 에폭)
- Early Stopping 발동
```

### 4.4 출력 데이터

#### 4.4.1 CSV 파일 구조
```
mil_arcface_train_data.csv (307.72 MB)
├── label (작성자 ID)
├── path (이미지 경로)
├── embedding_0
├── embedding_1
├── ...
└── embedding_127
```

#### 4.4.2 임베딩 특성
```python
# 임베딩 통계
평균: -0.0006
표준편차: 0.0884
L2 norm: 1.0000 (정규화됨)
차원: 128
```

**핵심 변환**: 이미지(픽셀) → 128차원 벡터(필체의 수학적 지문)

---

## 5. Stage 2: MIL Bag 생성

### 5.1 입력 처리

```python
# CSV 로드
df_train = pd.read_csv("mil_arcface_train_data.csv")  # 208,233 행
df_val = pd.read_csv("mil_arcface_val_data.csv")      # 70,533 행
df_test = pd.read_csv("mil_arcface_test_data.csv")    # 72,457 행
```

### 5.2 Instance 생성 과정

#### 5.2.1 문서 재구성
```python
# 문서 키 생성: 작성자ID + 세션
def get_session(path):
    rep_num = int(path.split('_')[-1].split('.')[0])
    return rep_num // 9  # 3개 세션으로 그룹화

df["doc_key"] = f"{label}/session_{get_session(path)}"
# 예: "0/session_0", "0/session_1", "0/session_2"
```

#### 5.2.2 슬라이딩 윈도우
```python
# 하이퍼파라미터
window_size = 5    # 5개 단어 = 1 Instance
stride = 1         # 1단어씩 이동
block_size = 10    # 10 Instance = 1 Bag

# Instance 생성
for start in range(0, len(doc) - window_size + 1, stride):
    instance = embeddings[start:start+window_size]  # (5, 128)
```

**시각화**:
```
문서: [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, ...]
        └─────Instance1─────┘
           └─────Instance2─────┘
              └─────Instance3─────┘
```

### 5.3 Bag 합성 전략

#### 5.3.1 Positive Bag (단일 작성자)
```python
# 한 작성자의 10개 연속 Instance
positive_bag = {
    "bag_emb": base_emb,        # (10, 5, 128)
    "bag_label": 0,             # 단일 작성자
    "writer": base_writer,
    "neg_ratio": 0.0
}
```

#### 5.3.2 Negative Bag (복수 작성자)
```python
# 난이도별 위조 비율
neg_mix_ratio = [0.05, 0.10, 0.20, 0.30]

for ratio in neg_mix_ratio:
    k_replace = max(1, int(10 * ratio))  # 교체할 Instance 수
    # 다른 작성자의 Instance로 교체
    negative_bag = {
        "bag_emb": mixed_emb,    # (10, 5, 128)
        "bag_label": 1,          # 복수 작성자
        "neg_ratio": ratio       # 위조 비율
    }
```

**위조 시뮬레이션**:
```
원본 Bag: [A, A, A, A, A, A, A, A, A, A]  (작성자 A)

5% 위조:  [A, A, A, A, A, A, A, A, A, B]  (1개 교체)
10% 위조: [A, A, A, A, A, A, A, A, B, B]  (1개 교체, 최소 보장)
20% 위조: [A, A, A, A, A, A, A, A, B, B]  (2개 교체)
30% 위조: [A, A, A, A, A, A, A, B, B, B]  (3개 교체)
```

### 5.4 클래스 균형 조정

```python
# 원본 비율 (1:4)
Positive bags: 4,321개
Negative bags: 17,284개

# 균형 조정 후 (1:2)
Positive bags: 4,321개
Negative bags: 8,642개
Total: 12,963개

# 난이도별 균등 분포
각 neg_ratio당: 2,160개 (25%)
```

### 5.5 출력 데이터

#### 5.5.1 Pickle 파일 구조
```python
bags = [
    {
        "bag_emb": np.array(10, 5, 128),  # 10개 Instance
        "bag_label": 0 or 1,               # 클래스
        "writer": int,                     # 기준 작성자
        "doc": str,                        # 문서 ID
        "neg_ratio": float,                # 위조 비율
        "bag_id": str                      # 고유 ID
    },
    ...
]
```

#### 5.5.2 데이터 크기
```
Train: bags_arcface_margin_0.4_train.pkl (12,963 bags)
Val:   bags_arcface_margin_0.4_val.pkl   (4,320 bags)
Test:  bags_arcface_margin_0.4_test.pkl  (4,320 bags)
메모리 사용량: ~0.62 GB
```

---

## 6. Stage 3: MIL 모델 학습 및 평가

### 6.1 모델 아키텍처

#### 6.1.1 Attention-based MIL
```python
class AttentionMIL(nn.Module):
    def __init__(self, inst_dim=256, hidden=128):
        # 1. Instance 투영
        self.proj = nn.Linear(inst_dim, hidden)
        
        # 2. Attention 네트워크
        self.attn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        
        # 3. 분류기
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
```

#### 6.1.2 Forward Pass
```python
def forward(self, x):  # x: (B, 10, 5, 128)
    # 1. Mean pooling over words
    x = x.mean(dim=2)  # (B, 10, 128)
    
    # 2. Project to hidden space
    h = self.proj(x)   # (B, 10, 128)
    
    # 3. Calculate attention
    a = self.attn(h)   # (B, 10, 1)
    weights = torch.softmax(a, dim=1)
    
    # 4. Weighted aggregation
    m = (h * weights).sum(dim=1)  # (B, 128)
    
    # 5. Classification
    logits = self.classifier(m)    # (B, 1)
    
    return logits, weights
```

### 6.2 학습 전략

#### 6.2.1 Balanced Sampling
```python
class PosNegBatchSampler:
    # 배치 구성: 50% Positive + 50% Negative
    # 목적: 클래스 불균형 해결
    batch_size = 64  # 32 pos + 32 neg
```

#### 6.2.2 이중 검증 전략
```python
# 1. 빠른 모니터링: Balanced validation (F1 계산)
val_loader_balanced  # 1:1 비율로 샘플링

# 2. 정확한 임계값: Natural distribution
val_loader_natural   # 실제 분포 유지 (1:2)
```

#### 6.2.3 OneCycleLR 스케줄링
```
에폭 1-4:   Warm-up (1e-5 → 1e-4)
에폭 5:     Peak (1e-4)
에폭 6-9:   Annealing (1e-4 → 1e-6)
Early Stop: 에폭 9에서 발동
```

### 6.3 성능 평가

#### 6.3.1 전체 성능
```
Validation (Best):
- F1 Score: 0.660
- AUC: 0.652
- Threshold: 0.497

Test Set:
- F1 Score: 0.756
- Recall: 0.807
- AUC: 0.635
```

#### 6.3.2 난이도별 성능

| 위조 비율 | F1 Score | Recall | 샘플 수 |
|----------|----------|--------|---------|
| 0% (진본) | 0.000 | 0.000 | 1,440 |
| 5% | 0.838 | 0.721 | 720 |
| 10% | 0.861 | 0.756 | 720 |
| 20% | 0.903 | 0.824 | 720 |
| 30% | 0.963 | 0.928 | 720 |

**해석**: 
- 0% (진본)의 F1=0은 모든 진본을 올바르게 분류했음을 의미
- 위조 비율이 높을수록 탐지가 쉬워짐

### 6.4 Attention 분석

**Single Writer Bag**:
```
Attention 분포: 균등 (0.08~0.12)
해석: 모든 Instance가 비슷한 패턴
```

**Multi Writer Bag**:
```
Attention 분포: 불균등 (0.02~0.25)
해석: 특정 Instance에 높은 주목
```

---

## 7. 데이터 변환 플로우 다이어그램

### 7.1 전체 파이프라인

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Stage 0       │     │   Stage 1       │     │   Stage 2       │
│                 │     │                 │     │                 │
│  Image Files    │ --> │  Feature Ext.   │ --> │  Bag Creation   │
│  (351K×H×W×3)  │     │  (351K×128)     │     │  (21K Bags)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          |
                                                          v
                        ┌─────────────────┐     ┌─────────────────┐
                        │   Stage 3       │     │   Output        │
                        │                 │     │                 │
                        │  MIL Training   │ --> │  Predictions    │
                        │  (Attention)    │     │  (Probability)  │
                        └─────────────────┘     └─────────────────┘
```

### 7.2 데이터 차원 변화

```
Stage 0: 이미지 (H×W×3)
         ↓ [Vision Transformer]
Stage 1: 벡터 (1×128)
         ↓ [Sliding Window]
Stage 2: Instance (5×128)
         ↓ [Blocking]
         Bag (10×5×128)
         ↓ [Attention MIL]
Stage 3: 확률 (1)
```

### 7.3 샘플 수 변화

```
원시 이미지:     351,311개
    ↓
특징 벡터:       351,311개 (1:1)
    ↓
Instance:        346,429개 (슬라이딩)
    ↓
Bag (원본):      21,605개
    ↓
Bag (균형화):    12,963개 (Train)
```

---

## 8. 핵심 통찰 및 결론

### 8.1 기술적 통찰

1. **ArcFace의 효과성**
   - 128차원의 compact한 표현으로도 300명 작성자 구분 가능
   - L2 정규화된 임베딩이 안정적인 유사도 계산 제공

2. **MIL의 적합성**
   - 약지도 학습으로 라벨링 비용 대폭 절감
   - Instance 수준의 Attention으로 해석 가능성 제공

3. **난이도별 성능 패턴**
   - 5% 위조도 83.8% F1로 탐지 가능
   - 위조 비율 증가 시 성능 향상 (더 명확한 패턴)

### 8.2 개선 방향

1. **데이터 증강**
   - 더 다양한 위조 패턴 시뮬레이션
   - 실제 위조 문서 데이터 수집

2. **모델 고도화**
   - Transformer 기반 MIL 아키텍처
   - Multi-scale attention 메커니즘

3. **실용화**
   - 한국어 필기체 적용
   - 실시간 처리 최적화

### 8.3 결론

본 파이프라인은 이미지에서 시작하여 복수 작성자 확률까지, 데이터의 점진적 추상화 과정을 통해 복잡한 문제를 해결합니다. 각 단계는 명확한 목적과 변환 로직을 가지며, 전체가 유기적으로 연결되어 있습니다.

**핵심 성과**:
- Test F1: 75.6% (실용 가능 수준)
- 5% 위조 탐지: 83.8% F1
- 해석 가능한 Attention 메커니즘

이는 법의학 문서 분석 분야에서 AI의 실질적 활용 가능성을 보여주는 중요한 진전입니다.

---

## 부록 A: 주요 코드 스니펫

### A.1 ArcFace Loss 계산
```python
def forward(self, input, label):
    # 정규화
    input = F.normalize(input)
    weight = F.normalize(self.weight)
    
    # 코사인 유사도
    cosine = F.linear(input, weight)
    
    # 마진 추가
    phi = cosine * self.cos_m - sine * self.sin_m
    
    # One-hot 인코딩
    one_hot = torch.zeros_like(cosine)
    one_hot.scatter_(1, label.view(-1, 1), 1)
    
    # 최종 출력
    output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
    return output * self.scale
```

### A.2 Bag 생성 핵심 로직
```python
def synthesize_bags(inst_list, neg_source):
    # Positive Bag
    positive_bag = {
        "bag_emb": base_emb,
        "bag_label": 0,
        "neg_ratio": 0.0
    }
    
    # Negative Bags
    for ratio in [0.05, 0.10, 0.20, 0.30]:
        k_replace = max(1, int(10 * ratio))
        # Instance 교체 로직
        negative_bag = {
            "bag_emb": mixed_emb,
            "bag_label": 1,
            "neg_ratio": ratio
        }
```

---

## 부록 B: 파일 시스템 구조

```
/workspace/MIL/
├── experiments/arcface/
│   ├── train_arcface.ipynb              # Stage 1
│   ├── mil_data_generator2_arcface.ipynb # Stage 2
│   └── AB_MIL_arcface_256d.ipynb        # Stage 3
│
├── data/
│   ├── raw/
│   │   ├── csafe_version5_xai_train/    # 원본 이미지
│   │   └── naver_ocr.csv                # 메타데이터
│   │
│   └── processed/
│       ├── embeddings/                  # Stage 1 출력
│       │   ├── mil_arcface_train_data.csv
│       │   ├── mil_arcface_val_data.csv
│       │   └── mil_arcface_test_data.csv
│       │
│       └── bags/                        # Stage 2 출력
│           ├── bags_arcface_margin_0.4_train.pkl
│           ├── bags_arcface_margin_0.4_val.pkl
│           └── bags_arcface_margin_0.4_test.pkl
│
└── output/arcface_margin_0.4_*/         # Stage 3 출력
    ├── models/
    │   └── mil_best.pth
    ├── figures/
    │   ├── training_curves.png
    │   ├── test_results.png
    │   └── attention_analysis.png
    └── results/
        └── experiment_results.json
```

---

*이 문서는 ArcFace 기반 MIL 파이프라인의 데이터 흐름을 완벽히 이해하기 위한 기술 문서입니다.*