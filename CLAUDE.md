# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

필기 분석 및 컴퓨터 비전에 중점을 둔 머신러닝 연구 프로젝트입니다. 프로젝트는 여러 구성 요소를 포함합니다:

- **Hand/**: Vision Transformer(ViT)와 오토인코더를 사용한 필기 분석 실험
- **KCI/**: 한국논문 실험 - 다양한 클래스 수와 불확실성 분석
- **MIL/**: 텍스트 탐지 및 인식을 위한 다중 인스턴스 학습
- **MCP/**: Claude 통합 강화를 위한 모델 컨텍스트 프로토콜 도구

## MIL 프로젝트 구조 (2025년 6월 26일 업데이트)

```
MIL/
├── experiments/                # 모든 실험 관련 코드
│   ├── autoencoder/           # Autoencoder 기반 실험
│   │   ├── mil_data_generator.ipynb      # 오토인코더 임베딩 추출
│   │   ├── mil_data_generator2.ipynb     # MIL Bag 생성
│   │   └── AB_MIL_autoencoder_128d.ipynb # Attention-Based MIL 학습 (베이스라인)
│   │
│   └── siamese/               # Siamese Network 기반 실험 (개선된 방법)
│       ├── train_siamese.ipynb           # Siamese 네트워크 학습
│       ├── mil_data_generator_siamese.ipynb      # Siamese 임베딩 추출
│       ├── mil_data_generator2_siamese.ipynb     # MIL Bag 생성
│       └── AB_MIL_siamese_128d.ipynb            # Attention-Based MIL 학습
│
├── data/                      # 데이터 관련 파일
│   ├── raw/                   # 원본 데이터
│   │   ├── naver_ocr.csv      # OCR 처리된 필기 데이터 메타정보
│   │   ├── csafe_vit_300classes_best_model.pth  # 사전학습 ViT 모델
│   │   └── csafe_version5_xai_train.tar.gz      # 원본 이미지 데이터
│   │
│   └── processed/             # 처리된 데이터
│       ├── embeddings/        # 특징 벡터 CSV 파일들
│       └── bags/              # MIL Bag PKL 파일들
│
├── output/                    # 실험 결과
│   ├── models/               # 학습된 모델 (.pth, .pt)
│   ├── figures/              # 그래프 및 시각화 결과
│   └── results/              # 실험 결과 JSON, 로그 등
│
├── docs/                     # 프로젝트 문서
│   ├── AB_MIL.md            # 전체 프로젝트 개요 및 결과 분석
│   ├── AB_MIL_data_gen.md   # 데이터 생성 및 환경 구축 가이드
│   └── MIL_data_inspect.md  # 데이터 검사 및 분석 문서
│
├── Roadmap.md                # 연구 로드맵 및 단계별 계획 (NEW)
│
└── archive/                  # 이전 실험들 (구 old/ 폴더)
```

## 개발 환경

### 컨테이너 설정
```bash
# 개발 컨테이너 시작
./start_claude_container.sh

# 컨테이너 포함 사항:
# - Node.js 18 런타임
# - 포트 2222를 통한 SSH 서버 접근
# - 협업을 위한 작업공간 마운트
```

### Python 환경
```bash
# Python 종속성 설치
pip install -r MIL/requirements.txt

# 주요 종속성:
# - CUDA 지원 PyTorch
# - torchvision, timm (비전 모델용)
# - scikit-learn, pandas, numpy
# - 이미지 처리용 OpenCV, PIL
# - 데이터 증강용 albumentations
```

### Docker 환경
```bash
# Dockerfile을 사용하여 빌드 및 실행 (Claude Code 통합)
docker build -t claude-workspace .
docker run -v /workspace:/workspace claude-workspace
```

## 핵심 아키텍처

### 필기 분석 (Hand/)
- **Vision Transformer 모델**: 사용자 정의 분류 헤드와 함께 `timm.create_model('vit_base_patch16_224')` 사용
- **오토인코더 통합**: 300 → 128차원 잠재 공간 압축
- **다중 데이터셋 훈련**: bootstrapping, simple_split, text_detection 데이터셋 결합
- **유사도 분석**: ROC 곡선 평가와 함께 코사인 유사도 기반 필기자 식별

### 다중 인스턴스 학습 (MIL/) - 복수 작성자 필기 문서 탐지 연구

## MIL 연구 개요

### 연구 로드맵
**상세 로드맵 파일**: `/workspace/MIL/Roadmap.md`
- 6개 Phase로 구성된 10주 연구 계획
- 베이스라인 개선부터 실용화까지 단계별 접근
- 주요 마일스톤 및 평가 지표 포함

### 연구 배경과 동기

#### 사회적 필요성
현대 사회에서 문서의 진위 판별은 법의학, 계약, 유언장 등 다양한 분야에서 핵심적 역할을 합니다. 특히 문서 위조가 점점 정교해지면서 전체가 아닌 일부분만 조작하는 사례가 증가하고 있습니다. 이러한 복잡한 위조를 탐지하기 위해서는 문서 내에 복수 작성자가 존재하는지를 자동으로 판별할 수 있는 기술이 필요합니다.

#### 기존 방법의 한계
- **전문가 의존성**: 필적 감정은 전문가의 주관적 판단에 의존하여 시간과 비용이 많이 소요
- **일관성 부족**: 감정인에 따라 결과가 달라질 수 있는 문제
- **접근성 격차**: 개인과 기업 간 법적 방어 능력의 불균형

### 연구 목적과 핵심 문제

#### 연구 목적
**"문서 내 숨겨진 복수 작성자를 자동으로 탐지하는 AI 시스템 개발"**

본 연구는 하나의 문서 안에 여러 작성자의 필적이 혼재하는지를 판별하는 혁신적인 접근법을 제시합니다. 이는 단순히 두 문서를 비교하는 기존 연구와 달리, 문서 내부의 미묘한 변화를 포착하는 더 복잡하고 현실적인 문제를 해결합니다.

#### 핵심 문제 정의
- **입력**: 하나의 문서 (여러 필적 패치로 구성)
- **출력**: 단일 작성자 문서 vs 복수 작성자 문서 판별
- **도전과제**: 개별 패치의 작성자 정보 없이 문서 전체 수준에서 판별

### 연구 방법론 - Multiple Instance Learning (MIL)

#### MIL 프레임워크
- **Bag**: 문서 전체 (여러 패치의 집합)
- **Instance**: 문서 내 개별 필적 패치
- **Label**: 문서 수준 레이블만 제공 (개별 패치 레이블 불필요)

#### 혁신적 접근법
MIL은 "약지도 학습(Weakly Supervised Learning)"의 우아한 해결책을 제공합니다:
1. **라벨링 비용 절감**: 모든 패치에 레이블을 붙일 필요 없음
2. **현실적 적용**: 실제 상황에서 쉽게 구축 가능한 데이터
3. **전역-지역 동시 학습**: 문서 전체와 개별 패치를 동시에 분석

### 학술적 의의

#### 1. 새로운 문제 정의
- **기존**: "두 문서가 같은 작성자에 의해 작성되었는가?" (1:1 비교)
- **본 연구**: "한 문서 내에 복수 작성자가 존재하는가?" (문서 내부 분석)

#### 2. MIL의 창의적 적용
필기 분석 분야에 MIL을 최초로 적용하여, 문서 분석을 '집합 내 이질성 탐지' 문제로 재정의

#### 3. 약지도 학습의 도전
최소한의 레이블 정보로 복잡한 패턴을 학습하는 AI의 능력 한계를 탐구

### 사회적 영향과 장기 비전

#### 단기 목표 (1-2년)
- 법의학 현장에서 활용 가능한 수준의 정확도 달성
- 한국어 필기체 분석 표준 벤치마크 구축
- 실무자를 위한 사용자 친화적 인터페이스 개발

#### 중장기 확장 (3-5년)
1. **다국어 지원**: 영어, 중국어, 일본어 등으로 확장
2. **응용 분야 확대**:
   - AI 생성 텍스트와 인간 작성 텍스트 구분
   - 의료 기록 조작 탐지
   - 역사적 문서의 공동 저자 분석
3. **법적 표준화**: AI 기반 필적 분석의 법정 증거 채택 기준 수립

#### 사회적 파급효과
- **정의 실현**: 정교한 문서 위조 범죄 예방 및 탐지
- **접근성 민주화**: 저비용 자동 분석으로 개인도 문서 진위를 검증 가능
- **신뢰 구축**: 문서 기반 거래와 계약의 투명성 향상

### 선행 연구와의 관계

본 연구는 Kim et al.(2024)의 문서 간 비교 연구를 기반으로, 더 복잡한 문서 내부 분석으로 발전시킨 것입니다:

- **선행 연구**: 두 문서의 작성자 동일성 판별 (94.45% 정확도)
- **본 연구**: 한 문서 내 복수 작성자 존재 탐지 (더 어려운 문제)
- **시너지**: 두 기술을 결합하여 종합적인 문서 분석 시스템 구축 가능


### 연구 데이터셋 - CSAFE 영어 필기 데이터

#### 데이터셋 개요
- **출처**: CSAFE (Center for Statistics and Applications in Forensic Evidence)
- **목적**: 법의학 필기 분석 연구를 위한 표준 데이터셋
- **규모**: 300명 작성자, 351,311개 단어 이미지
- **언어**: 영어
- **특징**: 통제된 환경에서 체계적으로 수집된 고품질 필기 데이터

#### 데이터 구성
- **수집 프로토콜**: 3개 세션, 27회 반복 작성
- **텍스트 유형**: 
  - WOZ (Wizard of Oz): 문학 텍스트
  - LND (London Letter): 편지 형식
  - PHR (Phrase): 짧은 구문
- **전처리**: EasyOCR로 단어 단위 분할
- **파일 구조**: `{detected_word}_{repetition}.png`

#### MIL 적용을 위한 주요 특성
1. **계층적 구조**: 문서(반복) → Bag, 단어 → Instance
2. **충분한 샘플**: 작성자당 평균 1,171개 단어
3. **명확한 레이블**: 300명의 구분된 작성자 ID
4. **실제 환경 반영**: 다양한 필기 스타일과 자연스러운 변동성




### 모델 컨텍스트 프로토콜 (MCP/)
- **ClaudePoint**: 프로젝트 체크포인트 관리 및 복원
- **Gemini 통합**: Claude와 Google Gemini API 간의 브리지
- **구성 관리**: 중앙집중식 MCP 서버 구성

## 일반적인 명령어

### MIL 실험 실행 순서

#### 1. Autoencoder 기반 실험 (기존 방법)
```bash
# Step 1: 오토인코더 임베딩 추출
jupyter notebook MIL/experiments/autoencoder/mil_data_generator.ipynb

# Step 2: MIL Bag 데이터 생성
jupyter notebook MIL/experiments/autoencoder/mil_data_generator2.ipynb

# Step 3: Attention-Based MIL 학습
jupyter notebook MIL/experiments/autoencoder/AB_MIL_autoencoder_128d.ipynb
```



### MCP 도구
```bash
# ClaudePoint 초기화
claudepoint setup

# 프로젝트 체크포인트 생성
claudepoint create -d "실험 체크포인트"

# 사용 가능한 체크포인트 목록
claudepoint list

# 이전 상태로 복원
claudepoint restore <체크포인트-이름>
```

## MIL 프로젝트 파일 명명 규칙

### 모델 저장
- 학습된 모델: `MIL/output/models/`
  - Siamese 네트워크: `siamese_best_model.pth`
  - MIL 모델: `ab_mil_{embedding_type}_best_model.pt`
  - 예: `ab_mil_siamese_128d_best_model.pt`

### 데이터 파일
- 원본 데이터: `MIL/data/raw/`
  - `naver_ocr.csv`: 필기 이미지 메타데이터
  - `csafe_vit_300classes_best_model.pth`: 사전학습 모델
  
- 처리된 데이터: `MIL/data/processed/`
  - 임베딩: `embeddings/mil_{embedding_type}_{split}_data.csv`
  - Bag 데이터: `bags/{split}_bags_{embedding_type}.pkl`
  - 예: `train_bags_siamese_128d.pkl`

### 결과 파일
- 그래프: `MIL/output/figures/`
  - `siamese_training_history.png`
  - `roc_curve_siamese.png`
  - `siamese_tsne_visualization.png`
  
- 실험 결과: `MIL/output/results/`
  - `siamese_training_results.json`
  - `ab_mil_siamese_128d_results.json`

## 주요 개발 패턴

### MIL 모델 아키텍처



#### 2. Attention-Based MIL
```python
class ABMILModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super(ABMILModel, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # x: (batch_size, num_instances, input_dim)
        attention_weights = torch.softmax(self.attention(x), dim=1)
        bag_representation = torch.sum(attention_weights * x, dim=1)
        output = self.classifier(bag_representation)
        return output, attention_weights
```

### 데이터 로딩
```python
# 다중 데이터셋 연결 패턴
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 여러 데이터셋 변형 결합
combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])
```

### MIL 평가 지표



#### MIL 모델 평가
- **Bag 수준 정확도**: 단일/복수 작성자 분류 정확도
- **ROC-AUC**: 분류 성능의 종합적 평가
- **Attention 가중치**: 어떤 패치가 중요한지 시각화
- **F1 Score**: 정밀도와 재현율의 조화 평균

### 검증 및 테스트 방식

#### 검증 과정 (학습 중)
- **샘플링 방식**: `max_pairs=10000`으로 제한하여 빠른 검증
- **목적**: 학습 진행 상황 모니터링 및 조기 종료
- **속도 우선**: 전체 학습 시간 단축을 위한 근사치 사용

#### 최종 테스트 (학습 완료 후)
- **전체 쌍 비교**: 모든 가능한 작성자 쌍을 비교하여 정확한 성능 측정
- **목적**: 정확한 모델 성능 보고 및 논문 작성용 결과
- **정확도 우선**: 시간이 걸리더라도 정밀한 평가 수행
- **일반화 성능**: 검증-테스트 성능 차이로 오버피팅 여부 확인

## GPU 메모리 관리

대용량 모델과 데이터셋으로 인해:
- 적절한 배치 크기 사용 (일반적으로 64)
- 필요시 그래디언트 누적 구현
- 실험 간 CUDA 캐시 정리
- `torch.cuda.memory_summary()`로 메모리 사용량 모니터링

## 실험 재현성

모든 실험에서 결정론적 설정 사용:
```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## 선행 연구 논문

### Kim, Park, Carriquiry (2024) - 핵심 참고 연구
**논문**: "A deep learning approach for the comparison of handwritten documents using latent feature vectors"
**위치**: `/workspace/MIL/paper/(SAM) Kim_Park_Carriquiry_2024Jan.pdf`

#### 주요 기여 및 결과
- **최적 성능**: 부트스트래핑 + Vision Transformer + Autoencoder 조합으로 AUC 0.987, 정확도 94.45% 달성
- **Autoencoder > Siamese Network**: 특히 부트스트래핑과 결합시 Autoencoder가 우수한 성능
- **짧은 문서 처리**: 14단어 짧은 문서에서도 93.47% 정확도 유지
- **차원 압축**: 300차원 → 50차원으로 압축하여 효율적 특징 표현

#### 기술적 구조
- **전처리**: Simple Split, Text Detection, Bootstrapping 방법 비교
- **특징 추출**: ResNet, EfficientNet, Vision Transformer 비교 평가
- **잠재벡터 생성**: Autoencoder vs Siamese Network 비교
- **유사도 측정**: 코사인 유사도 기반 문서 간 비교

#### 현재 MIL 프로젝트와 비교
1. **방법론 차이**: 선행 연구는 Autoencoder 우수성 입증, 현재 연구는 Siamese Network 기반 MIL 접근
2. **데이터 처리**: 부트스트래핑 기법의 중요성 확인
3. **평가 방식**: Score-based Likelihood Ratio (SLR) 제공으로 법정 활용 가능성
4. **한계점**: 하이퍼파라미터 최적화 부족, 도구 의존성, 해석 가능성 문제

#### 향후 연구 방향 제안
- Autoencoder 기반 MIL 실험 재시도 고려
- 부트스트래핑 기법을 MIL Bag 생성에 적용
- Vision Transformer 기반 특징 추출 최적화
- 짧은 문서/패치에 대한 강건성 평가

## 중요한 설정 및 제약사항

### 데이터 범위 및 경로 설정
- **데이터 범위**: 실제 존재하는 이미지는 0~299 라벨 (300명 작성자) 전체 사용 가능
- **CSV 메타데이터**: 0~474 라벨까지 있지만, 300~474는 실제 이미지 파일 없음
- **기본 데이터 분할** (전체 300명 사용):
  - 훈련: 0~179 (180명)
  - 검증: 180~239 (60명)
  - 테스트: 240~299 (60명)
- **선택적 부분 데이터 분할** (필요시 100명만 사용):
  - 필터링 조건: `data['label'] >= 200` 사용 (200~299 범위, 100명 작성자)
  - 훈련: 200~259 (60명)
  - 검증: 260~279 (20명)
  - 테스트: 280~299 (20명)

### 경로 설정 주의사항
- **이미지 경로**: `/workspace/MIL/data/raw/csafe_version5_xai_train/`
- **메타데이터**: `/workspace/MIL/data/raw/naver_ocr.csv`
- **사전학습 모델**: `/workspace/MIL/data/raw/csafe_vit_300classes_best_model.pth`
- **결과 저장**: 
  - 모델: `/workspace/MIL/output/models/`
  - 그래프: `/workspace/MIL/output/figures/`
  - 결과: `/workspace/MIL/output/results/`

### 데이터 검증 필수사항
1. CSV 경로를 실제 압축 해제된 폴더 구조에 맞게 업데이트
2. 필터링 조건이 실제 존재하는 데이터 범위와 일치하는지 확인
3. 모든 경로는 절대 경로로 설정하여 실행 위치에 무관하게 동작


## 코딩 규칙

### 주석 작성
- 모든 코드 주석은 한국어로 작성
- 변수명과 함수명은 영어 유지, 설명은 한국어로

### 개발 환경
- **모든 코드 작성은 Jupyter Notebook(.ipynb) 파일로만 진행**
- Python 스크립트(.py) 파일 대신 노트북 형식 사용
- 셀 단위로 코드를 구성하여 실험과 분석 수행
- **.ipynb 파일 작업 시 전용 도구 사용**:
  - `NotebookRead`: 노트북 파일 읽기
  - `NotebookEdit`: 노트북 셀 편집/삽입/삭제
  - `mcp__ide__executeCode`: Jupyter 커널에서 코드 실행

### 멀티 GPU 학습 환경
- **DataParallel(DP) 사용**: Jupyter Notebook 환경에서는 `nn.DataParallel`만 사용
- **DistributedDataParallel(DDP) 제외**: `mp.spawn`을 사용하는 DDP는 Jupyter와 호환되지 않아 사용하지 않음
- **GPU 설정**: `CUDA_VISIBLE_DEVICES`로 특정 GPU만 선택하여 사용 (예: GPU 2,3,4)
- **배치 크기**: DataParallel이 자동으로 배치를 GPU 수만큼 분할하므로 전체 배치 크기 설정

## 서버 스펙 정보 (10GPU 서버)

### 하드웨어 사양
- **CPU**: Intel Xeon Gold 6226R @ 2.90GHz (2소켓, 32코어, 64스레드)
- **메모리**: 251GB RAM (183GB 가용)
- **GPU**: NVIDIA GeForce RTX 3090 × 5대 (각 24GB, 총 120GB)
- **CUDA**: 12.0 (드라이버), 11.8 (컴파일러)
- **스토리지**:
  - 시스템: 200GB 
  - /home: 3.3TB 
  - /data: 11TB 

### 딥러닝 최적화 가이드
- **배치 크기 권장** (RTX 3090 24GB 기준):
  - ResNet50: 256-512
  - BERT-base: 32-64
  - Vision Transformer: 128-256
- **멀티 GPU 활용**: DataParallel 또는 DistributedDataParallel 사용
- **공유 서버 가이드**: 4명 공유, 1인당 1-2 GPU 권장
- **스토리지 관리**: 대용량 데이터는 /data 디렉토리 활용

## Gemini CLI 사용법 - 대용량 코드베이스 분석

Claude의 컨텍스트 한계를 초과하는 대용량 파일이나 코드베이스 분석 시, Gemini CLI의 대용량 컨텍스트 윈도우를 활용합니다.

### 파일 및 디렉토리 포함 문법

`@` 문법을 사용하여 Gemini 프롬프트에 파일과 디렉토리를 포함시킵니다. 경로는 gemini 명령을 실행하는 위치에서의 상대 경로입니다:

#### 예시:

**단일 파일 분석:**
```bash
gemini -p "@src/main.py 이 파일의 목적과 구조를 설명하세요"
```

**여러 파일:**
```bash
gemini -p "@package.json @src/index.js 코드에서 사용된 의존성을 분석하세요"
```

**전체 디렉토리:**
```bash
gemini -p "@src/ 이 코드베이스의 아키텍처를 요약하세요"
```

**여러 디렉토리:**
```bash
gemini -p "@src/ @tests/ 소스 코드에 대한 테스트 커버리지를 분석하세요"
```

**현재 디렉토리와 하위 디렉토리:**
```bash
gemini -p "@./ 이 전체 프로젝트의 개요를 제공하세요"
# 또는 --all_files 플래그 사용:
gemini --all_files -p "프로젝트 구조와 의존성을 분석하세요"
```

### 구현 검증 예시

**기능 구현 확인:**
```bash
gemini -p "@src/ @lib/ 이 코드베이스에 다크 모드가 구현되어 있나요? 관련 파일과 함수를 보여주세요"
```

**인증 구현 검증:**
```bash
gemini -p "@src/ @middleware/ JWT 인증이 구현되어 있나요? 모든 인증 관련 엔드포인트와 미들웨어를 나열하세요"
```

**특정 패턴 검색:**
```bash
gemini -p "@src/ WebSocket 연결을 처리하는 React 훅이 있나요? 파일 경로와 함께 나열하세요"
```

**에러 처리 확인:**
```bash
gemini -p "@src/ @api/ 모든 API 엔드포인트에 적절한 에러 처리가 구현되어 있나요? try-catch 블록의 예시를 보여주세요"
```

**rate limiting 확인:**
```bash
gemini -p "@backend/ @middleware/ API에 rate limiting이 구현되어 있나요? 구현 세부사항을 보여주세요"
```

**캐싱 전략 검증:**
```bash
gemini -p "@src/ @lib/ @services/ Redis 캐싱이 구현되어 있나요? 모든 캐시 관련 함수와 사용법을 나열하세요"
```

**보안 조치 확인:**
```bash
gemini -p "@src/ @api/ SQL 인젝션 방어가 구현되어 있나요? 사용자 입력이 어떻게 sanitize되는지 보여주세요"
```

**기능별 테스트 커버리지 검증:**
```bash
gemini -p "@src/payment/ @tests/ 결제 처리 모듈이 완전히 테스트되어 있나요? 모든 테스트 케이스를 나열하세요"
```

### Gemini CLI 사용 시점

다음과 같은 경우 `gemini -p`를 사용하세요:
- 전체 코드베이스나 대규모 디렉토리 분석
- 여러 대용량 파일 비교
- 프로젝트 전체의 패턴이나 아키텍처 이해 필요
- 현재 컨텍스트 윈도우가 작업에 불충분한 경우
- 100KB 이상의 파일들을 다룰 때
- 특정 기능, 패턴, 또는 보안 조치의 구현 여부 확인
- 전체 코드베이스에서 특정 코딩 패턴의 존재 확인

### 중요 참고사항

- @ 문법의 경로는 gemini 명령을 실행하는 현재 작업 디렉토리 기준 상대 경로
- CLI가 파일 내용을 컨텍스트에 직접 포함
- 읽기 전용 분석에는 --yolo 플래그 불필요
- Gemini의 컨텍스트 윈도우는 Claude의 컨텍스트가 오버플로우될 전체 코드베이스를 처리 가능
- 구현 확인 시 찾고자 하는 내용을 구체적으로 명시하여 정확한 결과 획득
