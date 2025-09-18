# MIL 연구 로드맵 - 복수 작성자 필기 문서 탐지 시스템

## 📋 프로젝트 개요

### 연구 목표
문서 내 복수 작성자 존재 여부를 자동으로 탐지하는 AI 시스템 개발

### 핵심 문제
- **입력**: 하나의 문서 (여러 필적 패치로 구성)
- **출력**: 단일 작성자 문서 vs 복수 작성자 문서 분류
- **도전과제**: 개별 패치의 작성자 정보 없이 문서 전체 수준에서 판별

### 현재 상황 (2025-06-28 업데이트)
- **모델**: AB-MIL with Autoencoder (128차원)
- **데이터**: CSAFE 영어 필기 데이터셋 (300명 중 200-299 라벨 사용)
- **최신 성능**: 
  - **정확도**: 55.28% (균형 데이터로 개선)
  - **AUC**: 0.5716 (기존 0.50에서 향상)
  - **F1 Score**: 0.6092
- **주요 개선사항**:
  - ✅ 슬라이딩 윈도우 (5개 단어) 기반 인스턴스 정의
  - ✅ 블록 단위 교체 (10개 윈도우)로 True Bag 생성
  - ✅ 클래스 균형 (1:1 비율) 달성
  - ✅ Siamese Network 실험 준비 완료 (노트북 수정 완료)
- **파일**: 
  - Autoencoder: `/workspace/MIL/experiments/autoencoder/AB_MIL_autoencoder_128d.ipynb`
  - Siamese: `/workspace/MIL/experiments/siamese/AB_MIL_siamese_128d.ipynb`

## 🗓️ 단계별 로드맵

### Phase 1: 문제 진단 및 데이터 검증 (1-2주) ✅ 완료

#### 1.1 데이터 품질 검증
- [x] MIL Bag 생성 로직 검증
  - [x] 단일/복수 작성자 Bag 균형 확인 (1:1 비율)
  - [x] Bag 내 인스턴스 수 분포 분석 (평균 57-60개)
  - [x] 메타데이터 정확성 검증 (교체 비율 1.7% 문제 발견)
- [x] 특징 벡터 품질 확인
  - [x] Autoencoder 임베딩 분석 (재구성 손실 미사용 문제)
  - [ ] t-SNE로 작성자별 클러스터링 시각화
  - [ ] 이상치(outlier) 탐지

#### 1.2 인스턴스 정의 개선 (긴급) ✅ 완료
- [x] 슬라이딩 윈도우 기반 인스턴스 재정의
  - [x] 단일 단어 → 5개 연속 단어 묶음
  - [x] 윈도우 크기 5개 적용 (stride=1)
  - [x] 겹침 정도(stride) 최적화
- [x] 블록 단위 교체 구현
  - [x] 개선된 process_true_bag 함수로 블록 교체
  - [x] 치명적 버그 수정 (empty label list 처리)
  - [x] 메타데이터 포함한 상세 정보 저장

#### 1.3 모델 구조 분석 ✅ 완료
- [x] Attention 메커니즘 동작 확인 (AB-MIL 구현 완료)
  - [x] Attention-based MIL 모델 구현
  - [x] Mixed Precision Training 적용
  - [x] 자원 최적화 (배치 크기 128, 멀티프로세싱)
- [x] 초기 성능 검증 (AUC 0.5716 달성)


### Phase 2: 베이스라인 개선 (2-3주) 🔄 진행중

#### 2.1 데이터 전처리 개선
- [x] Bag 생성 전략 재설계 ✅ 완료
  - [x] 슬라이딩 윈도우 기반 인스턴스 (5개 단어)
  - [x] 블록 단위 교체 (10개 윈도우 블록)
  - [x] 클래스 균형 달성 (1:1 비율)
- [ ] 추가 개선 사항
  - [ ] 다양한 블록 크기 실험 (5, 15, 20개 윈도우)
  - [ ] 교체 비율 조정 (현재 30% → 20%, 40%)
  - [ ] Hard negative mining 적용

#### 2.2 모델 아키텍처 개선
- [ ] Attention 메커니즘 변형
  - [ ] Multi-head attention 적용
  - [ ] Gated attention 구현
  - [ ] Self-attention 추가
- [ ] 분류기 구조 개선
  - [ ] 깊은 MLP 구조 실험
  - [ ] Residual connection 추가
  - [ ] Dropout 및 정규화 강화

#### 2.3 학습 전략 최적화
- [ ] 손실 함수 개선
  - [ ] Focal loss 적용
  - [ ] Class weight balancing
  - [ ] Contrastive loss 추가
- [ ] 학습률 스케줄링
  - [ ] Warm-up 적용
  - [ ] Cosine annealing
  - [ ] Cyclic learning rate

### Phase 3: Siamese Network 및 Metric Learning 통합 (3-4주)

#### 3.1 Siamese 기반 특징 추출
- [ ] Siamese network 학습 개선
  - [ ] Triplet loss vs Contrastive loss 비교
  - [ ] Hard negative mining 강화
  - [ ] 온라인 배치 생성 최적화
- [ ] 임베딩 품질 평가
  - [ ] 작성자 내/간 거리 분포 분석
  - [ ] Embedding space 시각화

#### 3.2 ArcFace 기반 Metric Learning 📐
- [ ] ArcFace 모델 구현 (`/workspace/MIL/experiments/arcface/`)
  - [ ] `train_arcface.ipynb` 생성
  - [ ] Angular margin loss 구현 (margin=0.5, scale=64)
  - [ ] 100명 작성자 분류 (label 200-299)
- [ ] ArcFace 특징 추출기 설계
  - [ ] 사전학습 ViT (`csafe_vit_300classes_best_model.pth`) 활용
  - [ ] 300 → 128차원 projection head
  - [ ] L2 정규화 + Angular margin
- [ ] 임베딩 추출 파이프라인
  - [ ] `mil_data_generator_arcface.ipynb` 생성
  - [ ] 단어별 128차원 ArcFace 임베딩 추출
  - [ ] CSV 저장 (기존 형식 유지)
- [ ] MIL 통합
  - [ ] `mil_data_generator2_arcface.ipynb` - Bag 생성
  - [ ] `AB_MIL_arcface_128d.ipynb` - MIL 학습
  - [ ] 윈도우/블록 교체 적용
- [ ] 성능 비교 분석
  - [ ] Autoencoder vs Siamese vs ArcFace
  - [ ] 작성자 검증(verification) 정확도
  - [ ] t-SNE 시각화로 특징 공간 분석

#### 3.3 Metric Learning-MIL 통합
- [ ] 특징 통합 전략
  - [ ] Autoencoder + Siamese + ArcFace 앙상블
  - [ ] 특징 연결(concatenation)
  - [ ] 특징 융합(fusion) 네트워크
  - [ ] Weighted combination 실험
- [ ] End-to-end 학습
  - [ ] Metric Learning + MIL 동시 학습
  - [ ] Multi-task learning 프레임워크
  - [ ] Joint optimization 전략

### Phase 4: 고급 MIL 기법 (4-5주)

#### 4.1 최신 MIL 알고리즘
- [ ] DSMIL (Dual-stream MIL) 구현
- [ ] TransMIL (Transformer-based MIL) 적용
- [ ] CLAM (Clustering-constrained Attention MIL)
- [ ] ProtoMIL (Prototype-based MIL)

#### 4.2 Self-supervised 학습
- [ ] SimCLR/MoCo를 이용한 사전학습
- [ ] Masked autoencoding 적용
- [ ] Contrastive MIL 구현

### Phase 5: 도메인 확장 (5-6주)

#### 5.1 한국어 필기 데이터 적용
- [ ] 한국어 데이터셋 구축
  - [ ] 데이터 수집 및 라벨링
  - [ ] 전처리 파이프라인 구축
- [ ] Cross-lingual transfer learning
  - [ ] 영어 모델 fine-tuning
  - [ ] Domain adaptation 기법

#### 5.2 실제 문서 시나리오
- [ ] 긴 문서 처리
  - [ ] 계층적 MIL 구조
  - [ ] Sliding window 접근법
- [ ] 다양한 문서 유형
  - [ ] 계약서, 유언장, 편지 등
  - [ ] 도메인별 특화 모델

### Phase 6: 시스템 통합 및 배포 (6-7주)

#### 6.1 성능 최적화
- [ ] 모델 경량화
  - [ ] Knowledge distillation
  - [ ] Pruning & Quantization
- [ ] 추론 속도 개선
  - [ ] ONNX 변환
  - [ ] TensorRT 최적화

#### 6.2 사용자 인터페이스
- [ ] Web 기반 데모 시스템
- [ ] API 서버 구축
- [ ] 시각화 대시보드
  - [ ] Attention heatmap
  - [ ] 신뢰도 점수
  - [ ] 의심 영역 하이라이트

## 📊 평가 지표 및 목표

### 주요 지표
1. **Bag 수준 정확도**: 85% 이상
2. **ROC-AUC**: 0.90 이상
3. **F1 Score**: 0.80 이상
4. **False Positive Rate**: 10% 이하
5. **추론 시간**: 문서당 1초 이내

### 검증 방법
- K-fold cross validation
- 독립적인 테스트셋 평가
- 실제 문서 사례 연구
- 전문가 평가와 비교

## 🔧 실험 관리

### 실험 추적
- [ ] MLflow/Weights & Biases 설정
- [ ] 하이퍼파라미터 로깅
- [ ] 모델 버전 관리
- [ ] 결과 자동 리포팅

### 재현성 확보
- [ ] Random seed 고정
- [ ] 환경 설정 문서화
- [ ] Docker 컨테이너 구축
- [ ] 코드 버전 관리

## 📚 참고 문헌 및 리소스

### 핵심 논문
1. Kim, Park, Carriquiry (2024) - "A deep learning approach for handwritten document comparison"
2. Ilse et al. (2018) - "Attention-based Deep Multiple Instance Learning"
3. Li et al. (2021) - "Dual-stream Multiple Instance Learning"
4. Shao et al. (2021) - "TransMIL: Transformer based Correlated Multiple Instance Learning"
5. Deng et al. (2019) - "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
6. Wang et al. (2018) - "CosFace: Large Margin Cosine Loss for Deep Face Recognition"

### 기술 스택
- **Framework**: PyTorch 2.0+
- **Vision Models**: timm, torchvision
- **실험 관리**: MLflow, Weights & Biases
- **시각화**: matplotlib, seaborn, plotly
- **배포**: FastAPI, Docker, Kubernetes

## 🚀 즉시 실행 가능한 개선 사항 (우선순위)

### 1. Siamese Network 기반 MIL 실험 (최우선) 🔥
**실행 순서**:
1. `/workspace/MIL/experiments/siamese/mil_data_generator_siamese.ipynb` - 임베딩 추출
2. `/workspace/MIL/experiments/siamese/mil_data_generator2_siamese.ipynb` - 균형 Bag 생성
3. `/workspace/MIL/experiments/siamese/AB_MIL_siamese_128d.ipynb` - MIL 학습

**기대 효과**: 
- 작성자 구분에 특화된 임베딩으로 AUC 0.70 이상 목표
- 이미 학습된 `siamese_improved_best_model.pth` 활용

### 2. Autoencoder 추가 개선
**개선 방안**:
- 블록 크기 다양화: [5, 10, 15, 20] 실험
- 교체 비율 조정: 20%, 30%, 40% 비교
- Focal Loss 적용으로 어려운 샘플 집중 학습

### 3. ArcFace 구현 (다음 단계)
**생성 파일 순서**:
1. `/workspace/MIL/experiments/arcface/train_arcface.ipynb`
2. `/workspace/MIL/experiments/arcface/mil_data_generator_arcface.ipynb`
3. `/workspace/MIL/experiments/arcface/mil_data_generator2_arcface.ipynb`
4. `/workspace/MIL/experiments/arcface/AB_MIL_arcface_128d.ipynb`

## 🎯 수정된 마일스톤

### M1 (1주차): 긴급 문제 해결 ✅ 완료
- [x] 데이터 검증 완료 (교체 비율 1.7% 문제 발견)
- [x] 인스턴스 정의 개선 구현 (슬라이딩 윈도우 5개)
- [x] 클래스 균형 달성 (1:1 비율)
- [x] 성능 향상 달성 (AUC 0.50 → 0.5716)

### M2 (2-3주차): 베이스라인 개선 🔄 진행중
- [x] 개선된 Bag 생성으로 재실험 완료
- [x] Autoencoder 기반 균형 데이터 학습 (AUC 0.5716)
- [x] Siamese Network 노트북 수정 완료
- [ ] **다음 단계: Siamese Network 기반 MIL 실험 실행** 🎯
  - [ ] 기존 Siamese 모델로 임베딩 추출
  - [ ] 균형 잡힌 MIL Bag 생성
  - [ ] AB-MIL 학습 및 평가
- [ ] 목표: AUC 0.70 이상 달성

### M3 (4-5주차): 고급 특징 추출기
- [ ] ArcFace 구현 및 학습
- [ ] 3가지 임베딩 비교 (Autoencoder, Siamese, ArcFace)
- [ ] 목표: AUC 0.80 이상 달성

### M4 (6-7주차): 고급 MIL 기법
- [ ] TransMIL 등 최신 알고리즘 적용
- [ ] 앙상블 방법 구현
- [ ] 정확도 85% 달성

### M5 (8-10주차): 실용화 및 논문
- [ ] 한국어 데이터 적용
- [ ] 웹 데모 구축
- [ ] 논문 작성

## 🔧 주요 개발 패턴

### 1. ArcFace 기반 특징 추출기
```python
class ArcFaceModel(nn.Module):
    def __init__(self, base_model, embedding_dim=128, num_classes=300, margin=0.5, scale=64):
        super(ArcFaceModel, self).__init__()
        self.base_model = base_model  # 사전학습된 ViT
        self.embedding_layer = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
    
    def forward(self, x, labels=None):
        # 특징 추출
        x = self.base_model(x)
        embeddings = self.embedding_layer(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        if labels is None:
            return embeddings
        
        # ArcFace 손실 계산
        cosine = F.linear(embeddings, F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output, embeddings
```

### 2. ArcFace 활용 전략
- **사전학습**: 300명 작성자 데이터로 ArcFace 모델 학습
- **특징 추출**: 학습된 모델로 고품질 임베딩 생성
- **MIL 통합**: ArcFace 임베딩을 MIL의 인스턴스 특징으로 활용
- **앙상블**: Siamese, Autoencoder, ArcFace 특징 결합

## 💡 위험 요소 및 대응 방안

### 기술적 위험
1. **데이터 부족**: 데이터 증강 및 합성 데이터 생성
2. **과적합**: 정규화 강화 및 앙상블 기법
3. **계산 자원**: 분산 학습 및 모델 경량화

### 연구적 위험
1. **새로운 접근법 실패**: 다양한 백업 방법론 준비
2. **평가 지표 논란**: 다각도 평가 및 전문가 검증
3. **일반화 문제**: 다양한 도메인 데이터 확보

## 📝 주간 체크리스트

### 매주 월요일
- [ ] 지난 주 실험 결과 정리
- [ ] 이번 주 목표 설정
- [ ] 코드 리뷰 및 리팩토링

### 매주 수요일
- [ ] 실험 중간 점검
- [ ] 문제점 분석 및 대응

### 매주 금요일
- [ ] 주간 보고서 작성
- [ ] 다음 주 계획 수립
- [ ] 논문 리뷰 (최소 1편)

---

**Last Updated**: 2025-06-28
**Author**: MIL Research Team
**Version**: 1.5
**Change Log**: 
- v1.5 (2025-06-28): Siamese Network 실험 노트북 수정 완료, 실행 준비 완료
- v1.4 (2025-06-27): Autoencoder 균형 데이터 실험 완료 (AUC 0.5716), Siamese 실험을 다음 우선순위로 설정
- v1.3 (2025-06-27): Phase 1 완료, M1 마일스톤 달성, Phase 2 진행 상황 업데이트
- v1.2 (2025-06-27): 인스턴스 정의 개선 방안 추가, 우선순위 섹션 신설, Phase 1 진행상황 업데이트
- v1.1 (2025-06-26): ArcFace 기반 metric learning 추가
- v1.0 (2025-06-25): 초기 로드맵 작성