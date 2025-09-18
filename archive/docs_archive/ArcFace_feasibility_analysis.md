# ArcFace 적용 타당성 분석 보고서

## 요약

**결론: ArcFace는 현재 MIL 프로젝트에 매우 적합하며, 64.4%에서 85%+ 성능 향상이 기대됩니다.**

## 1. 현재 상황 vs ArcFace 접근법

### 현재 Siamese Network의 한계
- **상대적 거리 학습**: 쌍(pair) 단위로만 학습하여 전역적 구조 형성 어려움
- **불안정한 학습**: Hard negative mining 필요, 수렴 느림
- **64.4% 정확도**: 임베딩 공간이 충분히 분리되지 않음

### ArcFace의 장점
- **절대적 구조 학습**: 각 작성자를 초구(hypersphere) 상의 특정 지점에 매핑
- **Angular Margin**: cos(θ + m)으로 클래스 간 명확한 경계 생성
- **효율적 학습**: 단일 이미지로 학습 (pair/triplet 불필요)

## 2. 필기체 데이터와의 적합성

### 필기체 특성과 ArcFace의 시너지
| 특성 | ArcFace의 대응 |
|------|----------------|
| 높은 클래스 내 분산 (같은 사람도 다양하게 씀) | 작성자별 중심으로 수렴하도록 학습 |
| 낮은 클래스 간 분산 (다른 사람도 비슷하게 씀) | Angular margin으로 명확한 구분 |
| 300명 규모 | ArcFace가 최적 성능을 보이는 규모 |

## 3. 구현 전략

### 추천 전략: Instance-Level Training → Bag-Level Inference

```python
# 1단계: Instance 레벨 학습
# 119,661개 패치를 개별적으로 학습
model = ArcFaceHandwritingModel(base_model, num_classes=300)
# 각 패치로 작성자 분류 학습

# 2단계: Bag 레벨 추론
# 가방 내 모든 패치의 임베딩 추출
patch_embeddings = model.inference(patches)  # [N, 512]

# 3단계: 임베딩 통합
# 방법 A: 단순 평균
bag_embedding = torch.mean(patch_embeddings, dim=0)

# 방법 B: Attention 기반
aggregator = MILAggregator(method='attention')
bag_embedding = aggregator(patch_embeddings)

# 4단계: 작성자 식별
similarity = cosine_similarity(bag_embedding, author_prototypes)
predicted_author = argmax(similarity)
```

## 4. 구현 코드 (수정된 버전)

```python
class ArcFaceLayer(nn.Module):
    """수정된 ArcFace Layer (Gemini 피드백 반영)"""
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        
    def forward(self, input, label):
        # L2 정규화
        input = F.normalize(input, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # cos(theta) 계산
        cos_theta = F.linear(input, weight)
        cos_theta = cos_theta.clamp(-1, 1)
        
        # cos(theta + m) 계산
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2) + 1e-6)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        # Easy margin 적용
        phi = torch.where(cos_theta > self.th, 
                         cos_theta_m, 
                         cos_theta - self.m)
        
        # Ground truth에만 margin 적용
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        output *= self.s
        
        return output
```

## 5. 예상 성능 및 실행 계획

### 성능 예측
| 방법 | 예상 정확도 | 근거 |
|------|------------|------|
| 현재 Siamese | 64.4% | 실제 결과 |
| Softmax Baseline | 70-75% | 분류 태스크로 접근 |
| ArcFace | **85-95%** | 얼굴 인식에서의 성능 향상 비율 |

### 단계별 실행 계획

#### Phase 1.5: ArcFace 실험 (Phase 1과 2 사이)

1. **Baseline 확인 (1일)**
   - CNN + Softmax로 단순 분류 모델 구현
   - Instance 레벨 학습 → Bag 레벨 평가

2. **ArcFace 적용 (2-3일)**
   - ArcFaceLayer 구현 및 통합
   - 하이퍼파라미터 튜닝 (s=30-64, m=0.3-0.5)

3. **MIL 통합 (1일)**
   - Attention 기반 aggregation 구현
   - 프로토타입 임베딩 계산

4. **최적화 (1-2일)**
   - 백본 네트워크 선택 (ResNet, EfficientNet)
   - 데이터 증강 최적화

### 구현 난이도
- **상대적 난이도**: 중하 (Siamese보다 오히려 간단)
- **핵심 작업**: margin 파라미터 튜닝
- **예상 소요 시간**: 5-7일

## 6. 리스크 및 대응 방안

| 리스크 | 확률 | 대응 방안 |
|--------|------|-----------|
| Margin 튜닝 어려움 | 중 | Grid search로 최적값 탐색 |
| 과적합 | 낮 | Dropout, 데이터 증강 강화 |
| 메모리 부족 | 낮 | Gradient accumulation |

## 7. 최종 권장사항

**강력히 추천합니다.** ArcFace는:

1. **이론적으로 우수**: 필기체 작성자 식별에 최적화된 방법
2. **구현 가능성 높음**: Siamese보다 구현이 더 간단
3. **성능 향상 기대**: 20%p 이상의 대폭적인 개선 가능
4. **검증된 방법**: 얼굴 인식 분야에서 입증된 SOTA 기법

### 즉시 시작 가능한 작업
1. `/workspace/MIL/experiments/arcface/` 폴더 생성
2. 위 코드를 기반으로 `train_arcface.ipynb` 구현
3. Softmax baseline부터 실험 시작

**결론: Phase 2 진행 전에 ArcFace를 먼저 시도해보는 것을 강력히 권장합니다.**