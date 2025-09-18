# MIL 프로젝트 개선 계획

## 현재 상황 요약
- **Autoencoder 접근**: 50.2% 정확도 (실패)
- **Siamese Network 접근**: 64.4% 정확도 (개선되었지만 부족)
- **핵심 문제**: 단일 패치 교체로 인한 미미한 True/False Bag 차이

## 개선 우선순위 및 구체적 실행 계획

### Phase 1: 즉시 개선 가능한 사항 (1-2일)

#### 1.1 Siamese Network 코드 개선
**파일**: `/workspace/MIL/experiments/siamese/train_siamese.ipynb`

- [ ] **최적 임계값 자동 탐지 구현**
  - `evaluate_accuracy` 함수 수정
  - ROC curve를 사용한 최적 threshold 찾기
  - F1 score, precision, recall 추가 계산

- [ ] **데이터셋 버그 수정**
  - 같은 이미지로 positive pair 생성 방지
  - 레이블당 이미지가 1개뿐인 경우 처리
  - 성능 최적화 (numpy array 사용)

- [ ] **학습 개선**
  - Learning rate scheduler 개선
  - Early stopping patience 조정
  - 더 많은 메트릭 로깅

**예상 성능 향상**: 64.4% → 68-70%

#### 1.2 데이터 증강 추가
- [ ] **필기체 특화 증강**
  - Elastic distortion
  - Random rotation (±15도)
  - Stroke width variation
  - Albumentations 라이브러리 활용

### Phase 2: 데이터 생성 전략 개선 (3-5일)

#### 2.1 오염 비율(Contamination Ratio) 다양화
**새 파일**: `/workspace/MIL/experiments/siamese/mil_data_generator2_siamese_improved.ipynb`

```python
# True Bag 생성 전략
contamination_ratios = {
    'easy': 0.5,      # 50% 교체
    'medium': 0.3,    # 30% 교체  
    'hard': 0.1,      # 10% 교체
    'very_hard': 0.05 # 5% 교체 (현재와 유사)
}
```

- [ ] **구현 계획**
  - 각 난이도별로 별도 데이터셋 생성
  - 커리큘럼 러닝: easy → hard 순서로 학습
  - 데이터셋 비율: easy(30%), medium(40%), hard(20%), very_hard(10%)

**예상 성능 향상**: 70% → 75-78%

#### 2.2 Hard Negative Mining
- [ ] **유사 작성자 쌍 찾기**
  - Siamese 임베딩으로 작성자 간 평균 거리 계산
  - 거리가 가까운 상위 10% 쌍 식별
  - 이들로 어려운 True Bag 생성

#### 2.3 복수 침입자 시나리오
- [ ] **다중 작성자 혼합**
  - 2명 혼합: 70%
  - 3명 혼합: 20%
  - 4명 혼합: 10%

### Phase 3: 사전 학습 특징 추출기 (5-7일)

#### 3.1 300명 작성자 분류 모델
**새 파일**: `/workspace/MIL/experiments/author_classifier/train_author_classifier.ipynb`

- [ ] **모델 아키텍처**
  - Base: ViT-B/16 (이미 있는 모델 활용)
  - Head: 300-class classification
  - Loss: CrossEntropy + Label Smoothing

- [ ] **학습 전략**
  - 전체 300명 데이터 사용
  - Strong augmentation
  - Mixup/CutMix 적용
  - 목표 정확도: 90%+

#### 3.2 특징 추출기 통합
- [ ] **MIL 파이프라인 수정**
  - 분류 모델의 penultimate layer 출력 사용
  - 768차원 → 128차원 projection
  - Freeze/Fine-tune 실험

**예상 성능 향상**: 75-78% → 80-85%

### Phase 4: MIL 모델 고도화 (7-10일)

#### 4.1 Attention 메커니즘 개선
**파일 수정**: `/workspace/MIL/experiments/siamese/AB_MIL_siamese_128d.ipynb`

- [ ] **Multi-head Attention**
  - 현재 single attention → 4-head attention
  - Position encoding 추가 (선택)

- [ ] **Gated Attention**
  - Attention gate 추가로 더 정교한 가중치 학습

#### 4.2 Loss Function 개선
- [ ] **Focal Loss 도입**
  - 어려운 샘플에 더 집중
  - Class imbalance 처리

- [ ] **Auxiliary Loss**
  - Instance-level 예측 추가
  - Multi-task learning

### Phase 5: 고급 실험 (10일+)

#### 5.1 Transformer 기반 MIL
- [ ] **Set Transformer 구현**
  - Bag을 set으로 처리
  - Permutation invariant

#### 5.2 대조 학습 (Contrastive Learning)
- [ ] **Bag-level contrastive loss**
  - SimCLR 스타일 적용
  - 더 풍부한 표현 학습

## 실험 추적 및 평가

### 메트릭 체계
```python
metrics = {
    'accuracy': float,
    'auc': float,
    'f1_score': float,
    'precision': float,
    'recall': float,
    'attention_entropy': float,  # Attention 분포의 엔트로피
    'false_positive_rate': float,
    'false_negative_rate': float
}
```

### 실험 버전 관리
- 각 실험마다 고유 ID 부여
- 하이퍼파라미터, 결과, 모델 체크포인트 저장
- Wandb 또는 TensorBoard 활용 고려

## 예상 일정 및 마일스톤

### Week 1 (Days 1-7)
- Phase 1 완료: Siamese 코드 개선
- Phase 2 시작: 데이터 생성 전략 구현
- **목표**: 70%+ 정확도 달성

### Week 2 (Days 8-14)  
- Phase 2 완료: 개선된 데이터 생성
- Phase 3 진행: 사전 학습 모델
- **목표**: 75%+ 정확도 달성

### Week 3 (Days 15-21)
- Phase 3 완료: 강력한 특징 추출기
- Phase 4 시작: MIL 모델 고도화
- **목표**: 80%+ 정확도 달성 (실용적 수준)

### Week 4+ (Days 22+)
- Phase 5: 고급 기법 실험
- 최종 모델 선정 및 최적화
- **목표**: 85%+ 정확도 (state-of-the-art)

## 리스크 및 대응 방안

### 리스크 1: 데이터 불균형
- **문제**: 작성자별 샘플 수 차이
- **대응**: Weighted sampling, SMOTE

### 리스크 2: 과적합
- **문제**: 복잡한 모델의 과적합
- **대응**: Dropout 증가, 정규화 강화, 더 많은 증강

### 리스크 3: 계산 자원 부족
- **문제**: GPU 메모리, 학습 시간
- **대응**: Gradient accumulation, Mixed precision training

## 성공 지표

1. **단기 (1주)**: 70% 정확도 돌파
2. **중기 (2-3주)**: 80% 정확도 달성 (실용적 수준)
3. **장기 (4주+)**: 85%+ 정확도, 논문 게재 가능 수준

## 다음 단계

1. **즉시 시작**: Phase 1.1 - Siamese Network 코드 개선
2. **병렬 진행**: 데이터 생성 전략 설계 문서 작성
3. **팀 리뷰**: 이 계획에 대한 피드백 수집