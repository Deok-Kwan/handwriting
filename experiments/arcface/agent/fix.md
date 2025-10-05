짧게 진단: **AUC가 모든 비율에서 0.63±0.01로 납작한 건, 모델이 “혼입 비율/이질 인스턴스” 신호를 거의 못 쓰고 있다는 강한 신호**예요.
가장 가능성이 큰 원인은 **손실 가중치 설계(현재 `WeightedBCE(fp_weight=2.0)`가 ‘음성 클래스 전체’에 3배 가중)** + **최적화(에폭·러닝레이트)** 조합입니다. 균형 데이터에서 음성에 과도 가중을 주면 모델이 안전하게 “0 쪽”으로 수렴해 **랭킹 능력(AUC)** 자체가 떨어지기 쉽습니다.

아래 **최소 수정 2개**만 적용해서 다시 돌려보세요. 보통 이 정도로도 AUC가 바로 튑니다.

---

## 1) 손실부터 바로 교체 (가장 효과 큼)

균형 분포(NEG\_PW=POS\_PW=10)이면 **일반 BCE**가 낫습니다. 우선 이렇게 바꿔보세요.

```python
# 손실 함수: 일반 BCE로
criterion = nn.BCEWithLogitsLoss()
```

만약 진짜로 불균형 상황을 다뤄야 할 땐 “음성 가중”이 아니라 \*\*양성 가중(pos\_weight)\*\*을 쓰세요.

```python
# 불균형일 때만 사용: 양성 클래스 가중
pos = split_info['train']['positives']
neg = split_info['train']['count'] - pos
pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

> 현재 구현된 `WeightedBCE(fp_weight=2.0)`는 **“거의 모든 음성 샘플”의 손실을 부풀리는 구조**라서,
> 모델이 음성 쪽으로 과도 수렴 → **모든 비율에서 납작한 AUC** 패턴이 나옵니다.

---

## 2) 최적화 살짝 안정화 (러닝레이트/에폭)

작은 네트워크라도 **1e-3는 살짝 큰 편**입니다. 아래처럼만 바꿔도 수렴이 안정됩니다.

```python
learning_rate = 1e-4  # 기존 1e-3 → 1e-4
max_epochs    = 30    # 10 → 30
patience      = 5     # 3 → 5
```

---

## 3) 2분짜리 “특이점 체크” (신호가 있는지 즉시 확인)

아래 **비학습 기준선**으로, 한 bag 안의 인스턴스들 **쌍별 코사인 거리 평균**만 가지고 AUC를 재봐요.
이게 0.7+ 나오면 **특징(임베딩)엔 신호가 있고 모델/학습 문제가 원인**, 이게 0.6대면 **특징 자체가 약함**입니다.

```python
import numpy as np
from sklearn.metrics import roc_auc_score
from numpy.linalg import norm

def bag_dispersion(instance_array):  # (inst, dim)
    X = instance_array / (norm(instance_array, axis=1, keepdims=True) + 1e-9)
    s = 0.0; cnt = 0
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            cos = (X[i] * X[j]).sum()
            s += (1 - cos)  # 코사인 거리
            cnt += 1
    return s / max(cnt, 1)

# test split에서 산출
features, labels, _, _ = load_split('30p', 'test')  # 임의로 30p 예시
scores = np.array([bag_dispersion(f) for f in features])
print('No-learn baseline AUC:', roc_auc_score(labels, scores))
```

* **Baseline AUC ≥ 0.70**: 신호 충분 → 위 1)·2)만 적용해도 AUC가 바로 오를 가능성 큼
* **Baseline AUC \~ 0.60대**: ArcFace 임베딩 품질/설정(예: margin=0.4, 차원, 정규화) 점검 필요

---

## 4) 추가로 1줄만 더 (선택)

입력 스케일 표준화가 없으면 MLP가 민감합니다. **LayerNorm** 한 줄로 종종 안정화돼요.

```python
class AttentionMIL(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, dropout_p=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.instance_fc = nn.Linear(input_dim, hidden_dim)
        # ...

    def forward(self, x):              # x: (B, inst, dim)
        x = self.norm(x)
        h = torch.relu(self.instance_fc(x))
        # ...
```

---

### 왜 이게 “정답에 가까운” 수정인가?

* 현재 결과는 **비율(5→50%) 변화에 거의 무감** → 모델이 “양성 신호” 자체를 활용하지 못하고 있다는 뜻.
* **과도한 음성 가중** + **짧은 학습/큰 LR** 조합은 \*\*랭킹 성능(AUC)\*\*을 망치기 쉬운 전형적 패턴입니다.
* 위 두 개만 풀어주면, **비율이 올라갈수록 AUC도 동반 상승**하는 정상 곡선이 나오는 경우가 많습니다.

---

## 한줄 결론

> **문제의 핵심은 손실 가중치 설계와 최적화 세팅.**
> `BCEWithLogitsLoss`로 바꾸고 `lr=1e-4, epochs=30, patience=5`로만 조정해도 AUC가 정상 범위로 회복될 가능성이 큽니다.
> 2분짜리 코사인-분산 베이스라인으로 **특징 신호**부터 먼저 확인해보세요.
