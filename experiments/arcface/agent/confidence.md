
각 항목은 **목적 → 구현 포인트 → 코드 스니펫 → 합격 기준(acceptance)** 순서로 되어 있어.

---

# 0) 공통 유틸 (메트릭/시드/런너)

먼저 어디서든 재사용할 **공통 함수**부터 추가해 둬.

```python
# === common_utils.py 같은 곳에 두고 import 해도 좋음 ===
import os, random, numpy as np, torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_metrics_from_probs(y_true, probs, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    probs  = np.asarray(probs).astype(float)
    preds  = (probs >= thr).astype(int)
    acc = accuracy_score(y_true, preds)
    f1  = f1_score(y_true, preds, zero_division=0)
    prec= precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, probs) if len(set(y_true))>1 else 0.5
    cm  = confusion_matrix(y_true, preds, labels=[0,1])
    return dict(acc=acc, f1=f1, precision=prec, recall=rec, auc=auc, thr=thr, cm=cm)

def find_best_threshold(probs, labels, metric='f1'):
    # (이미 네 코드에 있지만 독립 실행용으로 재기재)
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    best_thr, best_val = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 37):
        preds = (probs >= thr).astype(int)
        if metric == 'accuracy':
            val = accuracy_score(labels, preds)
        elif metric == 'precision':
            val = precision_score(labels, preds, zero_division=0)
        elif metric == 'recall':
            val = recall_score(labels, preds, zero_division=0)
        else:
            val = f1_score(labels, preds, zero_division=0)
        if val > best_val:
            best_val, best_thr = val, thr
    return best_thr, best_val
```

> 아래 스니펫들은 네 Stage 3 코드(학습/평가 루프, Weighted BCE 등)가 이미 로드되어 있다는 가정이야.
> (즉, `train_model`, `evaluate`, `MILDataset`, `AttentionMIL`, `WeightedBCE` 등을 그대로 재사용)

---

## 1) 다중 시드 재현성 (Multi‑seed Reproducibility)

**목적**: 파이프라인 전체(또는 S3만)의 산포를 수치화. 평균±표준편차가 작으면 신뢰도↑.

**구현 포인트**

* **두 모드**를 모두 점검:
  (A) **S3‑only**: Stage 2 PKL 고정, S3만 seed를 바꿔 학습.
  (B) **S2+S3**: Stage 2도 seed로 재생성(완전 종단 재현성).
* 각 시드마다 **Val‑기반 최적 임계값**으로 Test 평가.

```python
# 재현성 실험 러너
def run_repro_experiment(seed_list=[42, 202, 777], s2_regen=False, s2_seed_key="MIL_S2_SEED"):
    results = []
    for s in seed_list:
        print(f"\n=== Repro Run (seed={s}, s2_regen={s2_regen}) ===")
        set_all_seeds(s)

        if s2_regen:
            # Stage 2를 시드 고정으로 다시 생성하는 경우(네 Stage 2 스크립트를 함수화했거나 CLI로 호출)
            # 예: os.environ[s2_seed_key] = str(s); stage2_generate(forgery_ratio=0.5, enforce_mix=True, stride2=True)
            os.environ[s2_seed_key] = str(s)
            # TODO: 네 환경에 맞는 Stage2 재생성 함수를 호출하거나, 노트북 셀을 함수화해서 호출

        # Stage 3 데이터 로드 (생성 or 기존 PKL)
        # ... (네 기존 로더 코드 재사용: train_loader, val_loader, test_loader 준비)

        # 모델/손실/옵티마 초기화
        mil_model = AttentionMIL(input_dim=256, hidden_dim=128, dropout_p=0.1).to(device)
        criterion = WeightedBCE(fp_weight=2.0)  # 네가 LHF에서 쓴 설정
        optimizer = torch.optim.Adam(mil_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

        # 학습
        mil_model = train_model(mil_model, optimizer, scheduler, train_loader, val_loader,
                                max_epochs=5, patience=3, name=f'attn_mil_seed{s}')

        # 평가(Val probs로 best thr 찾고 Test에 적용)
        _, _, val_auc, _, val_probs, val_labels, _ = evaluate(mil_model, val_loader)
        best_thr, _ = find_best_threshold(val_probs, val_labels, metric='f1')
        test_loss, _, test_auc, _, test_probs, test_labels, _ = evaluate(mil_model, test_loader)
        m = compute_metrics_from_probs(test_labels, test_probs, thr=best_thr)
        m.update(dict(seed=s, test_auc=test_auc, best_thr=best_thr))
        results.append(m)

        print(f"seed={s} → Acc={m['acc']:.4f}, F1={m['f1']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}, AUC={test_auc:.4f}, thr={best_thr:.3f}")

    # 요약
    import pandas as pd
    df = pd.DataFrame(results)
    summary = df[['acc','f1','precision','recall','test_auc']].agg(['mean','std']).T
    print("\n=== Repro Summary (S3-only or S2+S3) ===")
    print(summary)
    return df, summary
```

**합격 기준(권장)**

* S3‑only: `std(Acc)` ≤ **1.0%p**, `std(AUC)` ≤ **0.01**
* S2+S3: `std(Acc)` ≤ **1.5%p**, `std(AUC)` ≤ **0.015**

---

## 2) y‑Scramble 부정 대조 (Negative Control)

**목적**: 모델이 진짜 신호를 학습했는지(=랜덤 라벨이면 성능 0.5로 붕괴) 확인.

**구현 포인트**

* **학습은 정상**으로 하고, **테스트 평가 시 라벨만 섞어**서 AUC 계산.
* 또는 **학습 라벨을 섞고 학습**하는 변형도 추가(권장).

```python
def run_y_scramble_test(trained_model=None):
    # (전제) 이미 정상 학습된 모델과 test_loader가 있음
    model = trained_model
    model.eval()
    with torch.no_grad():
        test_logits, test_labels = [], []
        for X, y in test_loader:
            X = X.to(device)
            logit = model(X)[0]
            test_logits.append(logit.cpu())
            test_labels.append(y)
    test_logits = torch.cat(test_logits).numpy()
    test_labels = torch.cat(test_labels).numpy().astype(int)
    test_probs  = 1 / (1 + np.exp(-test_logits))

    # 라벨 섞기
    y_perm = np.random.permutation(test_labels)
    auc_perm = roc_auc_score(y_perm, test_probs)
    print(f"y‑scramble AUC (test labels permuted): {auc_perm:.4f}")
    return auc_perm
```

**합격 기준**

* y‑scramble AUC ≈ **0.5 ± 0.03** (유의하게 0.55 이상이면 경보)

---

## 3) 위조 비율 스트레스 테스트 (30% / 10% / 5%)

**목적**: 현실 난이도(부분 위조)로 내려갔을 때의 **성능 저하 곡선** 확보.

**구현 포인트**

* Stage 2 생성기에 **위조 비율 파라미터**를 추가(예: word\_mix에서 B‑토큰 수 nB=round(14\*ratio)).
* 동일한 Stage 3 설정(Weighted BCE + Val‑optimal thr)으로 학습·평가.
* 각 비율별 Test 성능을 테이블로 정리.

```python
def run_forgery_ratio_sweep(ratios=[0.3, 0.1, 0.05], seed=42):
    out = []
    for r in ratios:
        print(f"\n=== Forgery Ratio Sweep: ratio_B={r:.2f} (≈ {int(round(14*r))}/14 tokens) ===")
        set_all_seeds(seed)
        # 1) Stage 2 재생성 (ratio_B 반영) - 네 generate_bags 함수에 ratio 파라미터를 추가했거나 wrapper를 작성
        # stage2_generate(ratio_B=r, enforce_mix=True, stride2=True)  # TODO: 네 구현에 맞춰 호출

        # 2) Stage 3 로드, 학습, 임계값 최적화, 평가
        mil_model = AttentionMIL(256,128,0.1).to(device)
        criterion = WeightedBCE(fp_weight=2.0)
        optimizer = torch.optim.Adam(mil_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
        mil_model = train_model(mil_model, optimizer, scheduler, train_loader, val_loader, max_epochs=5, patience=3, name=f'attn_ratio_{r}')

        _, _, val_auc, _, val_probs, val_labels, _ = evaluate(mil_model, val_loader)
        thr, _ = find_best_threshold(val_probs, val_labels, metric='f1')
        _, _, test_auc, _, test_probs, test_labels, _ = evaluate(mil_model, test_loader)
        m = compute_metrics_from_probs(test_labels, test_probs, thr=thr)
        m.update({'ratio_B': r, 'test_auc': test_auc})
        out.append(m)
        print(f"ratio={r:.2f} → Acc={m['acc']:.3f}, F1={m['f1']:.3f}, P={m['precision']:.3f}, R={m['recall']:.3f}, AUC={test_auc:.3f}, thr={thr:.3f}")
    return out
```

**합격 기준(권장 목표)**

* 30%: **F1 ≥ 0.85**, AUC ≥ **0.95**
* 10%: **F1 ≥ 0.78**, AUC ≥ **0.90**
* 5% : **F1 ≥ 0.72**, AUC ≥ **0.85**
  (데이터 특성에 따라 다를 수 있으니, **완만한 하락 곡선**이면 OK)

---

## 4) 규칙 OOD 테스트 (Train: Gate‑ON, Test: Gate‑OFF)

**목적**: 합성 규칙(최소 혼합 K 보장)에 과적합하지 않았는지 확인.

**구현 포인트**

* **학습/검증**: 현재처럼 **혼합 보장(게이트 ON)**.
* **테스트**: 같은 작성자 분할로, **게이트 OFF**(혼합 최소 조건 비활성화, 경계만 랜덤)로 **다시 Stage 2 생성**.
* **학습 가중치 변경 없이** ON‑모델로 OFF‑테스트셋을 평가.

```python
def run_ood_gate_test(seed=42):
    set_all_seeds(seed)
    # 1) Train/Val: Gate-ON 데이터로 학습 완료(이미 완료된 best 모델 사용해도 OK)
    # mil_model = ... (학습 완료 모델 로드 or 재학습)

    # 2) Test: Gate-OFF 데이터 재생성
    # stage2_generate(ratio_B=0.5, enforce_mix=False, stride2=True)  # 게이트 끄기 (Min mixed 보장 OFF)

    # 3) 평가
    _, _, val_auc, _, val_probs, val_labels, _ = evaluate(mil_model, val_loader)
    thr, _ = find_best_threshold(val_probs, val_labels, metric='f1')

    test_loss, _, test_auc, _, test_probs, test_labels, _ = evaluate(mil_model, test_loader)
    m = compute_metrics_from_probs(test_labels, test_probs, thr=thr)
    print(f"[Gate-OFF Test] Acc={m['acc']:.3f}, F1={m['f1']:.3f}, P={m['precision']:.3f}, R={m['recall']:.3f}, AUC={test_auc:.3f}")
    return m, test_auc
```

**합격 기준(권장)**

* Gate‑OFF에서 **AUC 하락 ≤ 0.03**, **F1 하락 ≤ 5–8 pp**
* 임계값을 Val‑optimal로 고정했을 때도 **급추락 없음**

---

## 5) 분포‑피처 베이스라인 (μ/σ²/거리)

**목적**: “집합 통계만으로도” 분류가 어느 정도 되는지 확인 → **MIL의 추가 가치** 검증.

**구현 포인트**

* 각 bag의 \*\*인스턴스 특성(10×256)\*\*에서:
  `mu = mean(10,256)`, `var = var(10,256)`, `d_mean = mean pairwise distance(10×10)`
* X = `[mu | var | d_mean]` → **LogisticRegression**(또는 작은 MLP) 학습/평가.
* MIL과 **AUC 격차** 확인.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def bags_to_stats(features_list):
    # features_list: list of (10,256) np.array
    X = []
    for F in features_list:
        mu  = F.mean(axis=0)                # (256,)
        var = F.var(axis=0)                 # (256,)
        # pairwise 거리 평균 (유클리드)
        diffs = F[:,None,:] - F[None,:,:]   # (10,10,256)
        dists = np.sqrt((diffs**2).sum(axis=2))
        d_mean = dists[np.triu_indices(10,1)].mean()  # 상삼각 평균
        x = np.concatenate([mu, var, [d_mean]], axis=0)  # (513,)
        X.append(x.astype(np.float32))
    return np.vstack(X)  # (N, 513)

def run_distribution_baseline(train_features, train_labels, val_features, val_labels, test_features, test_labels):
    X_tr = bags_to_stats(train_features); y_tr = np.array(train_labels)
    X_va = bags_to_stats(val_features);   y_va = np.array(val_labels)
    X_te = bags_to_stats(test_features);  y_te = np.array(test_labels)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, n_jobs=1))
    ])
    clf.fit(X_tr, y_tr)

    # 임계값 최적화 (Val)
    val_probs = clf.predict_proba(X_va)[:,1]
    thr, _ = find_best_threshold(val_probs, y_va, metric='f1')

    test_probs = clf.predict_proba(X_te)[:,1]
    m = compute_metrics_from_probs(y_te, test_probs, thr=thr)
    m['auc'] = roc_auc_score(y_te, test_probs)
    print(f"[Dist-Feat Baseline] Acc={m['acc']:.3f}, F1={m['f1']:.3f}, P={m['precision']:.3f}, R={m['recall']:.3f}, AUC={m['auc']:.3f}, thr={thr:.3f}")
    return m
```

**합격 기준(권장)**

* MIL 대비 **AUC 격차 ≥ 0.03** (예: MIL 0.978 vs 분포‑피처 0.93 이하)
* 격차가 작으면(≤0.01) **합성 규칙에 대한 과적합** 또는 **분포‑피처가 거의 충분**일 수 있으니 원인 분석 필요.

---

# 빠른 실행 시나리오

실제로는 아래 순서로 한 번에 돌리면 좋아.

```python
# 1) 재현성
df_repro, summary_repro = run_repro_experiment(seed_list=[42,202,777], s2_regen=False)  # 먼저 S3-only
# 필요하면 s2_regen=True로 한 번 더

# 2) y-scramble (학습된 모델 하나로)
_ = run_y_scramble_test(trained_model=mil_model)  # mil_model은 위 실행에서 마지막 or best 로드를 사용

# 3) 위조 비율 스트레스
ratio_results = run_forgery_ratio_sweep([0.3, 0.1, 0.05], seed=42)

# 4) Gate OOD
ood_metrics, ood_auc = run_ood_gate_test(seed=42)

# 5) 분포-피처 베이스라인
_ = run_distribution_baseline(train_features, train_labels, val_features, val_labels, test_features, test_labels)
```

> **주의**
>
> * (3)(4)는 **Stage 2 재생성**이 필요하므로, 네 Stage 2 함수/노트북을 **함수화**해서 파라미터(비율, gate on/off)를 받을 수 있게 하자.
> * 모든 실험은 **작성자 단위 분할**을 절대 유지.
> * 결과는 CSV/MLflow/W\&B로 로깅해서 seed/ratio/gate 설정과 함께 보존.

---

## 기대 효과/판단 기준 총정리

* **재현성**: std가 낮으면 “운빨" 아님.
* **y‑scramble**: AUC≈0.5면 “진짜 신호” 학습.
* **스트레스(30/10/5%)**: “완만한 성능 하락 곡선"이면 일반화 기대.
* **Gate‑OFF**: 규칙 OOD에도 큰 붕괴 없으면 합성 규칙 과적합 아님.
* **분포‑피처와 격차**: MIL의 추가 가치 확인.

이대로 작업해보고, 로그/표를 가져오면 내가 **운영 포인트(threshold)와 손실 가중**까지 다시 최적화해서 **정확도·정밀도 방어** 쪽으로 더 밀어줄게.
