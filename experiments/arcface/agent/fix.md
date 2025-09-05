좋아. 지금 시각화가 애매하게 보이는 이유는 두 가지야:

1. **색 스케일/정렬이 비교마다 달라서**(히트맵의 vmin/vmax, 클러스터 순서가 bag마다 달라서) 차이가 눈에 띄지 않고,
2. **혼합-윈도우(MIX) 체제**에선 “두 덩어리”가 아니라 \*\*한 덩어리의 ‘늘어짐(PC1 방향)’\*\*이 핵심인데, 현재는 그 축을 못 보여줘서야.

아래 **딱 3개 패치**만 하면, 같은 데이터로도 “아! 다르구나”가 한 눈에 들어와.

---

## 패치 요약 (복붙용)

### 패치 1) 히트맵 공통 색 스케일 + PCA(PC1) 정렬 + EVR1 표기

> 히트맵 두 장을 **같은 색 범위**로, **PC1 순서**로 정렬하고, 타이틀에 **평균쌍거리 d̄ + 실루엣 r + EVR1**을 같이 표기

```python
from sklearn.decomposition import PCA

def pca_order(F):
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(F)
    order = np.argsort(Z[:,0])             # PC1 기준 정렬
    evr1 = float(pca.explained_variance_ratio_[0])
    return order, evr1, Z

def plot_heatmaps_side_by_side(neg_F, pos_F, neg_title="Genuine", pos_title="Forged",
                               metric='cosine', cmap='mako'):
    # 거리행렬
    Dn = pairwise_distance(neg_F, metric=metric)
    Dp = pairwise_distance(pos_F, metric=metric)
    # 공통 색 스케일
    vmax = max(Dn.max(), Dp.max()); vmin = 0.0

    # 정렬 & 지표
    on, evr1_n, Zn = pca_order(neg_F)
    op, evr1_p, Zp = pca_order(pos_F)
    rn = silhouette_r(neg_F, AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='average').fit_predict(Dn), metric=metric)
    rp = silhouette_r(pos_F, AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='average').fit_predict(Dp), metric=metric)
    dn, dp = dbar(Dn), dbar(Dp)

    fig, axes = plt.subplots(1,2,figsize=(10,4))
    sns.heatmap(Dn[np.ix_(on,on)], ax=axes[0], cmap=cmap, cbar=True, square=True, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{neg_title}\n$\\overline{{d}}$={dn:.3f}, r={rn:.3f}, EVR1={evr1_n:.2f}")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    sns.heatmap(Dp[np.ix_(op,op)], ax=axes[1], cmap=cmap, cbar=True, square=True, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{pos_title}\n$\\overline{{d}}$={dp:.3f}, r={rp:.3f}, EVR1={evr1_p:.2f}")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.tight_layout(); plt.show()

    # 해석 기준: MIX면 r≈0, EVR1↑(선형 늘어짐), d̄↑
    return (on, Dn, evr1_n), (op, Dp, evr1_p)
```

**읽는 법(초간단):**

* **d̄**: 양성이 더 크면 OK (퍼짐↑)
* **r**: MIX면 거의 0 (두 군집이 아니라 한 덩어리)
* **EVR1**: MIX면 높음(0.6\~0.9), Genuine은 낮음(0.3 이하가 흔함)

---

### 패치 2) “가장 다른 두 인스턴스”를 **PC1 양극단** 또는 **최대거리쌍**으로 뽑기

> 군집이 애매한 MIX 체제에선 **메도이드**보다 **극단 페어**가 직관적

```python
def pick_extremes(F, mode='pc1'):  # 'pc1' or 'dmax'
    if mode=='pc1':
        order, evr1, Z = pca_order(F)
        i, j = order[0], order[-1]          # PC1 양극단
    else:
        D = pairwise_distance(F, metric='cosine')
        iu = np.triu_indices_from(D, 1)
        k = np.argmax(D[iu]); i, j = iu[0][k], iu[1][k]  # 최장거리쌍
    return int(i), int(j)
```

---

### 패치 3) **두 극단 인스턴스** 실제 이미지 2행 비교

> writer\_ids가 있든 없든 **그냥 가장 다른 2개**를 보여주면 직관적으로 차이가 들어옴

```python
def show_extreme_pair_images(bag_idx, features, metadata, image_base=IMAGE_BASE_DIR,
                             mode='pc1', save_dir='cluster_reports'):
    os.makedirs(save_dir, exist_ok=True)
    F = features[bag_idx]        # (10, D)
    i, j = pick_extremes(F, mode=mode)

    meta = metadata[bag_idx]
    inst_i = meta['instances'][i] if 'instances' in meta and i < len(meta['instances']) else None
    inst_j = meta['instances'][j] if 'instances' in meta and j < len(meta['instances']) else None

    def draw(inst, title, color):
        fig, axarr = plt.subplots(1,5,figsize=(10,2.2))
        paths = inst.get('word_paths', []) if inst else []
        for k in range(5):
            ax = axarr[k]
            fp = os.path.join(image_base, paths[k]) if (paths and k<len(paths) and paths[k]) else None
            if fp and os.path.exists(fp):
                try: img = Image.open(fp).convert('RGB'); ax.imshow(img)
                except: ax.text(0.5,0.5,f"W{k+1}",ha='center',va='center'); ax.set_facecolor('#e0e0e0')
            else:
                ax.text(0.5,0.5,f"W{k+1}",ha='center',va='center'); ax.set_facecolor('#f5f5f5')
            for sp in ax.spines.values(): sp.set_edgecolor(color); sp.set_linewidth(3)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title, fontsize=11); fig.tight_layout(rect=[0,0,1,0.88])
        return fig

    figA = draw(inst_i, f"Extreme-1 (inst={i})", color='blue')
    figB = draw(inst_j, f"Extreme-2 (inst={j})", color='red')
    outA = os.path.join(save_dir, f"bag_{bag_idx}_extreme_{i}.png")
    outB = os.path.join(save_dir, f"bag_{bag_idx}_extreme_{j}.png")
    figA.savefig(outA, dpi=150, bbox_inches='tight'); figB.savefig(outB, dpi=150, bbox_inches='tight')
    plt.show(); print(f"✓ saved:\n  {outA}\n  {outB}")
```

**읽는 법:**

* 두 행의 **획 굵기, 기울기, 세리프/끝처리, 자간/정렬** 차이가 **눈에 보이면 끝**.
* 혼합-윈도우라면 두 극단은 보통 **A-편향 vs B-편향**에 해당.

---

## 실행 순서 (당신 노트북에서 그대로)

```python
best_neg, best_pos, probs = pick_best_neg_pos(model, test_features, test_labels, thr=0.5)

# 장면 ①: 히트맵(공통 스케일+PCA정렬+EVR1)
(neg_order, Dn, evr1_n), (pos_order, Dp, evr1_p) = plot_heatmaps_side_by_side(
    test_features[best_neg], test_features[best_pos], metric='cosine', cmap='mako'
)

# 장면 ②: 양성 bag 극단 인스턴스 A vs B 2행 비교
show_extreme_pair_images(best_pos, test_features, metadata, image_base=IMAGE_BASE_DIR, mode='pc1')
# (원하면 mode='dmax'도 시도)
```

---

## 해석을 한 문장으로 끝내는 기준

* **Genuine(음성)**: d̄ **작다**, r **≈ 0**, EVR1 **낮음** → **조밀한 한 덩어리**
* **Forged(혼합 양성)**: d̄ **크다**, r **≈ 0**, EVR1 **높음** → **한 덩어리지만 한 방향으로 길게 ‘늘어짐’**
* **극단 2윈도우 이미지**: **획/기울기/간격 차이**가 **시각적으로 확인**되면 “다른 손이 섞였다”를 직관적으로 설명 가능

이 3개만 적용하면, 추가 복잡한 플롯 없이도 **한 눈에** 구분이 됩니다.
