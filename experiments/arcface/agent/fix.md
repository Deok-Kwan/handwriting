아래는 **baseline 코드**를 그대로 재사용하면서, 두 개의 실험 파일을 **정확한 구조**로 설계한안입니다.

* 파일명:

  * `experiments/arcface/agent/stage3_transformer_mil.ipynb` (또는 `.py`)
  * `experiments/arcface/agent/stage3_transmil.ipynb` (또는 `.py`)
* 핵심 원칙: **데이터 로딩/학습·평가 루프/손실/임계값 탐색**은 baseline을 최대한 재사용하고, **모델 정의 부분만 교체**합니다. 두 모델 모두 **입력/출력 인터페이스를 baseline(AttentionMIL)과 동일**하게 맞춰 `forward`에서 `(logits, weights)`를 반환하여 나머지 파이프라인 수정이 거의 없도록 합니다.

---

## 0) 공통: 파일의 상단 섹션 구성

두 실험 파일 모두 다음 순서를 **baseline과 동일하게 유지**합니다.

1. **환경 설정 & 시드 고정** (baseline 그대로)
2. **Stage2 Bag 로드 & Instance 평균 계산** (baseline 그대로)
3. **Dataset/DataLoader** (baseline 그대로)
4. **손실/옵티마이저/스케줄러** (baseline 그대로, 단 최종 Weighted BCE도 동일)
5. **학습/평가 함수(train\_one\_epoch/evaluate/train\_model)** (baseline 그대로)
6. **최종 파이프라인(Weighted BCE + Val 기반 임계값 최적화 + Test 보고)** (baseline 그대로)
7. **단, “모델 정의” 블록만 TransformerMIL / TransMIL로 교체**

> 즉, 아래 **모델 블록**만 각 파일에 붙여 넣으면 됩니다.

---

## 1) stage3\_transformer\_mil: **Transformer-based MIL(범주형)**

### 1-1. 설계 포인트

* **입력**: `x`는 `[B, N, D]` (B=batch, N=instance 개수=10, D=256)
* **전처리**: `Linear(256→d_model)`로 투영 + **Positional Encoding (sinusoidal 또는 learned)**
* **코어**: `nn.TransformerEncoder`(L개 레이어, H개 헤드)로 **인스턴스 간 상호작용** 학습
* **집계(Aggregation)**: **Attention Pooling 헤드**(trainable query)로 bag 표현 `z_bag` 생성
* **출력**: `logits = Linear(z_bag → 1)` + **인스턴스 중요도 `weights`**(해석용) 반환
* **인터페이스**: `forward -> (logits, weights)`  (baseline과 동일)

### 1-2. 모델 코드 블록 (이 블록만 붙여넣어 교체)

```python
# =========================
# Models: PosEnc & TransformerMIL
# =========================
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout_p: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2]  = torch.sin(position * div_term)
        pe[:, 1::2]  = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # not a parameter

    def forward(self, x: torch.Tensor):
        """
        x: [B, N, d_model]
        """
        B, N, D = x.size()
        x = x + self.pe[:N].unsqueeze(0)  # [1, N, D] broadcast
        return self.dropout(x)

class AttentionPooler(nn.Module):
    """
    ABMIL 스타일의 소프트 어텐션 풀러.
    입력 [B, N, d_model] -> 가중합 [B, d_model] + weights [B, N]
    """
    def __init__(self, d_model: int, hidden: int = 128, dropout_p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout_p)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, H):  # H: [B, N, d_model]
        A = torch.tanh(self.fc1(self.drop(H)))   # [B, N, hidden]
        A = self.fc2(A).squeeze(-1)              # [B, N]
        weights = torch.softmax(A, dim=1)        # [B, N]
        Z = torch.sum(weights.unsqueeze(-1) * H, dim=1)  # [B, d_model]
        return Z, weights

class TransformerMIL(nn.Module):
    """
    Transformer 기반 MIL (범주) - 인스턴스 간 self-attention + attention pooling 집계.
    forward: (logits, weights) 반환 -> baseline 학습 루프와 호환.
    """
    def __init__(
        self,
        input_dim=256,      # ArcFace 임베딩 차원
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout_p=0.1,
        pos_enc='sin'      # 'sin' or 'learned'
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        if pos_enc == 'sin':
            self.posenc = SinusoidalPositionalEncoding(d_model, max_len=128, dropout_p=dropout_p)
        else:
            self.pos_embedding = nn.Embedding(128, d_model)  # N<=128 가정
            nn.init.normal_(self.pos_embedding.weight, std=0.02)
            self.posenc = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout_p, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pooler  = AttentionPooler(d_model, hidden=128, dropout_p=dropout_p)
        self.classifier = nn.Linear(d_model, 1)

        # init
        nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight); nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        """
        x: [B, N, D]
        returns: logits[B], weights[B,N]
        """
        B, N, D = x.shape
        h = self.proj(x)  # [B, N, d_model]
        if self.posenc is not None:
            h = self.posenc(h)     # sinusoidal
        else:
            idx = torch.arange(N, device=h.device).unsqueeze(0).repeat(B,1)
            h = h + self.pos_embedding(idx)

        h = self.encoder(h)        # [B, N, d_model]
        z_bag, weights = self.pooler(h)   # [B, d_model], [B, N]
        logits = self.classifier(z_bag).squeeze(-1)  # [B]
        return logits, weights
```

### 1-3. 이 파일의 “모델 인스턴스” 교체 부분

```python
# --- 기존 ---
# mil_model = AttentionMIL(input_dim=256, hidden_dim=128, dropout_p=0.1).to(device)

# --- 교체 (TransformerMIL 실험용) ---
mil_model = TransformerMIL(
    input_dim=256, d_model=128, nhead=4, num_layers=2,
    dim_feedforward=256, dropout_p=0.1, pos_enc='sin'
).to(device)
```

> 나머지 손실/학습/평가/임계값 탐색/ROC 등은 baseline과 동일하게 실행됩니다.

---

## 2) stage3\_transmil: **TransMIL(특정 모델)**

### 2-1. 설계 포인트

* **핵심 차이**: **\[CLS] 토큰을 추가**하여 bag의 대표 토큰으로 사용하고, **self-attention**으로 인스턴스-인스턴스 관계를 학습.
* **bag 표현**: 마지막 인코더 출력의 **CLS 토큰 임베딩을 bag 표현**으로 사용.
* **해석성(weights)**: 마지막 레이어의 **CLS → tokens** 어텐션 맵을 **인스턴스 중요도**로 사용(헤드 평균).
* **인터페이스**: `forward -> (logits, weights)` (weights는 CLS가 본 인스턴스별 주의도)

### 2-2. 모델 코드 블록

```python
# =========================
# Model: TransMIL (CLS token + self-attn, attn map 반환)
# =========================
class TransformerEncoderLayerWithAttn(nn.Module):
    """
    표준 TransformerEncoderLayer를 커스텀하여
    마지막 self-attention의 attention weight를 반환할 수 있도록 구현.
    """
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # init
        for m in [self.linear1, self.linear2]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, src, attn_mask=None, key_padding_mask=None, need_attn=False):
        """
        src: [B, T, d_model]  (T = N + 1, CLS 포함)
        returns: out, last_attn (if need_attn)
        """
        # Self-attention
        attn_out, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True  # 평균된 헤드 가중치 반환 [B, T, T]
        )
        src2 = self.dropout1(attn_out)
        src  = self.norm1(src + src2)

        # FFN
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.dropout2(ff)
        out  = self.norm2(src + src2)

        if need_attn:
            return out, attn_weights  # [B, T, T]
        else:
            return out, None

class TransMIL(nn.Module):
    """
    TransMIL 스타일: [CLS] 토큰 + self-attention 인코더, CLS 임베딩으로 bag 분류.
    weights: 마지막 레이어의 CLS→tokens attention을 반환 (해석성).
    """
    def __init__(
        self,
        input_dim=256,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout_p=0.1,
        pos_enc='sin'  # 'sin' or 'learned'
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        if pos_enc == 'sin':
            self.posenc = SinusoidalPositionalEncoding(d_model, max_len=128, dropout_p=dropout_p)
            self.pos_embedding = None
        else:
            self.posenc = None
            self.pos_embedding = nn.Embedding(128, d_model)
            nn.init.normal_(self.pos_embedding.weight, std=0.02)

        # CLS 토큰 (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(d_model, nhead, dim_feedforward, dropout_p)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

        nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight); nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        """
        x: [B, N, D]
        returns: logits[B], weights[B, N]  (weights는 CLS가 본 tokens 주의도)
        """
        B, N, D = x.size()
        h = self.proj(x)  # [B, N, d_model]
        if self.posenc is not None:
            h = self.posenc(h)
        else:
            idx = torch.arange(N, device=h.device).unsqueeze(0).repeat(B,1)
            h = h + self.pos_embedding(idx)

        # prepend CLS
        cls_tok = self.cls_token.expand(B, -1, -1)     # [B,1,d_model]
        h = torch.cat([cls_tok, h], dim=1)             # [B, N+1, d_model]

        attn_map_last = None
        out = h
        for li, layer in enumerate(self.layers):
            # 마지막 레이어에서 attn map을 추출
            need_attn = (li == len(self.layers) - 1)
            out, attn_map = layer(out, need_attn=need_attn)
            if need_attn:
                attn_map_last = attn_map  # [B, T, T], T=N+1

        out = self.norm(out)                 # [B, N+1, d_model]
        cls_out = out[:, 0, :]               # [B, d_model]
        logits = self.classifier(cls_out).squeeze(-1)

        # 해석: CLS -> tokens attention을 weights로 사용 (CLS=0번째, tokens=1..N)
        if attn_map_last is not None:
            # attn_map_last: [B, T, T], (query idx, key idx)
            weights = attn_map_last[:, 0, 1:]    # [B, N]
            weights = torch.softmax(weights, dim=1)
        else:
            # fallback: uniform
            weights = torch.full((B, N), 1.0 / N, device=x.device)

        return logits, weights
```

### 2-3. 이 파일의 “모델 인스턴스” 교체 부분

```python
# --- 기존 ---
# mil_model = AttentionMIL(input_dim=256, hidden_dim=128, dropout_p=0.1).to(device)

# --- 교체 (TransMIL 실험용) ---
mil_model = TransMIL(
    input_dim=256, d_model=128, nhead=4, num_layers=2,
    dim_feedforward=256, dropout_p=0.1, pos_enc='sin'
).to(device)
```

> 마찬가지로 나머지 파이프라인은 baseline 그대로 사용합니다.

---

## 3) 하이퍼파라미터 & 실행 팁

* **공통 권장값**: `d_model=128`, `nhead=4`, `num_layers=2`, `dim_ff=256`, `dropout=0.1`
* **배치/시드/에포크/스케줄러**: baseline 동일 (batch=16, epoch=10, patience=3…)
* **Positional Encoding**:

  * 기본은 `sinusoidal` 추천(길이 변화에 안정적).
  * 인덱스가 사실상 “문서 내 윈도우 순서”를 의미하므로 **learned embedding**으로도 OK.
* **해석성**:

  * `TransformerMIL` → `AttentionPooler`의 `weights`가 인스턴스 중요도.
  * `TransMIL` → **마지막 레이어의 CLS→tokens attention**이 중요도.
* **변형 실험(옵션)**:

  * `TransformerMIL`의 풀러를 **Top‑k pooling**으로 바꾸어 희소 양성에 대한 민감도 비교.
  * `TransMIL`에 **CLS + AttentionPooling 병렬 결합**(concat) 후 classifier에 투입 (DSMIL과 유사한 듀얼집계).

---

## 4) 저장/로그 네이밍(혼동 방지)

* 모델 가중치:

  * `best_transformer_mil.pth`, `best_transmil.pth`
* 결과 JSON/CSV:

  * `stage3_transformer_mil_results.json`, `stage3_transmil_results.json`
* 실험 태그(콘솔/ WandB 등):

  * `tmil_d128_h4_L2_ff256_drop01_sin`
  * `transmil_cls_d128_h4_L2_ff256_drop01_sin`

---

## 5) 체크리스트(짧게)

* [ ] baseline과 **동일 입출력 인터페이스** 유지(`(logits, weights)` 반환)
* [ ] Positional Encoding 적용(둘 중 하나)
* [ ] `nn.TransformerEncoder`(TMIL) vs `CLS+custom attn`(TransMIL) 구분 적용
* [ ] Weighted BCE/임계값 검색/ROC·F1 보고 **그대로 재사용**
* [ ] 저장/로그 이름 분리

---

### 자기점검(1–2문장)

* 두 실험 파일은 baseline 학습 루프와 완전 호환되도록 **동일한 forward 시그니처**를 유지하므로, 교체만으로 바로 실행 가능합니다. 다만 `TransMIL`의 **attention 가중치 추출**은 마지막 레이어의 self-attention에 의존하므로, 레이어 수 변경 시에도 **CLS→tokens 맵**을 올바르게 가져오는지(차원 `[B, T, T]`)를 확인하세요.
