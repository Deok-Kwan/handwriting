"""
Stage 3 모델 비교 (모델만 다름): Attention vs Transformer MIL

Baseline과 완전히 동일한 조건(데이터/하이퍼파라미터/평가)에서
오로지 모델(Architecture)만 다르게 하여 성능을 비교합니다.

실행 방법(예):
  python experiments/arcface/agent/stage3_baseline_transformer.py
"""

import os
import random
import pickle
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
)
from tqdm import tqdm


# -----------------------------
# 환경 설정 + 시드 고정
# -----------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("MIL_STAGE3_GPU", "3")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(42)


# -----------------------------
# 데이터 로드 (Baseline과 동일 경로/전처리) + 크기 검증
# -----------------------------
def to_instance_means(bags):
    # (10,5,256) -> (10,256)
    return [bag.mean(axis=1).astype(np.float32) for bag in bags]


class MILDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def load_data_loaders(batch_size: int = 16) -> Tuple[DataLoader, DataLoader, DataLoader]:
    embedding_margin = "0.4"
    bags_dir = "/workspace/MIL/data/processed/bags"
    train_pkl = os.path.join(bags_dir, f"bags_arcface_margin_{embedding_margin}_50p_random_train.pkl")
    val_pkl = os.path.join(bags_dir, f"bags_arcface_margin_{embedding_margin}_50p_random_val.pkl")
    test_pkl = os.path.join(bags_dir, f"bags_arcface_margin_{embedding_margin}_50p_random_test.pkl")

    with open(train_pkl, "rb") as f:
        train_data = pickle.load(f)
    with open(val_pkl, "rb") as f:
        val_data = pickle.load(f)
    with open(test_pkl, "rb") as f:
        test_data = pickle.load(f)

    train_features = to_instance_means(train_data["bags"])
    val_features = to_instance_means(val_data["bags"])
    test_features = to_instance_means(test_data["bags"])

    train_labels = train_data["labels"]
    val_labels = val_data["labels"]
    test_labels = test_data["labels"]

    print(
        f"Train bags: {len(train_labels)}, Val bags: {len(val_labels)}, Test bags: {len(test_labels)}"
    )

    # Baseline 보고치(3600/1200/1200)와 동일하지 않으면 즉시 중단
    assert (
        len(train_labels) == 3600 and len(val_labels) == 1200 and len(test_labels) == 1200
    ), "데이터 분할 크기가 baseline(3600/1200/1200)과 다릅니다. 동일 스냅샷으로 맞춰주세요."

    train_loader = DataLoader(MILDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MILDataset(val_features, val_labels), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(MILDataset(test_features, test_labels), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# -----------------------------
# 손실함수 (Baseline: WeightedBCE)
# -----------------------------
class WeightedBCE(nn.Module):
    def __init__(self, fp_weight: float = 2.0) -> None:
        super().__init__()
        self.fp_weight = fp_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.bce(logits, labels)
        fp_mask = (labels == 0).float()  # Negative class의 FP에 가중치
        loss = loss * (1 + self.fp_weight * fp_mask)
        return loss.mean()


# -----------------------------
# 모델 정의: AttentionMIL (Baseline), TransformerMIL (비교 대상)
# -----------------------------
class AttentionMIL(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.instance_fc = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.att_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.att_fc2 = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.instance_fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.instance_fc.bias)
        nn.init.xavier_uniform_(self.att_fc1.weight)
        nn.init.zeros_(self.att_fc1.bias)
        nn.init.xavier_uniform_(self.att_fc2.weight)
        nn.init.zeros_(self.att_fc2.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor):
        h = torch.relu(self.instance_fc(x))
        h = self.dropout(h)
        a = torch.tanh(self.att_fc1(h))
        a = self.att_fc2(a).squeeze(-1)
        weights = torch.softmax(a, dim=1)
        bag_repr = torch.sum(weights.unsqueeze(-1) * h, dim=1)
        bag_repr = self.dropout(bag_repr)
        logits = self.classifier(bag_repr).squeeze(-1)
        return logits, weights


class TransformerMIL(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout_p, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))  # learnable CLS
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, N, D)
        b, n, _ = X.shape
        h = self.input_proj(X)
        cls_tokens = self.cls_token.expand(b, 1, -1)
        h = torch.cat([cls_tokens, h], dim=1)  # (B, N+1, H)
        out = self.transformer_encoder(h)
        cls_out = out[:, 0, :]
        logits = self.classifier(self.dropout(cls_out)).squeeze(-1)
        return logits


# -----------------------------
# 학습/평가 루프 (Baseline과 동일 로직)
# -----------------------------
def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loader: DataLoader, criterion: nn.Module):
    model.train()
    total_loss = 0.0
    preds_all, labels_all = [], []
    for X, y in tqdm(loader, desc="Train", leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        if isinstance(model, AttentionMIL):
            logits, _ = model(X)
        else:
            logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    return total_loss / len(loader.dataset), accuracy_score(labels_all, preds_all)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    probs_all, preds_all, labels_all = [], [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Eval", leave=False):
            X, y = X.to(device), y.to(device)
            if isinstance(model, AttentionMIL):
                logits, _ = model(X)
            else:
                logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            probs_all.extend(probs.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y.cpu().numpy())
    acc = accuracy_score(labels_all, preds_all)
    auc_v = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.0
    f1 = f1_score(labels_all, preds_all) if len(set(preds_all)) > 1 else 0.0
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": acc,
        "auc": auc_v,
        "f1": f1,
        "probs": np.array(probs_all),
        "labels": np.array(labels_all),
        "preds": np.array(preds_all),
    }


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    max_epochs: int = 10,
    patience: int = 3,
    name: str = "model",
):
    best_auc = 0.0
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": [], "val_f1": []}

    print(f"\n🚀 {name} 모델 학습 시작...")
    print(f"   Max epochs: {max_epochs}, Patience: {patience}")

    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs} – {name}")
        tr_loss, tr_acc = train_one_epoch(model, optimizer, train_loader, criterion)
        val_res = evaluate(model, val_loader, criterion)
        val_loss, val_acc, val_auc, val_f1 = val_res["loss"], val_res["accuracy"], val_res["auc"], val_res["f1"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        print(f"  Train: Loss={tr_loss:.4f}, Acc={tr_acc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}, F1={val_f1:.4f}")

        scheduler.step(val_auc)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, f"best_{name}.pth")
            print(f"  ✅ New best AUC: {best_auc:.4f} – model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  ⏳ No improvement. Patience: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print("  🛑 Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  📂 Best model loaded (AUC: {best_auc:.4f})")

    return model, history


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    best_thr, best_val = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, 37):
        preds = (probs >= thr).astype(int)
        val = f1_score(labels, preds, zero_division=0)
        if val > best_val:
            best_val, best_thr = val, thr
    return best_thr, best_val


def main() -> None:
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader, test_loader = load_data_loaders(batch_size=16)

    # 공정 비교 설정 (Baseline과 동일)
    criterion = WeightedBCE(fp_weight=2.0)
    learning_rate = 1e-3
    max_epochs = 10
    patience = 3
    scheduler_patience = 1

    print("\n🔬 모델 비교 실험 (Baseline과 완전 동일 조건)")
    print("=" * 60)
    print("손실 함수: WeightedBCE(fp_weight=2.0)")
    print(f"학습률: {learning_rate}")
    print(f"최대 에포크: {max_epochs}, Patience: {patience}")
    print(f"Scheduler Patience: {scheduler_patience}")
    print("=" * 60)

    results: Dict[str, Dict[str, Any]] = {}

    # 1) Baseline Attention MIL
    seed_everything(42)
    att_model = AttentionMIL(input_dim=256, hidden_dim=128, dropout_p=0.1).to(device)
    att_opt = torch.optim.Adam(att_model.parameters(), lr=learning_rate)
    att_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(att_opt, mode="max", factor=0.5, patience=scheduler_patience, verbose=True)
    att_model, _ = train_model(att_model, att_opt, att_sch, train_loader, val_loader, criterion, max_epochs=max_epochs, patience=patience, name="attention_mil")
    results["Attention"] = {"val": evaluate(att_model, val_loader, criterion), "test": evaluate(att_model, test_loader, criterion)}

    # 2) Transformer MIL
    seed_everything(42)
    tr_model = TransformerMIL(input_dim=256, hidden_dim=128, num_heads=4, num_layers=2, dropout_p=0.1).to(device)
    tr_opt = torch.optim.Adam(tr_model.parameters(), lr=learning_rate)
    tr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(tr_opt, mode="max", factor=0.5, patience=scheduler_patience, verbose=True)
    tr_model, _ = train_model(tr_model, tr_opt, tr_sch, train_loader, val_loader, criterion, max_epochs=max_epochs, patience=patience, name="transformer_mil")
    results["Transformer"] = {"val": evaluate(tr_model, val_loader, criterion), "test": evaluate(tr_model, test_loader, criterion)}

    # 최종 성능 리포팅: Validation 최적 임계값을 Test에 적용
    print("\n📊 모델별 최종 성능 비교")
    print("=" * 80)
    final = {}
    for name, res in results.items():
        val_res, tst_res = res["val"], res["test"]
        thr, best_f1_val = find_best_threshold(val_res["probs"], val_res["labels"])
        test_preds_adj = (tst_res["probs"] >= thr).astype(int)
        acc = accuracy_score(tst_res["labels"], test_preds_adj)
        f1 = f1_score(tst_res["labels"], test_preds_adj, zero_division=0)
        prec = precision_score(tst_res["labels"], test_preds_adj, zero_division=0)
        rec = recall_score(tst_res["labels"], test_preds_adj, zero_division=0)
        auc_v = tst_res["auc"]
        final[name] = {"threshold": thr, "accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "auc": auc_v}

        print(f"\n{name}:")
        print(f"  최적 임계값: {thr:.3f} (Val F1: {best_f1_val:.3f})")
        print(f"  Test Accuracy: {acc:.3f}")
        print(f"  Test F1: {f1:.3f}")
        print(f"  Test Precision: {prec:.3f}")
        print(f"  Test Recall: {rec:.3f}")
        print(f"  Test AUC: {auc_v:.3f}")

    print("\n" + "=" * 80)
    print("📈 모델 성능 요약 테이블")
    print("=" * 80)
    print(f"{'Model':<15} {'Accuracy':<10} {'F1':<8} {'Precision':<11} {'Recall':<8} {'AUC':<8}")
    print("-" * 80)
    for name, r in final.items():
        print(f"{name:<15} {r['accuracy']:<10.3f} {r['f1']:<8.3f} {r['precision']:<11.3f} {r['recall']:<8.3f} {r['auc']:<8.3f}")

    best_auc = max(final.items(), key=lambda x: x[1]["auc"]) if final else (None, None)
    best_f1 = max(final.items(), key=lambda x: x[1]["f1"]) if final else (None, None)
    print("\n🏆 최고 성능:")
    if best_auc[0] is not None:
        print(f"  AUC 기준: {best_auc[0]} (AUC: {best_auc[1]['auc']:.3f})")
    if best_f1[0] is not None:
        print(f"  F1 기준:  {best_f1[0]} (F1: {best_f1[1]['f1']:.3f})")


if __name__ == "__main__":
    main()

