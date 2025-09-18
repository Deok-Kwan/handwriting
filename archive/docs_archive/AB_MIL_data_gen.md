# MIL + Siamese Network 실험 환경 구축 가이드

## 📁 필요한 파일 목록

### 1. 업로드해야 할 파일들
```
# 데이터 파일
- naver_ocr.csv (원본 경로: /data/csafeproject/another_datas/CSAFE/naver_ocr.csv)
- csafe_version5_xai_train/ 폴더의 이미지들 (선택사항, CSV에 경로만 있으면 됨)

# 모델 가중치
- csafe_vit_300classes_best_model.pth (원본 경로: /data/csafeproject/CSAFE_version5/trained_model/)

# 코드 파일
- mil_data_generator.ipynb
- mil_data_generator2.ipynb
- AB_MIL_autoencoder_128d.ipynb
```

## 🛠️ 환경 설정

### Step 1: 패키지 설치
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.4.12
pip install pandas numpy tqdm scikit-learn matplotlib seaborn
pip install pillow
```

### Step 2: 디렉토리 구조 생성
```bash
mkdir -p data/images
mkdir -p models
mkdir -p notebooks
mkdir -p output/csv
mkdir -p output/pkl
```

## 📝 실행 순서

### 1. Siamese Network 학습 (새로 작성 필요)

`train_siamese.ipynb` 생성:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import timm
from tqdm import tqdm
import random

# 1. Siamese Network 정의
class SiameseNetwork(nn.Module):
    def __init__(self, base_model, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.base_model = base_model
        self.embedding_layer = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward_one(self, x):
        x = self.base_model(x)
        x = self.embedding_layer(x)
        return F.normalize(x, p=2, dim=1)
    
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2

# 2. Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, out1, out2, label):
        distance = F.pairwise_distance(out1, out2)
        loss = torch.mean((1-label) * torch.pow(distance, 2) +
                         label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss

# 3. Dataset 클래스
class SiameseHandwritingDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        self.label_groups = df.groupby('label').groups
        self.labels = list(self.label_groups.keys())
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 50% 확률로 positive/negative pair 생성
        if random.random() > 0.5:
            # Positive pair (같은 작성자)
            label = self.df.iloc[idx]['label']
            indices = self.label_groups[label]
            positive_idx = random.choice(indices)
            
            img1_path = self.df.iloc[idx]['image_path']
            img2_path = self.df.iloc[positive_idx]['image_path']
            label_tensor = torch.tensor(0.0)  # 같은 작성자 = 0
        else:
            # Negative pair (다른 작성자)
            label1 = self.df.iloc[idx]['label']
            label2 = random.choice([l for l in self.labels if l != label1])
            negative_idx = random.choice(self.label_groups[label2])
            
            img1_path = self.df.iloc[idx]['image_path']
            img2_path = self.df.iloc[negative_idx]['image_path']
            label_tensor = torch.tensor(1.0)  # 다른 작성자 = 1
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, label_tensor

# 4. 학습 코드
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transform 정의
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 데이터 로드
data = pd.read_csv('naver_ocr.csv')
data = data[data['label'] > 299].reset_index(drop=True)

# Train/Val 분할
train_data = data[(data['label'] >= 300) & (data['label'] <= 414)]
val_data = data[(data['label'] > 414) & (data['label'] <= 444)]

# Dataset 및 DataLoader
train_dataset = SiameseHandwritingDataset(train_data, transform)
val_dataset = SiameseHandwritingDataset(val_data, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# 모델 초기화
base_model = timm.create_model('vit_base_patch16_224', pretrained=True)
base_model.head = nn.Linear(base_model.head.in_features, 300)
base_model.load_state_dict(torch.load('csafe_vit_300classes_best_model.pth'))

siamese_model = SiameseNetwork(base_model, embedding_dim=128).to(device)

# 옵티마이저 및 손실함수
optimizer = torch.optim.Adam(siamese_model.parameters(), lr=0.0001)
criterion = ContrastiveLoss(margin=1.0)

# 학습
num_epochs = 20
best_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    siamese_model.train()
    train_loss = 0.0
    for img1, img2, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        out1, out2 = siamese_model(img1, img2)
        loss = criterion(out1, out2, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    siamese_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for img1, img2, labels in val_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            out1, out2 = siamese_model(img1, img2)
            loss = criterion(out1, out2, labels)
            val_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Save best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(siamese_model.state_dict(), 'siamese_best_model.pth')
        print('Model saved!')
```

### 2. mil_data_generator.ipynb 수정

기존 코드에서 다음 부분만 수정:

```python
# 기존 Autoencoder 모델 대신 Siamese 모델 사용
class SiameseEmbedder(nn.Module):
    def __init__(self, siamese_model):
        super().__init__()
        self.siamese_model = siamese_model
    
    def get_latent_vector(self, x):
        return self.siamese_model.forward_one(x)

# 모델 로드 부분 수정
base_model = timm.create_model('vit_base_patch16_224', pretrained=True)
base_model.head = nn.Linear(base_model.head.in_features, 300)
base_model.load_state_dict(torch.load('csafe_vit_300classes_best_model.pth'))

siamese_model = SiameseNetwork(base_model, embedding_dim=128)
siamese_model.load_state_dict(torch.load('siamese_best_model.pth'))
modified_model = SiameseEmbedder(siamese_model).to(device)

# data_info 변경
data_info = 'siamese_128d'

# 출력 파일명이 자동으로 변경됨:
# mil_siamese_128d_train_data.csv
# mil_siamese_128d_val_data.csv
# mil_siamese_128d_test_data.csv
```

### 3. mil_data_generator2.ipynb 수정

입력 파일명만 변경:
```python
data_info = 'siamese_128d'  # 'autoencoder_128d' 대신
train_file = f"mil_{data_info}_train_data.csv"
val_file = f"mil_{data_info}_val_data.csv"
test_file = f"mil_{data_info}_test_data.csv"

# 출력 파일:
# train_bags_siamese_128d.pkl
# val_bags_siamese_128d.pkl
# test_bags_siamese_128d.pkl
```

### 4. AB_MIL_autoencoder_128d.ipynb 수정

데이터 로드 부분만 변경:
```python
# 데이터 로드
train_bags = pd.read_pickle("train_bags_siamese_128d.pkl")
val_bags = pd.read_pickle("val_bags_siamese_128d.pkl")
test_bags = pd.read_pickle("test_bags_siamese_128d.pkl")

# 모델 저장 이름 변경
model_name = 'ab_mil_siamese_128d'
```

## ⚠️ 중요 사항

1. **경로 수정**: 모든 파일 경로를 새로운 환경에 맞게 수정
2. **GPU 메모리**: 배치 크기 조정 필요시 train_loader의 batch_size 수정
3. **데이터 경로**: naver_ocr.csv의 image_path가 실제 이미지 위치를 가리키도록 수정
4. **저장 경로**: 모든 출력 파일 경로 확인

## 🎯 예상 결과

1. Siamese 네트워크 학습 후: `siamese_best_model.pth` 생성
2. 새로운 임베딩으로 CSV 생성: `mil_siamese_128d_*.csv` 파일들
3. Bag 데이터 생성: `*_bags_siamese_128d.pkl` 파일들
4. MIL 모델 학습: 기존 50% 정확도보다 향상된 결과 기대

## 📊 성능 검증

Siamese 임베딩이 제대로 학습되었는지 확인:
```python
# t-SNE 시각화
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 임베딩 추출 후
embeddings = []
labels = []
# ... 임베딩 추출 코드 ...

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab20', alpha=0.6)
plt.colorbar(scatter)
plt.title('Siamese Embeddings t-SNE Visualization')
plt.show()
```

작성자별로 클러스터가 형성되면 성공적으로 학습된 것입니다.