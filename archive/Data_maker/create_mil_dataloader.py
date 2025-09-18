# MIL 데이터셋을 딥러닝 모델 학습에 사용하기 위한 데이터 로더(예: PyTorch DataLoader)를 생성하는 역할


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import random
import albumentations as A
from PIL import Image

class MILDataset(Dataset):
    """
    MIL (Multiple Instance Learning) 데이터셋 클래스.
    구조화된 MIL 데이터셋을 로드하고 학습/추론에 사용할 수 있는 형태로 변환합니다.
    
    지원하는 모드:
    - 'classification': 문서가 단일 작성자인지 다중 작성자인지 분류
    - 'segmentation': 작성자별 필기 영역 분할
    - 'writer_identification': 작성자 식별
    """
    
    def __init__(self, 
                 root_dir, 
                 split='train', 
                 mode='classification',
                 img_size=(512, 512),
                 transform=None,
                 mask_value_scale=1.0,
                 convert_mask_to_onehot=False):
        """
        Args:
            root_dir (str): 구조화된 데이터셋 루트 디렉토리
            split (str): 'train', 'val', 'test' 중 하나
            mode (str): 'classification', 'segmentation', 'writer_identification' 중 하나
            img_size (tuple): 이미지 리사이즈 크기
            transform (callable, optional): 샘플에 적용할 추가 변환
            mask_value_scale (float): 마스크 값 스케일링 계수 (마스크 값 범위에 따라 조정 필요)
            convert_mask_to_onehot (bool): 마스크를 원-핫 인코딩으로 변환할지 여부
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.mode = mode
        self.img_size = img_size
        self.transform = transform
        self.mask_value_scale = mask_value_scale
        self.convert_mask_to_onehot = convert_mask_to_onehot
        
        split_file = self.root_dir / 'splits' / f'{split}.json'
        if not split_file.exists():
            raise FileNotFoundError(f"분할 파일이 존재하지 않습니다: {split_file}")
            
        with open(split_file, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
            
        print(f"MIL {split} 데이터셋 로드: {len(self.samples)}개 샘플")
        
        # 작성자 ID 사전 구성 (writer_identification 모드용)
        if mode == 'writer_identification':
            self.author_ids = self._collect_author_ids()
            print(f"총 작성자 수: {len(self.author_ids)}")
            
        # 기본 변환 설정
        self.base_transform = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _collect_author_ids(self):
        """모든 고유 작성자 ID를 수집하고 인덱스 사전을 구성합니다."""
        author_set = set()
        for sample in self.samples:
            for author in sample['authors']:
                author_set.add(author)
        return {author: idx for idx, author in enumerate(sorted(author_set))}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # 파일 경로 구성
        img_path = self.root_dir / sample_info['img_path']
        mask_path = self.root_dir / sample_info['mask_path']
        meta_path = self.root_dir / sample_info['metadata_path']
        
        # 이미지 로드
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 마스크 로드 및 처리
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"마스크를 로드할 수 없습니다: {mask_path}")
        
        # 메타데이터 로드 (필요시)
        metadata = None
        if self.mode == 'writer_identification':
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # 변환 적용 (albumentations 사용)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            transformed = self.base_transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # 이미지를 PyTorch 텐서로 변환
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # 모드에 따른 반환값 구성
        if self.mode == 'classification':
            # 단일/다중 작성자 분류
            is_multi = 1 if sample_info['is_multi_author'] else 0
            label = torch.tensor(is_multi, dtype=torch.float32)
            
            return {
                'image': image,
                'label': label,
                'path': str(img_path)
            }
            
        elif self.mode == 'segmentation':
            # 마스크 값 스케일링
            mask = mask.astype(np.float32) * self.mask_value_scale
            
            # 마스크를 원-핫 인코딩으로 변환 (선택 사항)
            if self.convert_mask_to_onehot:
                # 최대 작성자 ID 찾기
                max_id = np.max(mask) if np.max(mask) > 0 else 1
                mask_onehot = np.zeros((int(max_id+1), *mask.shape), dtype=np.float32)
                for i in range(int(max_id+1)):
                    mask_onehot[i] = (mask == i).astype(np.float32)
                mask = torch.from_numpy(mask_onehot)
            else:
                mask = torch.from_numpy(mask).unsqueeze(0).float()
            
            return {
                'image': image,
                'mask': mask,
                'path': str(img_path)
            }
            
        elif self.mode == 'writer_identification':
            # 다중 작성자 문서의 경우
            if sample_info['is_multi_author']:
                # 인스턴스 레이블 (마스크 기반)
                instance_ids = np.unique(mask)[1:]  # 배경(0) 제외
                
                # 인스턴스 바운딩 박스 및 레이블 추출
                instances = []
                for instance_id in instance_ids:
                    # 인스턴스 영역 추출
                    instance_mask = (mask == instance_id).astype(np.uint8)
                    
                    # 바운딩 박스 계산
                    contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        x, y, w, h = cv2.boundingRect(contours[0])
                        
                        # 해당 인스턴스의 작성자 ID 찾기
                        author_id = None
                        if metadata and 'instance_labels' in metadata:
                            for instance_info in metadata['instance_labels']:
                                if instance_info['author_id_int'] == instance_id:
                                    author_id = instance_info['author_id_str']
                                    break
                        
                        if author_id and author_id in self.author_ids:
                            instances.append({
                                'bbox': [x, y, x+w, y+h],
                                'label': self.author_ids[author_id]
                            })
                
                return {
                    'image': image,
                    'instances': instances,
                    'mask': torch.from_numpy(mask).unsqueeze(0).float(),
                    'path': str(img_path)
                }
            
            # 단일 작성자 문서의 경우 
            else:
                author_id = sample_info['authors'][0]
                label = self.author_ids[author_id]
                
                return {
                    'image': image,
                    'label': torch.tensor(label, dtype=torch.long),
                    'path': str(img_path)
                }
        
        else:
            raise ValueError(f"지원하지 않는 모드입니다: {self.mode}")

# 데이터 증강 설정 예시
def get_train_transforms(img_size=(512, 512)):
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        ], p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_transforms(img_size=(512, 512)):
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_dataloader(root_dir, batch_size=8, split='train', mode='classification', img_size=(512, 512), num_workers=4):
    """편리한 데이터 로더 생성 함수"""
    
    if split == 'train':
        transform = get_train_transforms(img_size)
    else:
        transform = get_val_transforms(img_size)
    
    dataset = MILDataset(
        root_dir=root_dir,
        split=split,
        mode=mode,
        img_size=img_size,
        transform=transform
    )
    
    shuffle = (split == 'train')
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

# 사용 예시
if __name__ == "__main__":
    # 데이터 로더 생성
    train_loader = get_dataloader(
        root_dir='./mil_dataset',
        batch_size=8,
        split='train',
        mode='classification'
    )
    
    # 샘플 데이터 확인
    for batch in train_loader:
        print(f"이미지 크기: {batch['image'].shape}")
        print(f"레이블: {batch['label']}")
        break

    # 세그멘테이션 모드
    seg_loader = get_dataloader(
        root_dir='./mil_dataset',
        batch_size=8,
        split='train',
        mode='segmentation'
    )
    
    for batch in seg_loader:
        print(f"이미지 크기: {batch['image'].shape}")
        print(f"마스크 크기: {batch['mask'].shape}")
        break 