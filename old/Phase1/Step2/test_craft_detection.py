#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
import argparse
import random
from tqdm import tqdm

# 상대 경로 대신 절대 경로 사용
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from craft_utils import CraftTextDetector
from craft_mil_dataset import CraftMILDataset, get_craft_mil_dataloader

def visualize_text_detection(image_path, output_dir, craft_detector=None):
    """
    이미지에서 텍스트 감지 및 시각화 (단어 단위)
    
    Args:
        image_path: 이미지 파일 경로
        output_dir: 출력 디렉토리
        craft_detector: CraftTextDetector 인스턴스
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # CRAFT 텍스트 감지기 초기화
    if craft_detector is None:
        craft_detector = CraftTextDetector()
    
    # 텍스트 영역 감지
    prediction_result = craft_detector.detect_text_regions(image)
    
    # 다각형을 경계 상자로 변환
    bboxes = craft_detector.convert_polys_to_bboxes(prediction_result["boxes"])
    
    # 텍스트 영역 분할 함수 (단어 단위로 분할)
    def split_large_text_regions(bboxes, max_width=100, max_height=40):
        """큰 텍스트 영역을 더 작은 단어 단위로 분할 (더 공격적인 분할)"""
        result = []
        for bbox in bboxes:
            x, y, w, h = bbox
            
            # 강제 분할 기준: 크기가 임계값 초과 또는 비율이 비정상적
            force_split = (w > max_width or h > max_height or w/h > 3.0 or h/w > 3.0)
            
            if w > max_width or force_split:
                # 더 공격적인 분할: 최소 2개, 크기에 따라 더 많이 분할
                num_splits = max(3, w // (max_width // 2))
                split_width = w // num_splits
                overlap = int(split_width * 0.1)  # 오버랩 감소
                
                for i in range(num_splits):
                    # 오버랩을 고려한 시작점과 너비 계산
                    split_x = max(0, x + i * split_width - (i > 0) * overlap)
                    if i == num_splits - 1:  # 마지막 분할
                        split_w = x + w - split_x
                    else:
                        split_w = split_width + overlap
                    
                    # 높이도 크면 추가 분할
                    if h > max_height:
                        h_splits = max(3, h // (max_height // 2))
                        split_height = h // h_splits
                        h_overlap = int(split_height * 0.1)
                        
                        for j in range(h_splits):
                            # 높이 방향 오버랩 적용
                            split_y = max(0, y + j * split_height - (j > 0) * h_overlap)
                            if j == h_splits - 1:
                                split_h = y + h - split_y
                            else:
                                split_h = split_height + h_overlap
                                
                            result.append([split_x, split_y, split_w, split_h])
                    else:
                        result.append([split_x, y, split_w, h])
            elif h > max_height:
                # 세로로 긴 영역 분할 (더 공격적)
                num_splits = max(3, h // (max_height // 2))
                split_height = h // num_splits
                v_overlap = int(split_height * 0.1)
                
                for i in range(num_splits):
                    split_y = max(0, y + i * split_height - (i > 0) * v_overlap)
                    if i == num_splits - 1:
                        split_h = y + h - split_y
                    else:
                        split_h = split_height + v_overlap
                        
                    result.append([x, split_y, w, split_h])
            else:
                # 면적 기반 분할 기준 강화
                if w * h > max_width * max_height:  # 임계값 감소
                    w_splits = max(2, int(w / (max_width * 0.7)))
                    h_splits = max(2, int(h / (max_height * 0.7)))
                    
                    split_width = w // w_splits
                    split_height = h // h_splits
                    w_overlap = int(split_width * 0.1)
                    h_overlap = int(split_height * 0.1)
                    
                    for i in range(w_splits):
                        for j in range(h_splits):
                            split_x = max(0, x + i * split_width - (i > 0) * w_overlap)
                            split_y = max(0, y + j * split_height - (j > 0) * h_overlap)
                            
                            if i == w_splits - 1:
                                split_w = x + w - split_x
                            else:
                                split_w = split_width + w_overlap
                                
                            if j == h_splits - 1:
                                split_h = y + h - split_y
                            else:
                                split_h = split_height + h_overlap
                                
                            result.append([split_x, split_y, split_w, split_h])
                else:
                    result.append(bbox)
        
        # 크기가 아주 작은 영역 필터링하면서 너무 큰 영역도 추가 필터링
        filtered_result = []
        for x, y, w, h in result:
            if w >= 8 and h >= 8 and w <= max_width*1.5 and h <= max_height*1.5:
                filtered_result.append([x, y, w, h])
                
        return filtered_result
    
    # 큰 텍스트 영역 분할 (단어 단위로)
    split_bboxes = split_large_text_regions(bboxes)
    
    # 원본 이미지에 경계 상자 그리기
    image_with_boxes = image.copy()
    
    # 다양한 색상으로 텍스트 영역 표시
    colors = [
        (0, 255, 0),    # 녹색
        (255, 0, 0),    # 빨강
        (0, 0, 255),    # 파랑
        (255, 255, 0),  # 노랑
        (255, 0, 255),  # 마젠타
        (0, 255, 255),  # 시안
        (128, 0, 0),    # 어두운 빨강
        (0, 128, 0),    # 어두운 녹색
        (0, 0, 128)     # 어두운 파랑
    ]
    
    # 원본 이미지에 분할된 경계 상자 그리기
    for i, bbox in enumerate(split_bboxes):
        x, y, w, h = bbox
        color = colors[i % len(colors)]
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
    
    # 결과 저장
    output_image_path = os.path.join(output_dir, 'text_detection_result.jpg')
    plt.figure(figsize=(15, 10))
    plt.imshow(image_with_boxes)
    plt.title('CRAFT Text Detection Result (Word Level)')
    plt.axis('off')
    plt.savefig(output_image_path)
    plt.close()
    
    # 히트맵 저장
    heatmap = craft_detector.visualize_detection(image, prediction_result=prediction_result)
    output_heatmap_path = os.path.join(output_dir, 'text_detection_heatmap.jpg')
    plt.figure(figsize=(15, 10))
    plt.imshow(heatmap)
    plt.title('CRAFT Text Detection Heatmap')
    plt.axis('off')
    plt.savefig(output_heatmap_path)
    plt.close()
    
    print(f"텍스트 감지 결과 저장 완료: {output_dir}, 감지된 단어 수: {len(split_bboxes)}")

def visualize_patches(image_path, output_dir, patch_size=(224, 224), craft_detector=None):
    """
    이미지에서 텍스트 기반 패치 추출 및 시각화 (단어 단위)
    
    Args:
        image_path: 이미지 파일 경로
        output_dir: 출력 디렉토리
        patch_size: 패치 크기 (width, height)
        craft_detector: CraftTextDetector 인스턴스
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # CRAFT 텍스트 감지기 초기화
    if craft_detector is None:
        craft_detector = CraftTextDetector()
    
    # 텍스트 영역 감지
    prediction_result = craft_detector.detect_text_regions(image)
    
    # 다각형을 경계 상자로 변환
    bboxes = craft_detector.convert_polys_to_bboxes(prediction_result["boxes"])
    
    # 텍스트 영역 분할 함수 (CraftMILDataset에서 가져온 로직)
    def split_large_text_regions(bboxes, max_width=200, max_height=80):
        """큰 텍스트 영역을 개별 단어 단위로 분할"""
        result = []
        for bbox in bboxes:
            x, y, w, h = bbox
            # 단어 수준으로 더 작게 분할하기 위해 최대 크기 제한 적용
            if w > max_width:
                # 가로로 긴 영역을 더 작은 단위로 분할
                num_splits = max(2, w // max_width)  # 덜 공격적인 분할
                split_width = w // num_splits
                overlap = int(split_width * 0.15)  # 오버랩 증가
                
                for i in range(num_splits):
                    # 오버랩을 고려한 시작점과 너비 계산
                    split_x = max(0, x + i * split_width - (i > 0) * overlap)
                    if i == num_splits - 1:  # 마지막 분할
                        split_w = x + w - split_x
                    else:
                        split_w = split_width + overlap
                    
                    # 높이도 크면 추가 분할
                    if h > max_height:
                        h_splits = max(2, h // max_height)
                        split_height = h // h_splits
                        h_overlap = int(split_height * 0.15)
                        
                        for j in range(h_splits):
                            # 높이 방향 오버랩 적용
                            split_y = max(0, y + j * split_height - (j > 0) * h_overlap)
                            if j == h_splits - 1:
                                split_h = y + h - split_y
                            else:
                                split_h = split_height + h_overlap
                                
                            result.append([split_x, split_y, split_w, split_h])
                    else:
                        result.append([split_x, y, split_w, h])
            elif h > max_height:
                # 세로로 긴 영역 분할
                num_splits = max(2, h // max_height)
                split_height = h // num_splits
                v_overlap = int(split_height * 0.15)
                
                for i in range(num_splits):
                    split_y = max(0, y + i * split_height - (i > 0) * v_overlap)
                    if i == num_splits - 1:
                        split_h = y + h - split_y
                    else:
                        split_h = split_height + v_overlap
                        
                    result.append([x, split_y, w, split_h])
            else:
                # 면적이 큰 경우 추가 분할
                if w * h > max_width * max_height * 1.5:  # 임계값 증가
                    w_splits = max(2, int(w / max_width))
                    h_splits = max(2, int(h / max_height))
                    
                    split_width = w // w_splits
                    split_height = h // h_splits
                    w_overlap = int(split_width * 0.15)
                    h_overlap = int(split_height * 0.15)
                    
                    for i in range(w_splits):
                        for j in range(h_splits):
                            split_x = max(0, x + i * split_width - (i > 0) * w_overlap)
                            split_y = max(0, y + j * split_height - (j > 0) * h_overlap)
                            
                            if i == w_splits - 1:
                                split_w = x + w - split_x
                            else:
                                split_w = split_width + w_overlap
                                
                            if j == h_splits - 1:
                                split_h = y + h - split_y
                            else:
                                split_h = split_height + h_overlap
                                
                            result.append([split_x, split_y, split_w, split_h])
                else:
                    result.append(bbox)
                    
        # 크기가 아주 작은 영역 필터링
        filtered_result = []
        for x, y, w, h in result:
            if w >= 8 and h >= 8:  # 최소 크기 제한
                filtered_result.append([x, y, w, h])
                
        return filtered_result

    # NMS(Non-Maximum Suppression) 함수
    def non_max_suppression(bboxes, overlap_threshold=0.15):
        """비최대 억제를 통해 중복 경계 상자 제거"""
        if len(bboxes) == 0:
            return []
            
        # 면적으로 정렬
        areas = [w * h for x, y, w, h in bboxes]
        indices = np.argsort(areas)[::-1]
        
        selected_indices = []
        
        while len(indices) > 0:
            # 가장 큰 경계 상자 선택
            current_index = indices[0]
            selected_indices.append(current_index)
            
            # 다른 경계 상자와의 IoU 계산
            remaining_indices = []
            for idx in indices[1:]:
                # IoU 계산
                x1, y1, w1, h1 = bboxes[current_index]
                x2, y2, w2, h2 = bboxes[idx]
                
                # 교차 영역 계산
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right < x_left or y_bottom < y_top:
                    intersection_area = 0.0
                else:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                
                # 두 경계 상자 영역
                bbox1_area = w1 * h1
                bbox2_area = w2 * h2
                
                # IoU 계산
                iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
                
                if iou <= overlap_threshold:
                    remaining_indices.append(idx)
            
            indices = remaining_indices
        
        return [bboxes[i] for i in selected_indices]
    
    # 큰 텍스트 영역 분할 (단어 단위로)
    bboxes = split_large_text_regions(bboxes)
    
    # 필터링 및 패딩 적용
    padding_ratio = 0.05  # 패딩 비율 증가
    padded_bboxes = []
    h, w = image.shape[:2]
    min_text_area = 30  # 최소 텍스트 영역 크기 증가
    
    for bbox in bboxes:
        x, y, width, height = bbox
        
        # 너무 작은 영역 필터링
        if width * height < min_text_area:
            continue
        
        # 패딩 추가
        padding_w = max(2, int(width * padding_ratio))
        padding_h = max(2, int(height * padding_ratio))
        
        padded_x = max(0, x - padding_w)
        padded_y = max(0, y - padding_h)
        padded_width = min(w - padded_x, width + 2 * padding_w)
        padded_height = min(h - padded_y, height + 2 * padding_h)
        
        padded_bboxes.append([padded_x, padded_y, padded_width, padded_height])
    
    # 중복 제거 (NMS 적용)
    padded_bboxes = non_max_suppression(padded_bboxes)
    
    # 원본 이미지에 경계 상자 그리기
    image_with_boxes = image.copy()
    patches = []
    
    for i, bbox in enumerate(padded_bboxes):
        x, y, width, height = bbox
        cv2.rectangle(image_with_boxes, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # 패치 추출
        # 경계 조건 확인
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        width = max(1, min(width, w-x))
        height = max(1, min(height, h-y))
        
        patch = image[y:y+height, x:x+width]
        
        # 가로세로 비율 유지하며 리사이즈 (더 나은 방법)
        h_patch, w_patch = patch.shape[:2]
        target_w, target_h = patch_size
        
        # 확대율 제한 (너무 큰 확대 방지)
        scale_w = target_w / w_patch
        scale_h = target_h / h_patch
        
        # 확대율 제한 (최대 3배)
        max_scale = 3.0
        if scale_w > max_scale or scale_h > max_scale:
            # 확대율 제한 적용
            scale = min(max_scale, min(scale_w, scale_h))
            new_w = int(w_patch * scale)
            new_h = int(h_patch * scale)
            
            # 작은 패치는 중앙에 배치
            pad_left = (target_w - new_w) // 2
            pad_right = target_w - new_w - pad_left
            pad_top = (target_h - new_h) // 2
            pad_bottom = target_h - new_h - pad_top
            
            # 먼저 크기 조정
            resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 패딩 추가
            patch_resized = cv2.copyMakeBorder(
                resized, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
        else:
            # 기존 비율 유지 로직
            if h_patch / w_patch > target_h / target_w:  # 세로가 더 긴 경우
                new_h = target_h
                new_w = int(w_patch * (target_h / h_patch))
                # 너무 작은 경우 최소 크기 보장
                new_w = max(new_w, 20)
                resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                # 좌우 패딩 추가
                pad_left = (target_w - new_w) // 2
                pad_right = target_w - new_w - pad_left
                patch_resized = cv2.copyMakeBorder(
                    resized, 0, 0, pad_left, pad_right, 
                    cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
            else:  # 가로가 더 긴 경우
                new_w = target_w
                new_h = int(h_patch * (target_w / w_patch))
                # 너무 작은 경우 최소 크기 보장
                new_h = max(new_h, 20)
                resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                # 상하 패딩 추가
                pad_top = (target_h - new_h) // 2
                pad_bottom = target_h - new_h - pad_top
                patch_resized = cv2.copyMakeBorder(
                    resized, pad_top, pad_bottom, 0, 0, 
                    cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
            
        patches.append(patch_resized)
    
    # 결과 저장
    output_image_path = os.path.join(output_dir, 'patches_on_image.jpg')
    plt.figure(figsize=(15, 10))
    plt.imshow(image_with_boxes)
    plt.title('Extracted Text Patches (Word Level)')
    plt.axis('off')
    plt.savefig(output_image_path)
    plt.close()
    
    # 패치 그리드 시각화
    if patches:
        n_cols = min(5, len(patches))
        n_rows = (len(patches) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(n_cols * 3, n_rows * 3))
        for i, patch in enumerate(patches):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(patch)
            plt.title(f'Patch {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        output_patches_path = os.path.join(output_dir, 'extracted_patches.jpg')
        plt.savefig(output_patches_path)
        plt.close()
    
    print(f"패치 추출 결과 저장 완료: {output_dir}, 패치 수: {len(patches)}")

def test_craft_mil_dataset(root_dir, output_dir, num_samples=5, split='train'):
    """
    CraftMILDataset 테스트 및 시각화
    
    Args:
        root_dir: 데이터셋 루트 디렉토리
        output_dir: 출력 디렉토리
        num_samples: 시각화할 샘플 수
        split: 'train', 'val', 'test' 중 하나
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # CRAFT 텍스트 감지기 초기화
    craft_detector = CraftTextDetector()
    
    # CraftMILDataset 초기화
    dataset = CraftMILDataset(
        root_dir=root_dir,
        split=split,
        craft_detector=craft_detector
    )
    
    # 무작위 샘플 선택
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in indices:
        sample = dataset[idx]
        
        patches = sample['patches']
        bag_label = sample['bag_label'].item()
        authors = sample['authors']
        path = sample['path']
        
        # 출력 디렉토리 이름 생성
        sample_name = f"{split}_sample_{idx}"
        sample_output_dir = os.path.join(output_dir, sample_name)
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # 원본 이미지 로드 및 저장
        image = cv2.imread(os.path.join(root_dir, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(15, 10))
        plt.imshow(image)
        plt.title(f"Original Image - Label: {'Multi-author' if bag_label == 1 else 'Single-author'}, Authors: {', '.join(authors)}")
        plt.axis('off')
        plt.savefig(os.path.join(sample_output_dir, 'original_image.jpg'))
        plt.close()
        
        # 패치 그리드 시각화
        n_patches = patches.shape[0]
        if n_patches > 0:
            n_cols = min(5, n_patches)
            n_rows = (n_patches + n_cols - 1) // n_cols
            
            plt.figure(figsize=(n_cols * 3, n_rows * 3))
            for i in range(n_patches):
                patch = patches[i].numpy().transpose(1, 2, 0)  # CHW -> HWC
                patch = np.clip(patch, 0, 1)
                
                plt.subplot(n_rows, n_cols, i + 1)
                plt.imshow(patch)
                plt.title(f'Patch {i+1}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(sample_output_dir, 'mil_patches.jpg'))
            plt.close()
        
        # 메타데이터 저장
        metadata = {
            'sample_idx': idx,
            'bag_label': bag_label,
            'authors': authors,
            'path': path,
            'num_patches': n_patches
        }
        
        with open(os.path.join(sample_output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"샘플 {idx} 시각화 완료: {sample_output_dir}")

def test_craft_mil_dataloader(root_dir, output_dir, batch_size=4, split='train'):
    """
    CraftMILDataloader 테스트
    
    Args:
        root_dir: 데이터셋 루트 디렉토리
        output_dir: 출력 디렉토리
        batch_size: 배치 크기
        split: 'train', 'val', 'test' 중 하나
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # CRAFT 텍스트 감지기 초기화
    craft_detector = CraftTextDetector()
    
    # 데이터 로더 초기화
    dataloader = get_craft_mil_dataloader(
        root_dir=root_dir,
        batch_size=batch_size,
        split=split,
        craft_detector=craft_detector,
        num_workers=0  # 디버깅을 위해 0으로 설정
    )
    
    # 첫 번째 배치 가져오기
    batch = next(iter(dataloader))
    
    # 배치 정보 출력
    patches = batch['patches']
    patch_counts = batch['patch_counts']
    bag_labels = batch['bag_labels']
    authors = batch['authors']
    paths = batch['paths']
    
    print(f"배치 크기: {len(patches)}")
    print(f"샘플별 패치 수: {patch_counts}")
    print(f"가방 레이블: {bag_labels}")
    
    # 배치 정보 저장
    batch_info = {
        'batch_size': len(patches),
        'patch_counts': patch_counts,
        'bag_labels': bag_labels.tolist(),
        'authors': authors,
        'paths': paths
    }
    
    with open(os.path.join(output_dir, 'batch_info.json'), 'w', encoding='utf-8') as f:
        json.dump(batch_info, f, indent=2, ensure_ascii=False)
    
    print(f"배치 정보 저장 완료: {os.path.join(output_dir, 'batch_info.json')}")

def main(args=None):
    # Jupyter 노트북 환경에서 실행 중인지 확인
    is_jupyter = False
    try:
        import IPython
        if IPython.get_ipython() is not None:
            is_jupyter = True
    except (ImportError, NameError):
        pass  # IPython이 설치되지 않았거나 IPython 환경이 아님
    
    if args is None:
        if is_jupyter:
            # Jupyter 노트북에서 실행 시 기본값 사용
            print("Jupyter 노트북 환경에서 실행 중 - 기본 인자 사용")
            args = argparse.Namespace(
                root_dir='mil_dataset',
                output_dir='craft_mil_results',
                split='train',
                test_mode='all',
                sample_image=None,
                num_samples=5
            )
        else:
            # 명령줄에서 실행 시 argparse 사용
            parser = argparse.ArgumentParser(description='CRAFT 텍스트 감지 및 MIL 데이터셋 테스트')
            parser.add_argument('--root_dir', type=str, default='mil_dataset', help='데이터셋 루트 디렉토리')
            parser.add_argument('--output_dir', type=str, default='craft_mil_results', help='출력 디렉토리')
            parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='데이터셋 분할')
            parser.add_argument('--test_mode', type=str, default='all', choices=['detection', 'patches', 'dataset', 'dataloader', 'all'], help='테스트 모드')
            parser.add_argument('--sample_image', type=str, help='단일 이미지 테스트를 위한 이미지 경로')
            parser.add_argument('--num_samples', type=int, default=5, help='데이터셋 테스트를 위한 샘플 수')
            
            args = parser.parse_args()
    
    # CRAFT 텍스트 감지기 초기화
    craft_detector = CraftTextDetector()
    
    # 테스트 모드에 따라 다른 기능 실행
    if args.test_mode == 'detection' or args.test_mode == 'all':
        if args.sample_image:
            # 단일 이미지에 대한 텍스트 감지 테스트
            detection_output_dir = os.path.join(args.output_dir, 'text_detection')
            visualize_text_detection(args.sample_image, detection_output_dir, craft_detector)
        else:
            # 데이터셋에서 무작위 이미지 선택
            with open(os.path.join(args.root_dir, 'splits', f'{args.split}.json'), 'r', encoding='utf-8') as f:
                samples = json.load(f)
            
            sample = random.choice(samples)
            image_path = os.path.join(args.root_dir, sample['img_path'])
            
            detection_output_dir = os.path.join(args.output_dir, 'text_detection')
            visualize_text_detection(image_path, detection_output_dir, craft_detector)
    
    if args.test_mode == 'patches' or args.test_mode == 'all':
        if args.sample_image:
            # 단일 이미지에 대한 패치 추출 테스트
            patches_output_dir = os.path.join(args.output_dir, 'patches')
            visualize_patches(args.sample_image, patches_output_dir, craft_detector=craft_detector)
        else:
            # 데이터셋에서 무작위 이미지 선택
            with open(os.path.join(args.root_dir, 'splits', f'{args.split}.json'), 'r', encoding='utf-8') as f:
                samples = json.load(f)
            
            sample = random.choice(samples)
            image_path = os.path.join(args.root_dir, sample['img_path'])
            
            patches_output_dir = os.path.join(args.output_dir, 'patches')
            visualize_patches(image_path, patches_output_dir, craft_detector=craft_detector)
    
    if args.test_mode == 'dataset' or args.test_mode == 'all':
        # CraftMILDataset 테스트
        dataset_output_dir = os.path.join(args.output_dir, 'dataset')
        test_craft_mil_dataset(args.root_dir, dataset_output_dir, args.num_samples, args.split)
    
    if args.test_mode == 'dataloader' or args.test_mode == 'all':
        # CraftMILDataloader 테스트
        dataloader_output_dir = os.path.join(args.output_dir, 'dataloader')
        test_craft_mil_dataloader(args.root_dir, dataloader_output_dir, batch_size=4, split=args.split)

if __name__ == '__main__':
    main() 