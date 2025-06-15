# 2번

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from pathlib import Path
import random
from tqdm import tqdm

# 상대 경로 대신 절대 경로 사용
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from craft_utils import CraftTextDetector

class CraftMILDataset(Dataset):
    """
    CRAFT 텍스트 감지를 사용하여 텍스트 영역 기반 패치를 추출하는 MIL 데이터셋 클래스
    """
    
    def __init__(self, 
                 root_dir, 
                 split='train',
                 patch_size=(224, 224),
                 transform=None,
                 max_patches_per_bag=100,
                 min_text_area=25,  # 더 작은 텍스트 영역까지 포함
                 padding_ratio=0.03,  # 패딩 비율 최적화
                 overlap_threshold=0.15,  # 더 엄격한 중복 제거
                 craft_detector=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 precompute_patches=False,
                 cache_dir=None,
                 enable_merge=False):  # 병합 비활성화 (단어 단위 유지)
        """
        Args:
            root_dir: MIL 데이터셋 루트 디렉토리
            split: 'train', 'val', 'test' 중 하나
            patch_size: 추출할 패치 크기 (width, height)
            transform: 패치에 적용할 변환 함수
            max_patches_per_bag: 가방(이미지)당 최대 패치 수
            min_text_area: 고려할 텍스트 영역의 최소 크기 (픽셀 단위)
            padding_ratio: 텍스트 경계 상자 주변에 추가할 패딩 비율
            overlap_threshold: 패치 중복 제거를 위한 IoU 임계값
            craft_detector: 사용할 CraftTextDetector 인스턴스
            device: 'cuda' 또는 'cpu'
            precompute_patches: 사전 패치 계산 여부
            cache_dir: 사전 계산된 패치를 저장할 캐시 디렉토리
            enable_merge: 인접 텍스트 상자 병합 활성화 여부
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.transform = transform
        self.max_patches_per_bag = max_patches_per_bag
        self.min_text_area = min_text_area
        self.padding_ratio = padding_ratio
        self.overlap_threshold = overlap_threshold
        self.device = device
        self.precompute_patches = precompute_patches
        self.enable_merge = enable_merge  # 병합 기능 활성화 여부
        
        # 분할 파일 로드
        split_file = self.root_dir / 'splits' / f'{split}.json'
        if not split_file.exists():
            raise FileNotFoundError(f"분할 파일이 존재하지 않습니다: {split_file}")
            
        with open(split_file, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
            
        print(f"MIL {split} 데이터셋 로드: {len(self.samples)}개 샘플")
        
        # CRAFT 텍스트 감지기 초기화
        if craft_detector is None:
            self.craft_detector = CraftTextDetector(device=device)
        else:
            self.craft_detector = craft_detector
            
        # 캐시 디렉토리 설정
        if precompute_patches:
            if cache_dir is None:
                self.cache_dir = self.root_dir / f'patch_cache_{split}'
            else:
                self.cache_dir = Path(cache_dir)
            
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"패치 캐시 디렉토리: {self.cache_dir}")
            
            # 패치 사전 계산
            self._precompute_all_patches()
    
    def _precompute_all_patches(self):
        """모든 샘플에 대한 패치를 사전 계산하고 캐시"""
        print("모든 이미지에 대한 패치 사전 계산 중...")
        
        for idx in tqdm(range(len(self.samples))):
            sample_info = self.samples[idx]
            img_path = self.root_dir / sample_info['img_path']
            
            # 이미지 로드
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 패치 추출
            patches, bboxes = self._extract_patches(image)
            
            # 캐시 저장
            if len(patches) > 0:
                cache_path = self.cache_dir / f"{idx}_patches.npz"
                np.savez_compressed(
                    cache_path, 
                    patches=np.array(patches),
                    bboxes=np.array(bboxes)
                )
    
    def _split_large_text_regions(self, bboxes, max_width=100, max_height=40):
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
    
    def _merge_adjacent_bboxes(self, bboxes, max_gap=10, max_height_diff=10):
        """인접한 텍스트 상자를 병합하여 글자 잘림 방지"""
        if len(bboxes) <= 1:
            return bboxes
            
        # x 좌표로 정렬
        sorted_bboxes = sorted(bboxes, key=lambda box: box[0])
        
        merged = []
        current_box = sorted_bboxes[0]
        
        for next_box in sorted_bboxes[1:]:
            curr_x, curr_y, curr_w, curr_h = current_box
            next_x, next_y, next_w, next_h = next_box
            
            # x 방향으로 인접하고 y 좌표가 비슷한지 확인
            is_x_adjacent = (next_x - (curr_x + curr_w)) <= max_gap
            is_y_similar = abs(curr_y - next_y) <= max_height_diff and abs((curr_y + curr_h) - (next_y + next_h)) <= max_height_diff
            
            # 인접한 상자 병합
            if is_x_adjacent and is_y_similar:
                # 새로운 병합된 박스 생성
                merged_x = curr_x
                merged_y = min(curr_y, next_y)
                merged_w = max(curr_x + curr_w, next_x + next_w) - merged_x
                merged_h = max(curr_y + curr_h, next_y + next_h) - merged_y
                
                current_box = [merged_x, merged_y, merged_w, merged_h]
            else:
                merged.append(current_box)
                current_box = next_box
        
        # 마지막 상자 추가
        merged.append(current_box)
        return merged
    
    def _extract_patches(self, image):
        """
        이미지에서 텍스트 기반 패치 추출 (단어 수준)
        
        Args:
            image: RGB 이미지 (NumPy 배열)
            
        Returns:
            tuple: (patches, bboxes)
                - patches: 추출된 패치 목록
                - bboxes: 패치에 해당하는 경계 상자 목록 
        """
        # 이미지 높이와 너비
        h, w = image.shape[:2]
        
        # 텍스트 영역 감지
        prediction_result = self.craft_detector.detect_text_regions(image)
        
        # 다각형을 경계 상자로 변환
        original_bboxes = self.craft_detector.convert_polys_to_bboxes(prediction_result["boxes"])
        
        # 큰 텍스트 영역 분할 (단어 단위로)
        original_bboxes = self._split_large_text_regions(original_bboxes)
        
        # 인접한 텍스트 상자 병합 (활성화된 경우에만)
        if self.enable_merge:
            original_bboxes = self._merge_adjacent_bboxes(original_bboxes)
        
        # 필터링 및 패딩 적용
        padded_bboxes = []
        for bbox in original_bboxes:
            x, y, width, height = bbox
            
            # 너무 작은 영역 필터링
            if width * height < self.min_text_area:
                continue
                
            # 패딩 추가
            padding_w = max(2, int(width * self.padding_ratio))
            padding_h = max(2, int(height * self.padding_ratio))
            
            # 패딩이 적용된 경계 상자
            padded_x = max(0, x - padding_w)
            padded_y = max(0, y - padding_h)
            padded_width = min(w - padded_x, width + 2 * padding_w)
            padded_height = min(h - padded_y, height + 2 * padding_h)
            
            padded_bboxes.append([padded_x, padded_y, padded_width, padded_height])
        
        # 중복 제거 (NMS 적용)
        filtered_bboxes = self._non_max_suppression(padded_bboxes)
        
        # 경계 상자를 기반으로 패치 추출
        patches = []
        final_bboxes = []
        
        for bbox in filtered_bboxes:
            x, y, width, height = bbox
            
            # 경계 조건 확인
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            width = max(1, min(width, w-x))
            height = max(1, min(height, h-y))
            
            # 패치 추출
            patch = image[y:y+height, x:x+width]
            
            # 패치 품질 검증 - 텍스트 비율 계산 및 위치 확인
            if not self._is_valid_patch(patch):
                continue
            
            # 텍스트 중심 크롭 - 텍스트를 중앙에 위치시키기
            patch = self._center_text_in_patch(patch)
            
            # 고정 크기로 리사이즈 (가로세로 비율 유지)
            h_patch, w_patch = patch.shape[:2]
            target_w, target_h = self.patch_size
            
            # 확대율 제한 (최대 2.5배로 더 제한)
            scale_w = target_w / w_patch
            scale_h = target_h / h_patch
            
            # 확대율 제한 (최대 2.5배)
            max_scale = 2.5
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
                
                # 먼저 크기 조정 (Lanczos 보간법 사용)
                resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                # 패딩 추가 (흰색 배경)
                patch_resized = cv2.copyMakeBorder(
                    resized, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
            else:
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
            final_bboxes.append(bbox)
            
            # 최대 패치 수 제한
            if len(patches) >= self.max_patches_per_bag:
                break
        
        return patches, final_bboxes
    
    def _non_max_suppression(self, bboxes):
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
                iou = self._calculate_iou(bboxes[current_index], bboxes[idx])
                if iou <= self.overlap_threshold:
                    remaining_indices.append(idx)
            
            indices = remaining_indices
        
        return [bboxes[i] for i in selected_indices]
    
    def _calculate_iou(self, bbox1, bbox2):
        """두 경계 상자 간의 IoU(Intersection over Union) 계산"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 교차 영역 계산
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 두 경계 상자 영역
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        
        # IoU 계산
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        
        return iou
    
    def _is_valid_patch(self, patch, min_text_ratio=0.05, max_bg_ratio=0.8):
        """
        패치가 유효한지 검증 (텍스트 비율 및 위치 확인)
        
        Args:
            patch: 검증할 패치 이미지
            min_text_ratio: 최소 텍스트 비율 (0.01 → 0.05로 증가)
            max_bg_ratio: 최대 배경 비율 (0.9 → 0.8로 감소)
            
        Returns:
            bool: 패치 유효성 여부
        """
        if patch.size == 0 or patch.shape[0] < 10 or patch.shape[1] < 10:
            return False
            
        # 그레이스케일 변환
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # 이진화 - 이진화 임계값 낮춤 (더 많은 텍스트 포함)
        _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        
        # 텍스트 비율 계산 (어두운 픽셀 비율)
        text_ratio = 1.0 - np.sum(binary == 255) / binary.size
        
        # 너무 적은 텍스트 또는 너무 많은 배경이 있는 패치 필터링
        if text_ratio < min_text_ratio or text_ratio > (1 - max_bg_ratio):
            return False
            
        # 텍스트 위치 검사 - 패치 가장자리에 텍스트가 집중되어 있는지 확인
        # 패치를 9개 영역으로 나누고 가장자리 영역의 텍스트 비율 검사
        h, w = binary.shape
        h_third, w_third = h // 3, w // 3
        
        # 중앙 영역
        center = binary[h_third:2*h_third, w_third:2*w_third]
        center_text_ratio = 1.0 - np.sum(center == 255) / center.size
        
        # 가장자리 영역의 텍스트 비율이 중앙보다 크게 높으면 가장자리에 잘린 텍스트가 있을 가능성
        edge_regions = [
            binary[:h_third, :],                # 상단
            binary[2*h_third:, :],              # 하단
            binary[:, :w_third],                # 좌측
            binary[:, 2*w_third:]               # 우측
        ]
        
        for edge in edge_regions:
            edge_text_ratio = 1.0 - np.sum(edge == 255) / edge.size
            # 가장자리 텍스트 비율이 중앙의 2배 이상이면 잘린 텍스트로 간주
            if edge_text_ratio > center_text_ratio * 2 and edge_text_ratio > 0.1:
                return False
                
        return True
    
    def _center_text_in_patch(self, patch):
        """
        패치 내의 텍스트를 중앙에 위치시키기
        
        Args:
            patch: 원본 패치 이미지
            
        Returns:
            numpy.ndarray: 텍스트가 중앙에 위치한 패치
        """
        if patch.size == 0:
            return patch
            
        # 그레이스케일 변환
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # 이진화
        _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
        
        # 텍스트 영역 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return patch
            
        # 모든 텍스트 영역을 포함하는 경계 상자 찾기
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # 경계가 실제로 존재하는지 확인
        if x_min >= x_max or y_min >= y_max:
            return patch
            
        # 텍스트 영역의 중심점
        text_center_x = (x_min + x_max) // 2
        text_center_y = (y_min + y_max) // 2
        
        # 패치의 중심점
        patch_center_x = patch.shape[1] // 2
        patch_center_y = patch.shape[0] // 2
        
        # 이동 거리 계산 (텍스트 중심을 패치 중심으로)
        shift_x = patch_center_x - text_center_x
        shift_y = patch_center_y - text_center_y
        
        # 중심 이동이 작으면 원본 반환
        if abs(shift_x) < 5 and abs(shift_y) < 5:
            return patch
            
        # 이동 변환 행렬
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        # 이미지 이동
        centered_patch = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]), 
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        return centered_patch
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        데이터셋에서 샘플 가져오기
        
        Returns:
            dict: 다음 키를 포함하는 샘플:
                - patches: 추출된 패치 텐서 (N, C, H, W)
                - bag_label: 가방 레이블 (1 = 다중 작성자, 0 = 단일 작성자)
                - authors: 작성자 ID 목록
                - path: 원본 이미지 경로
        """
        sample_info = self.samples[idx]
        
        # 사전 계산된 패치 사용 여부
        if self.precompute_patches:
            cache_path = self.cache_dir / f"{idx}_patches.npz"
            if cache_path.exists():
                # 캐시에서 패치 로드
                cached_data = np.load(cache_path)
                patches = cached_data['patches']
                bboxes = cached_data['bboxes']
            else:
                # 이미지 로드
                img_path = self.root_dir / sample_info['img_path']
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 패치 추출
                patches, bboxes = self._extract_patches(image)
        else:
            # 이미지 로드
            img_path = self.root_dir / sample_info['img_path']
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 패치 추출
            patches, bboxes = self._extract_patches(image)
        
        # 충분한 패치가 없으면 이미지를 그리드로 분할
        if len(patches) < 1:
            # 이미지 로드
            img_path = self.root_dir / sample_info['img_path']
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            h, w = image.shape[:2]
            grid_h, grid_w = self.patch_size
            
            # 그리드 위치 계산
            grid_patches = []
            grid_bboxes = []
            
            for y in range(0, h, grid_h):
                for x in range(0, w, grid_w):
                    if len(grid_patches) >= self.max_patches_per_bag:
                        break
                    
                    # 패치 경계가 이미지를 벗어나지 않도록 조정
                    end_y = min(y + grid_h, h)
                    end_x = min(x + grid_w, w)
                    
                    # 패치 추출
                    patch = image[y:end_y, x:end_x]
                    
                    # 고정 크기로 리사이즈
                    patch_resized = cv2.resize(patch, self.patch_size)
                    
                    grid_patches.append(patch_resized)
                    grid_bboxes.append([x, y, end_x - x, end_y - y])
            
            patches = grid_patches
            bboxes = grid_bboxes
        
        # 변환 적용
        transformed_patches = []
        for patch in patches:
            # PIL 이미지로 변환 (변환 함수가 PIL 이미지를 요구하는 경우)
            patch_pil = Image.fromarray(patch)
            
            if self.transform:
                transformed_patch = self.transform(patch_pil)
            else:
                # 기본 변환 - 정규화 및 텐서 변환
                transformed_patch = np.array(patch_pil) / 255.0
                transformed_patch = np.transpose(transformed_patch, (2, 0, 1))  # HWC -> CHW
                transformed_patch = torch.from_numpy(transformed_patch).float()
            
            transformed_patches.append(transformed_patch)
        
        # 패치를 하나의 텐서로 결합
        if len(transformed_patches) > 0:
            patches_tensor = torch.stack(transformed_patches)
        else:
            # 패치가 없으면 빈 텐서 생성
            patches_tensor = torch.zeros((0, 3, self.patch_size[1], self.patch_size[0]))
        
        # 가방 레이블 (다중 작성자 = 1, 단일 작성자 = 0)
        bag_label = 1 if sample_info['is_multi_author'] else 0
        
        return {
            'patches': patches_tensor,
            'bag_label': torch.tensor(bag_label, dtype=torch.float32),
            'authors': sample_info['authors'],
            'bboxes': bboxes,
            'path': str(sample_info['img_path'])
        }

def get_craft_mil_dataloader(root_dir, batch_size=4, split='train', patch_size=(224, 224), 
                            num_workers=4, craft_detector=None, transform=None,
                            max_patches_per_bag=100, precompute_patches=False):
    """
    CRAFT 기반 MIL 데이터 로더 생성 함수
    
    Args:
        root_dir: 데이터셋 루트 디렉토리
        batch_size: 배치 크기
        split: 'train', 'val', 'test' 중 하나
        patch_size: 패치 크기 (width, height)
        num_workers: 데이터 로더 워커 수
        craft_detector: 사용할 CraftTextDetector 인스턴스
        transform: 적용할 변환
        max_patches_per_bag: 가방당 최대 패치 수
        precompute_patches: 패치 사전 계산 여부
        
    Returns:
        DataLoader: CRAFT 기반 MIL 데이터 로더
    """
    dataset = CraftMILDataset(
        root_dir=root_dir,
        split=split,
        patch_size=patch_size,
        transform=transform,
        craft_detector=craft_detector,
        max_patches_per_bag=max_patches_per_bag,
        precompute_patches=precompute_patches
    )
    
    shuffle = (split == 'train')
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_mil_patches  # 가변 길이 패치 처리를 위한 사용자 정의 collate 함수
    )

def collate_mil_patches(batch):
    """
    가변 길이 MIL 패치 묶기 함수
    
    Args:
        batch: 배치 항목 목록
        
    Returns:
        dict: 배치 데이터
    """
    batch_data = {
        'patches': [item['patches'] for item in batch],
        'patch_counts': [len(item['patches']) for item in batch],
        'bag_labels': torch.tensor([item['bag_label'] for item in batch]),
        'authors': [item['authors'] for item in batch],
        'bboxes': [item['bboxes'] for item in batch],
        'paths': [item['path'] for item in batch]
    }
    
    return batch_data 