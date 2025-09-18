#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Craft 기반 단어 패치화 파이프라인

이 스크립트는 CRAFT 텍스트 감지 알고리즘을 사용하여 문서 이미지에서
단어 단위 패치를 추출하는 최적화된 파이프라인을 구현합니다.
하나의 패치에 하나의 단어가 온전히 포함되도록 최적화되었습니다.
"""

import os
import cv2
import numpy as np
import torch
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import time

# CRAFT 모델 관련 임포트
try:
    from craft_text_detector import Craft
except ImportError:
    print("CRAFT 텍스트 감지기 패키지가 설치되지 않았습니다.")
    print("pip install craft-text-detector 명령으로 설치하세요.")
    exit(1)

class CraftTextDetector:
    """CRAFT 텍스트 감지기 래퍼 클래스"""
    
    def __init__(self, 
                 text_threshold=0.3,  # 0.4에서 더 낮춤
                 link_threshold=0.5,  # 0.7에서 더 낮춤
                 low_text=0.2,        # 0.3에서 더 낮춤
                 long_size=1536,       # 1920에서 감소
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        CRAFT 텍스트 감지기 초기화
        
        Args:
            text_threshold (float): 텍스트 신뢰도 임계값
            link_threshold (float): 링크 신뢰도 임계값
            low_text (float): 낮은 텍스트 신뢰도 임계값
            long_size (int): 입력 이미지의 긴 쪽 크기
            device (str): 사용할 장치 ('cuda' 또는 'cpu')
        """
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.long_size = long_size
        self.device = device
        
        # CRAFT 모델 초기화
        self.craft = Craft(
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            low_text=self.low_text,
            long_size=self.long_size
        )
        
        # 수동으로 장치 설정 (API에서 직접 지원하지 않는 경우)
        if hasattr(self.craft, 'net') and hasattr(self.craft.net, 'to'):
            try:
                self.craft.net.to(self.device)
            except Exception as e:
                print(f"장치 설정 중 오류 발생: {e}")
                print("CPU 사용으로 대체합니다.")
                self.device = 'cpu'
                self.craft.net.to('cpu')
    
    def detect_text_regions(self, image):
        """
        이미지에서 텍스트 영역 감지
        
        Args:
            image (numpy.ndarray): 입력 이미지 (RGB)
            
        Returns:
            dict: 감지 결과 (boxes, polys, heatmap 등 포함)
        """
        # CRAFT 모델로 텍스트 영역 감지
        prediction_result = self.craft.detect_text(image)
        
        return prediction_result
    
    def convert_polys_to_bboxes(self, polys):
        """
        다각형을 경계 상자로 변환
        
        Args:
            polys (list): 다각형 좌표 리스트
            
        Returns:
            list: 경계 상자 리스트 [x, y, width, height]
        """
        bboxes = []
        for poly in polys:
            # 다각형의 경계 상자 계산
            x_min = np.min(poly[:, 0])
            y_min = np.min(poly[:, 1])
            x_max = np.max(poly[:, 0])
            y_max = np.max(poly[:, 1])
            
            # [x, y, width, height] 형식으로 변환
            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            bboxes.append(bbox)
        
        return bboxes


class OptimizedWordPatchExtractor:
    """최적화된 단어 패치 추출기 클래스"""
    
    def __init__(self,
                 patch_size=(224, 224),
                 padding_ratio=0.15,     # 0.03에서 증가
                 min_text_area=45,       # 25에서 증가
                 overlap_threshold=0.22,  # 0.15에서 증가
                 max_scale=1.5,          # 2.5에서 감소
                 enable_merge=True,      # False에서 변경
                 max_gap=18,             # 10에서 증가
                 max_height_diff=18,     # 10에서 증가
                 min_quality_score=60,
                 craft_detector=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        단어 패치 추출기 초기화
        
        Args:
            patch_size (tuple): 패치 크기 (너비, 높이)
            padding_ratio (float): 패딩 비율
            min_text_area (int): 최소 텍스트 영역 크기
            overlap_threshold (float): 중복 제거 임계값
            max_scale (float): 최대 확대 비율
            enable_merge (bool): 인접 텍스트 상자 병합 활성화 여부
            max_gap (int): 병합 시 최대 간격
            max_height_diff (int): 병합 시 최대 높이 차이
            min_quality_score (int): 최소 품질 점수
            craft_detector (CraftTextDetector): CRAFT 텍스트 감지기 인스턴스
            device (str): 사용할 장치 ('cuda' 또는 'cpu')
        """
        self.patch_size = patch_size
        self.padding_ratio = padding_ratio
        self.min_text_area = min_text_area
        self.overlap_threshold = overlap_threshold
        self.max_scale = max_scale
        self.enable_merge = enable_merge
        self.max_gap = max_gap
        self.max_height_diff = max_height_diff
        self.min_quality_score = min_quality_score
        self.device = device
        
        # CRAFT 텍스트 감지기 초기화
        if craft_detector is None:
            self.craft_detector = CraftTextDetector(device=device)
        else:
            self.craft_detector = craft_detector
    
    def preprocess_image(self, image):
        """
        이미지 전처리
        
        Args:
            image (numpy.ndarray): 입력 이미지
            
        Returns:
            numpy.ndarray: 전처리된 이미지
        """
        # 이미지가 그레이스케일인 경우 RGB로 변환
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA 이미지
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # 이미지 크기 확인 및 조정
        h, w = image.shape[:2]
        max_dim = 2048
        
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 대비 향상
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge((l, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # 노이즈 제거 (약한 블러) - 주석 처리
        # denoised = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
        # return denoised
        
        return enhanced_image # 블러 처리되지 않은 이미지 반환
    
    def word_level_segmentation(self, image, bboxes):
        """
        텍스트 영역을 단어 수준으로 세분화
        
        Args:
            image (numpy.ndarray): 입력 이미지
            bboxes (list): 경계 상자 리스트 [x, y, width, height]
            
        Returns:
            list: 단어 수준 경계 상자 리스트
        """
        word_bboxes = []
        
        for bbox in bboxes:
            x, y, w, h = bbox
            
            # 너무 작은 영역 필터링
            if w * h < self.min_text_area:
                continue
            
            # 텍스트 영역 추출
            text_region = image[y:y+h, x:x+w]
            
            # 크기 기반 분할 결정
            max_width = 180  # 100에서 증가
            max_height = 70  # 40에서 증가
            force_split_ratio = 4.0  # 3.0에서 증가
            
            if w > max_width or h > max_height or w/h > force_split_ratio or h/w > force_split_ratio:
                # 큰 텍스트 영역 또는 비정상적인 비율 -> 분할 필요
                
                # 그레이스케일 변환
                gray = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
                
                # 이진화
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # 수평 프로젝션 프로필 계산 (단어 간 간격 감지)
                h_proj = np.sum(binary, axis=0)
                
                # 간격 감지 (값이 0인 열이 연속으로 나타나는 구간)
                gaps = []
                in_gap = False
                gap_start = 0
                
                for i, val in enumerate(h_proj):
                    if val == 0 and not in_gap:
                        in_gap = True
                        gap_start = i
                    elif val > 0 and in_gap:
                        in_gap = False
                        gap_end = i
                        if gap_end - gap_start >= 5:  # 최소 간격 크기
                            gaps.append((gap_start, gap_end))
                
                # 간격을 기준으로 단어 분할
                if gaps:
                    prev_end = 0
                    for gap_start, gap_end in gaps:
                        if gap_start > prev_end:
                            # 간격 이전 영역을 단어로 추출
                            word_x = x + prev_end
                            word_w = gap_start - prev_end
                            word_bboxes.append([word_x, y, word_w, h])
                        prev_end = gap_end
                    
                    # 마지막 간격 이후 영역
                    if prev_end < w:
                        word_x = x + prev_end
                        word_w = w - prev_end
                        word_bboxes.append([word_x, y, word_w, h])
                else:
                    # 간격이 감지되지 않으면 원본 경계 상자 사용
                    word_bboxes.append(bbox)
            else:
                # 작은 텍스트 영역 -> 이미 단어 수준으로 간주
                word_bboxes.append(bbox)
        
        return word_bboxes
    
    def improved_merge_adjacent_bboxes(self, bboxes):
        """
        개선된 인접 텍스트 상자 병합 알고리즘
        
        Args:
            bboxes (list): 경계 상자 리스트 [x, y, width, height]
            
        Returns:
            list: 병합된 경계 상자 리스트
        """
        if not self.enable_merge or len(bboxes) <= 1:
            return bboxes
        
        # x 좌표로 정렬
        sorted_bboxes = sorted(bboxes, key=lambda box: box[0])
        
        # 병합 결과
        merged = []
        current_box = sorted_bboxes[0]
        
        for next_box in sorted_bboxes[1:]:
            curr_x, curr_y, curr_w, curr_h = current_box
            next_x, next_y, next_w, next_h = next_box
            
            # x 방향으로 인접한지 확인
            is_x_adjacent = (next_x - (curr_x + curr_w)) <= self.max_gap
            
            # y 좌표가 비슷한지 확인 (상단과 하단 모두)
            is_y_top_similar = abs(curr_y - next_y) <= self.max_height_diff
            is_y_bottom_similar = abs((curr_y + curr_h) - (next_y + next_h)) <= self.max_height_diff
            
            # 높이가 비슷한지 확인
            height_ratio = max(curr_h, next_h) / max(1, min(curr_h, next_h))
            is_height_similar = height_ratio <= 1.5
            
            # 텍스트 방향이 비슷한지 확인 (기울기 계산)
            curr_angle = np.arctan2(curr_h, curr_w) * 180 / np.pi
            next_angle = np.arctan2(next_h, next_w) * 180 / np.pi
            is_angle_similar = abs(curr_angle - next_angle) <= 15
            
            # 병합 조건: x 인접 + (y 상단 유사 또는 y 하단 유사) + 높이 유사 + 각도 유사
            if is_x_adjacent and (is_y_top_similar or is_y_bottom_similar) and is_height_similar and is_angle_similar:
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
    
    def _non_max_suppression(self, bboxes):
        """
        비최대 억제 (NMS) 적용
        
        Args:
            bboxes (list): 경계 상자 리스트 [x, y, width, height]
            
        Returns:
            list: 중복이 제거된 경계 상자 리스트
        """
        if len(bboxes) == 0:
            return []
        
        # 경계 상자를 [x1, y1, x2, y2] 형식으로 변환
        boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in bboxes])
        
        # 면적 계산
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 면적 기준 내림차순 정렬
        order = areas.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 현재 상자와 나머지 상자 간의 IoU 계산
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # 중복 제거
            inds = np.where(iou <= self.overlap_threshold)[0]
            order = order[inds + 1]
        
        # 원래 형식 [x, y, width, height]로 변환
        filtered_bboxes = [[boxes[i, 0], boxes[i, 1], 
                           boxes[i, 2] - boxes[i, 0], 
                           boxes[i, 3] - boxes[i, 1]] for i in keep]
        
        return filtered_bboxes
    
    def improved_center_text_in_patch(self, patch):
        """
        텍스트를 패치 중앙에 위치시키는 개선된 함수
        
        Args:
            patch (numpy.ndarray): 입력 패치
            
        Returns:
            numpy.ndarray: 텍스트가 중앙에 위치한 패치
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # 이진화 (적응형 임계값 적용)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 텍스트 영역 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return patch  # 텍스트 영역을 찾지 못한 경우 원본 반환
        
        # 모든 텍스트 영역을 포함하는 경계 상자 계산
        x_min, y_min = patch.shape[1], patch.shape[0]
        x_max, y_max = 0, 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:  # 작은 노이즈 무시
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # 텍스트 영역이 너무 작으면 원본 반환
        if x_max - x_min < 5 or y_max - y_min < 5:
            return patch
        
        # 텍스트 영역 추출
        text_region = patch[y_min:y_max, x_min:x_max]
        
        # 패치 크기
        h_patch, w_patch = patch.shape[:2]
        
        # 텍스트 영역 크기
        h_text, w_text = text_region.shape[:2]
        
        # 중앙 위치 계산
        pad_left = (w_patch - w_text) // 2
        pad_right = w_patch - w_text - pad_left
        pad_top = (h_patch - h_text) // 2
        pad_bottom = h_patch - h_text - pad_top
        
        # 패딩 추가 (흰색 배경)
        centered = cv2.copyMakeBorder(
            text_region, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
        return centered
    
    def optimal_resize(self, patch, target_size, original_size):
        """
        확대/축소 비율에 따라 최적의 보간법을 선택하여 리사이징
        
        Args:
            patch (numpy.ndarray): 입력 패치
            target_size (tuple): 목표 크기 (너비, 높이)
            original_size (tuple): 원본 크기 (높이, 너비)
            
        Returns:
            numpy.ndarray: 리사이징된 패치
        """
        h_orig, w_orig = original_size
        target_w, target_h = target_size
        
        # 확대/축소 비율 계산
        scale_w = target_w / w_orig
        scale_h = target_h / h_orig
        scale = min(scale_w, scale_h)
        
        # 확대/축소 비율에 따른 보간법 선택
        if scale > 1.5:  # 큰 확대
            # 확대 시 선명도 유지를 위해 INTER_CUBIC 사용
            interpolation = cv2.INTER_CUBIC
            # 확대율 제한
            scale = min(scale, self.max_scale)
        elif scale > 1.0:  # 작은 확대
            # 작은 확대에는 INTER_LINEAR가 효율적
            interpolation = cv2.INTER_LINEAR
        elif scale > 0.5:  # 작은 축소
            # 작은 축소에는 INTER_AREA가 효과적
            interpolation = cv2.INTER_AREA
        else:  # 큰 축소
            # 큰 축소에는 INTER_AREA가 모아레 방지에 효과적
            interpolation = cv2.INTER_AREA
        
        # 새 크기 계산 (확대율 제한 적용)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        
        # 리사이징
        resized = cv2.resize(patch, (new_w, new_h), interpolation=interpolation)
        
        # 패딩 추가
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        
        # 패딩 추가 (흰색 배경)
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
        return padded
    
    def enhance_text_clarity(self, patch):
        """
        텍스트 선명도 향상 처리
        
        Args:
            patch (numpy.ndarray): 입력 패치
            
        Returns:
            numpy.ndarray: 선명도가 향상된 패치
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # 이진화 (적응형 임계값)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 노이즈 제거
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 텍스트 영역 마스크 생성
        text_mask = cv2.bitwise_not(binary)
        
        # 원본 이미지에 마스크 적용
        result = patch.copy()
        for c in range(3):
            result[:, :, c] = cv2.bitwise_and(result[:, :, c], result[:, :, c], mask=binary)
        
        # 배경을 흰색으로 설정
        white_bg = np.ones_like(patch) * 255
        for c in range(3):
            white_bg[:, :, c] = cv2.bitwise_and(white_bg[:, :, c], white_bg[:, :, c], mask=text_mask)
        
        # 텍스트와 흰색 배경 합성
        enhanced = cv2.add(result, white_bg)
        
        # 선명도 향상 (언샵 마스킹)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        
        return sharpened
    
    def calculate_patch_quality_score(self, patch):
        """
        패치 품질 점수 계산 (0-100)
        
        Args:
            patch (numpy.ndarray): 입력 패치
            
        Returns:
            float: 품질 점수 (0-100)
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # 1. 대비 점수 (0-25)
        contrast = np.std(gray)
        contrast_score = min(25, contrast / 2)
        
        # 2. 텍스트 비율 점수 (0-25)
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text_ratio = np.sum(binary > 0) / binary.size
        
        # 이상적인 텍스트 비율은 10-30%
        if 0.1 <= text_ratio <= 0.3:
            text_ratio_score = 25
        elif 0.05 <= text_ratio < 0.1 or 0.3 < text_ratio <= 0.4:
            text_ratio_score = 15
        elif 0.01 <= text_ratio < 0.05 or 0.4 < text_ratio <= 0.5:
            text_ratio_score = 5
        else:
            text_ratio_score = 0
        
        # 3. 텍스트 중심성 점수 (0-25)
        h, w = gray.shape
        center_region = binary[h//4:3*h//4, w//4:3*w//4]
        center_text_ratio = np.sum(center_region > 0) / center_region.size
        total_text_ratio = np.sum(binary > 0) / binary.size
        
        if total_text_ratio > 0:
            center_score = min(25, 25 * (center_text_ratio / total_text_ratio))
        else:
            center_score = 0
        
        # 4. 선명도 점수 (0-25)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(25, laplacian_var / 20)
        
        # 총점 계산
        total_score = contrast_score + text_ratio_score + center_score + sharpness_score
        
        return total_score
    
    def enhanced_is_valid_patch(self, patch, min_text_ratio=0.01, max_text_ratio=0.7, min_contrast=10):
        """
        개선된 패치 품질 검증 함수
        
        Args:
            patch (numpy.ndarray): 입력 패치
            min_text_ratio (float): 최소 텍스트 비율
            max_text_ratio (float): 최대 텍스트 비율
            min_contrast (float): 최소 대비
            
        Returns:
            bool: 패치가 유효한지 여부
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # 대비 계산
        contrast = np.std(gray)
        if contrast < min_contrast:
            print(f"(실패: 대비 낮음 {contrast:.2f} < {min_contrast})", end="")
            return False
        
        # 이진화 (적응형 임계값 적용)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 텍스트 비율 계산
        text_pixels = np.sum(binary > 0)
        total_pixels = binary.size
        if total_pixels == 0: # Avoid division by zero
             print("(실패: 텍스트 픽셀 없음)", end="")
             return False
        text_ratio = text_pixels / total_pixels
        
        # 텍스트 비율 검증
        if text_ratio < min_text_ratio or text_ratio > max_text_ratio:
            print(f"(실패: 텍스트 비율 벗어남 {text_ratio:.2f} not in [{min_text_ratio}, {max_text_ratio}])", end="")
            return False
        
        # 텍스트 위치 검증 (중앙 영역에 텍스트가 있는지)
        h, w = binary.shape
        center_region = binary[h//4:3*h//4, w//4:3*w//4]
        center_text_pixels = np.sum(center_region > 0)
        
        if text_pixels == 0: # Avoid division by zero
             print("(실패: 중앙 텍스트 비율 계산 불가 - 텍스트 픽셀 없음)", end="")
             return False
        center_ratio = center_text_pixels / text_pixels
        if center_ratio < 0.3:
            print(f"(실패: 텍스트 중앙 집중도 낮음 {center_ratio:.2f} < 0.3)", end="")
            return False
        
        # 텍스트 연결성 검증
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        
        if num_contours > 30: # 기준값을 10에서 30으로 높임
            print(f"(실패: 텍스트 분산됨 - contour {num_contours} > 30)", end="")
            return False
        
        # 텍스트 선명도 검증
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 20: # 기준값을 50에서 20으로 낮춤
            print(f"(실패: 텍스트 흐릿함 {laplacian_var:.2f} < 20)", end="")
            return False
        
        return True  # 모든 검증 통과
    
    def extract_word_patch(self, image, bbox, target_size=None):
        """
        단어 중심 패치 추출 함수 (원본 크롭 + 단순 리사이징)
        
        Args:
            image (numpy.ndarray): 입력 이미지 (전처리된 이미지 아님!)
            bbox (list): 경계 상자 [x, y, width, height]
            target_size (tuple): 목표 크기 (너비, 높이), None이면 self.patch_size 사용
            
        Returns:
            numpy.ndarray: 추출 및 리사이징된 패치
        """
        if target_size is None:
            target_size = self.patch_size
            
        x, y, w, h = bbox
        
        # 경계 조건 확인 (원본 이미지 기준)
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w-1))
        y = max(0, min(y, img_h-1))
        w = max(1, min(w, img_w-x))
        h = max(1, min(h, img_h-y))
        
        # --- 원본 패치 추출 로직 변경 --- 
        # 최소한의 패딩만 적용 (선택적, 여기서는 0으로 설정)
        padding_w = 0 # max(5, int(w * 0.02)) # 필요시 작은 패딩 추가
        padding_h = 0 # max(5, int(h * 0.02))
        
        padded_x = max(0, x - padding_w)
        padded_y = max(0, y - padding_h)
        padded_w = min(img_w - padded_x, w + 2 * padding_w)
        padded_h = min(img_h - padded_y, h + 2 * padding_h)
        
        # 원본 이미지에서 직접 크롭
        patch = image[padded_y:padded_y+padded_h, padded_x:padded_x+padded_w]
        
        # 단순 리사이징 (선명도 유지를 위해 INTER_LINEAR 또는 INTER_CUBIC 고려)
        # 축소 시에는 INTER_AREA가 좋지만, 여기서는 품질 유지가 더 중요할 수 있음
        if patch.shape[0] > target_size[1] or patch.shape[1] > target_size[0]:
             interpolation = cv2.INTER_AREA # 축소
        else:
             interpolation = cv2.INTER_LINEAR # 확대 또는 동일 크기
             
        resized_patch = cv2.resize(patch, target_size, interpolation=interpolation)
        # --- 로직 변경 끝 --- 
        
        return resized_patch
    
    def extract_patches(self, image):
        """
        이미지에서 단어 패치 추출
        
        Args:
            image (numpy.ndarray): 입력 이미지
            
        Returns:
            tuple: (패치 리스트, 경계 상자 리스트)
        """
        print("\n--- 패치 추출 시작 ---")
        # 이미지 전처리 (CRAFT 감지용)
        preprocessed_image = self.preprocess_image(image)
        print(f"이미지 전처리 완료. 크기: {preprocessed_image.shape}")
        
        # 텍스트 영역 감지
        prediction_result = self.craft_detector.detect_text_regions(preprocessed_image)
        print(f"CRAFT 감지 완료. 감지된 다각형 수: {len(prediction_result['boxes'])}")
        
        # 다각형을 경계 상자로 변환
        bboxes = self.craft_detector.convert_polys_to_bboxes(prediction_result["boxes"])
        print(f"다각형 -> 경계 상자 변환 완료. 초기 경계 상자 수: {len(bboxes)}")
        
        # 단어 수준 세분화
        word_bboxes = self.word_level_segmentation(preprocessed_image, bboxes)
        print(f"단어 수준 세분화 완료. 단어 경계 상자 수: {len(word_bboxes)}")
        
        # 인접한 텍스트 상자 병합 (활성화된 경우에만)
        if self.enable_merge:
            merged_bboxes = self.improved_merge_adjacent_bboxes(word_bboxes)
            print(f"인접 박스 병합 완료. 병합 후 경계 상자 수: {len(merged_bboxes)}")
            word_bboxes = merged_bboxes # 병합 결과를 다음 단계에 사용
        else:
            print("인접 박스 병합 비활성화됨.")
        
        # 중복 제거 (NMS 적용)
        filtered_bboxes = self._non_max_suppression(word_bboxes)
        print(f"NMS 적용 완료. 최종 후보 경계 상자 수: {len(filtered_bboxes)}")
        
        # 경계 상자를 기반으로 패치 추출
        patches = []
        final_bboxes = []
        rejected_count = 0
        
        print(f"\n--- 패치 필터링 시작 (총 {len(filtered_bboxes)}개 후보) ---")
        for i, bbox in enumerate(filtered_bboxes):
            # 패치 추출 (수정된 함수 사용, 원본 이미지 전달)
            patch = self.extract_word_patch(image, bbox)
            
            # 패치 품질 검증 (여전히 필요할 수 있음, 하지만 원본 패치 기준)
            quality_score = self.calculate_patch_quality_score(patch)
            is_valid = self.enhanced_is_valid_patch(patch)
            
            print(f"  후보 {i+1}: 품질 점수={quality_score:.2f}, 유효성={is_valid}", end="")
            
            if quality_score >= self.min_quality_score and is_valid:
                # --- 선명화 처리 제거 --- 
                # enhanced_patch = self.enhance_text_clarity(patch)
                # patches.append(enhanced_patch)
                patches.append(patch) # 처리되지 않은 원본 패치 추가
                # --- 제거 끝 --- 
                final_bboxes.append(bbox)
                print(" -> 통과")
            else:
                rejected_count += 1
                print(" -> 기각")
        
        print(f"--- 패치 필터링 완료 --- 통과: {len(patches)}개, 기각: {rejected_count}개")
        print("--- 패치 추출 종료 ---")
        
        return patches, final_bboxes


def visualize_results(image, bboxes, patches, output_path=None):
    """
    결과 시각화
    
    Args:
        image (numpy.ndarray): 원본 이미지
        bboxes (list): 경계 상자 리스트
        patches (list): 패치 리스트
        output_path (str): 출력 파일 경로 (None이면 화면에 표시)
    """
    # 원본 이미지에 경계 상자 표시
    image_with_boxes = image.copy()
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_with_boxes, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 패치 그리드 생성
    n_patches = len(patches)
    if n_patches > 0:
        cols = min(8, n_patches)
        rows = math.ceil(n_patches / cols)
        
        plt.figure(figsize=(20, 4 * rows))
        
        # 원본 이미지 표시
        plt.subplot(rows + 1, 1, 1)
        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title("Detected Word Regions")
        plt.axis('off')
        
        # 패치 표시
        for i, patch in enumerate(patches):
            plt.subplot(rows + 1, cols, cols + i + 1)
            plt.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            plt.title(f"Patch {i+1}")
            plt.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    else:
        print("추출된 패치가 없습니다.")


def save_patches(patches, output_dir):
    """
    패치 저장
    
    Args:
        patches (list): 패치 리스트
        output_dir (str): 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, patch in enumerate(patches):
        patch_path = os.path.join(output_dir, f"patch_{i:04d}.jpg")
        cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))


def main():
    """메인 함수"""
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(description="Craft 기반 단어 패치화 파이프라인")
    parser.add_argument("--input", "-i", required=True, help="입력 이미지 파일 또는 디렉토리")
    parser.add_argument("--output", "-o", required=True, help="출력 디렉토리")
    parser.add_argument("--patch-size", type=int, nargs=2, default=[224, 224], help="패치 크기 (너비 높이)")
    parser.add_argument("--visualize", action="store_true", help="결과 시각화")
    parser.add_argument("--gpu", action="store_true", help="GPU 사용 (가능한 경우)")
    
    args = parser.parse_args()
    
    # 장치 설정
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"장치: {device}")
    
    # 패치 추출기 초기화
    extractor = OptimizedWordPatchExtractor(
        patch_size=tuple(args.patch_size),
        device=device
    )
    
    # 입력 처리
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if input_path.is_file():
        # 단일 이미지 처리
        print(f"이미지 처리 중: {input_path}")
        
        # 이미지 로드
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"이미지를 로드할 수 없음: {input_path}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 패치 추출
        start_time = time.time()
        patches, bboxes = extractor.extract_patches(image)
        elapsed_time = time.time() - start_time
        
        print(f"처리 완료: {len(patches)}개 패치 추출 (소요 시간: {elapsed_time:.2f}초)")
        
        # 결과 저장
        os.makedirs(output_dir, exist_ok=True)
        
        # 패치 저장
        patch_dir = output_dir / "patches"
        save_patches(patches, str(patch_dir))
        
        # 경계 상자 정보 저장
        boxes_data = []
        for i, bbox in enumerate(bboxes):
            boxes_data.append({
                "id": i,
                "x": int(bbox[0]),
                "y": int(bbox[1]),
                "width": int(bbox[2]),
                "height": int(bbox[3]),
                "patch_file": f"patch_{i:04d}.jpg"
            })
        
        boxes_path = output_dir / "boxes.json"
        with open(boxes_path, 'w', encoding='utf-8') as f:
            json.dump(boxes_data, f, indent=4)
        
        # 시각화
        if args.visualize:
            vis_path = output_dir / "visualization.jpg"
            visualize_results(image, bboxes, patches, str(vis_path))
    
    elif input_path.is_dir():
        # 디렉토리 내 모든 이미지 처리
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = [f for f in input_path.glob('**/*') if f.suffix.lower() in image_extensions]
        
        print(f"총 {len(image_files)}개 이미지 처리 예정")
        
        for img_file in tqdm(image_files):
            # 출력 서브디렉토리 생성
            rel_path = img_file.relative_to(input_path)
            output_subdir = output_dir / rel_path.parent / rel_path.stem
            
            # 이미지 로드
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"이미지를 로드할 수 없음: {img_file}")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 패치 추출
            patches, bboxes = extractor.extract_patches(image)
            
            # 결과 저장
            os.makedirs(output_subdir, exist_ok=True)
            
            # 패치 저장
            patch_dir = output_subdir / "patches"
            save_patches(patches, str(patch_dir))
            
            # 경계 상자 정보 저장
            boxes_data = []
            for i, bbox in enumerate(bboxes):
                boxes_data.append({
                    "id": i,
                    "x": int(bbox[0]),
                    "y": int(bbox[1]),
                    "width": int(bbox[2]),
                    "height": int(bbox[3]),
                    "patch_file": f"patch_{i:04d}.jpg"
                })
            
            boxes_path = output_subdir / "boxes.json"
            with open(boxes_path, 'w', encoding='utf-8') as f:
                json.dump(boxes_data, f, indent=4)
            
            # 시각화
            if args.visualize:
                vis_path = output_subdir / "visualization.jpg"
                visualize_results(image, bboxes, patches, str(vis_path))
    
    else:
        print(f"입력 경로가 존재하지 않음: {args.input}")


if __name__ == "__main__":
    main()
