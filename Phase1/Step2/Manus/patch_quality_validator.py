#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Craft 기반 패치 품질 검증 모듈

이 모듈은 단어 패치의 품질을 검증하고 향상시키는 
고급 알고리즘을 제공합니다.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import math

class PatchQualityValidator:
    """패치 품질 검증 클래스"""
    
    def __init__(self,
                 min_text_ratio=0.05,
                 max_text_ratio=0.7,
                 min_contrast=20,
                 min_quality_score=60,
                 min_center_ratio=0.3,
                 max_contours=10,
                 min_sharpness=50):
        """
        패치 품질 검증기 초기화
        
        Args:
            min_text_ratio (float): 최소 텍스트 비율
            max_text_ratio (float): 최대 텍스트 비율
            min_contrast (float): 최소 대비
            min_quality_score (int): 최소 품질 점수
            min_center_ratio (float): 중앙 영역 최소 텍스트 비율
            max_contours (int): 최대 윤곽선 수
            min_sharpness (float): 최소 선명도
        """
        self.min_text_ratio = min_text_ratio
        self.max_text_ratio = max_text_ratio
        self.min_contrast = min_contrast
        self.min_quality_score = min_quality_score
        self.min_center_ratio = min_center_ratio
        self.max_contours = max_contours
        self.min_sharpness = min_sharpness
    
    def is_valid_patch(self, patch):
        """
        패치 유효성 검증
        
        Args:
            patch (numpy.ndarray): 입력 패치
            
        Returns:
            bool: 패치가 유효한지 여부
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # 대비 계산
        contrast = np.std(gray)
        if contrast < self.min_contrast:
            return False  # 대비가 너무 낮음
        
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
        text_ratio = text_pixels / total_pixels
        
        # 텍스트 비율 검증
        if text_ratio < self.min_text_ratio or text_ratio > self.max_text_ratio:
            return False  # 텍스트 비율이 범위를 벗어남
        
        # 텍스트 위치 검증 (중앙 영역에 텍스트가 있는지)
        h, w = binary.shape
        center_region = binary[h//4:3*h//4, w//4:3*w//4]
        center_text_pixels = np.sum(center_region > 0)
        
        if center_text_pixels / text_pixels < self.min_center_ratio:
            return False  # 텍스트가 주로 가장자리에 있음
        
        # 텍스트 연결성 검증
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > self.max_contours:
            return False  # 텍스트가 너무 분산되어 있음
        
        # 텍스트 선명도 검증
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < self.min_sharpness:
            return False  # 텍스트가 흐릿함
        
        return True  # 모든 검증 통과
    
    def calculate_quality_score(self, patch):
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
    
    def validate_single_word(self, patch):
        """
        패치가 단일 단어를 포함하는지 검증
        
        Args:
            patch (numpy.ndarray): 입력 패치
            
        Returns:
            bool: 단일 단어 포함 여부
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 수평 프로젝션 프로필 계산
        h_proj = np.sum(binary, axis=0)
        
        # 프로필 정규화
        if np.max(h_proj) > 0:
            h_proj = h_proj / np.max(h_proj)
        
        # 큰 간격 감지 (단어 간 간격)
        gaps = []
        in_gap = False
        gap_start = 0
        gap_threshold = 0.1  # 간격 임계값
        
        for i, val in enumerate(h_proj):
            if val <= gap_threshold and not in_gap:
                in_gap = True
                gap_start = i
            elif (val > gap_threshold or i == len(h_proj) - 1) and in_gap:
                in_gap = False
                gap_end = i
                gap_width = gap_end - gap_start
                
                # 패치 너비의 15% 이상인 간격만 고려
                if gap_width >= len(h_proj) * 0.15:
                    gaps.append((gap_start, gap_end))
        
        # 큰 간격이 있으면 여러 단어가 포함된 것으로 간주
        if gaps:
            return False
        
        # 연결 요소 분석
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # 배경 제외
        num_labels -= 1
        
        # 큰 연결 요소 수 계산 (작은 노이즈 제외)
        min_area = binary.size * 0.005  # 패치 크기의 0.5% 이상
        large_components = 0
        
        for i in range(1, num_labels + 1):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                large_components += 1
        
        # 큰 연결 요소가 너무 많으면 여러 단어가 포함된 것으로 간주
        if large_components > 10:
            return False
        
        return True  # 단일 단어로 간주
    
    def enhance_patch_quality(self, patch):
        """
        패치 품질 향상
        
        Args:
            patch (numpy.ndarray): 입력 패치
            
        Returns:
            numpy.ndarray: 품질이 향상된 패치
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
    
    def apply_adaptive_enhancement(self, patch):
        """
        패치 특성에 따른 적응형 품질 향상
        
        Args:
            patch (numpy.ndarray): 입력 패치
            
        Returns:
            numpy.ndarray: 품질이 향상된 패치
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # 이미지 품질 분석
        contrast = np.std(gray)
        brightness = np.mean(gray)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 결과 이미지 초기화
        enhanced = patch.copy()
        
        # 대비가 낮은 경우 대비 향상
        if contrast < 40:
            # CLAHE 적용
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            
            # 컬러 이미지인 경우
            if len(patch.shape) == 3:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                l = clahe.apply(l)
                lab = cv2.merge((l, a, b))
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                enhanced = clahe.apply(enhanced)
        
        # 밝기가 너무 높거나 낮은 경우 조정
        if brightness < 50 or brightness > 200:
            # 히스토그램 평활화
            if len(patch.shape) == 3:
                hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                v = cv2.equalizeHist(v)
                hsv = cv2.merge((h, s, v))
                enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            else:
                enhanced = cv2.equalizeHist(enhanced)
        
        # 선명도가 낮은 경우 선명도 향상
        if laplacian_var < 100:
            # 언샵 마스킹
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # 텍스트 영역 강화
        # 이진화
        if len(enhanced.shape) == 3:
            gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        else:
            gray_enhanced = enhanced
            
        _, binary = cv2.threshold(gray_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 노이즈 제거
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 텍스트 영역 마스크 생성
        text_mask = binary
        bg_mask = cv2.bitwise_not(binary)
        
        # 텍스트와 배경 분리
        if len(enhanced.shape) == 3:
            text = np.zeros_like(enhanced)
            for c in range(3):
                text[:, :, c] = cv2.bitwise_and(enhanced[:, :, c], enhanced[:, :, c], mask=text_mask)
            
            # 배경을 흰색으로 설정
            bg = np.ones_like(enhanced) * 255
            for c in range(3):
                bg[:, :, c] = cv2.bitwise_and(bg[:, :, c], bg[:, :, c], mask=bg_mask)
            
            # 텍스트와 배경 합성
            final = cv2.add(text, bg)
        else:
            text = cv2.bitwise_and(enhanced, enhanced, mask=text_mask)
            bg = cv2.bitwise_and(np.ones_like(enhanced) * 255, np.ones_like(enhanced) * 255, mask=bg_mask)
            final = cv2.add(text, bg)
        
        return final
    
    def center_text_in_patch(self, patch):
        """
        텍스트를 패치 중앙에 위치시키는 함수
        
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
    
    def optimal_resize(self, patch, target_size, original_size, max_scale=1.5):
        """
        확대/축소 비율에 따라 최적의 보간법을 선택하여 리사이징
        
        Args:
            patch (numpy.ndarray): 입력 패치
            target_size (tuple): 목표 크기 (너비, 높이)
            original_size (tuple): 원본 크기 (높이, 너비)
            max_scale (float): 최대 확대 비율
            
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
            scale = min(scale, max_scale)
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
    
    def visualize_patch_quality(self, patches, scores=None, output_path=None):
        """
        패치 품질 시각화
        
        Args:
            patches (list): 패치 리스트
            scores (list): 품질 점수 리스트 (None이면 계산)
            output_path (str): 출력 파일 경로 (None이면 화면에 표시)
        """
        n_patches = len(patches)
        if n_patches == 0:
            print("시각화할 패치가 없습니다.")
            return
        
        # 품질 점수 계산 (제공되지 않은 경우)
        if scores is None:
            scores = [self.calculate_quality_score(patch) for patch in patches]
        
        # 품질 점수에 따라 정렬
        sorted_indices = np.argsort(scores)[::-1]  # 내림차순
        sorted_patches = [patches[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        # 그리드 크기 계산
        cols = min(5, n_patches)
        rows = math.ceil(n_patches / cols)
        
        plt.figure(figsize=(15, 3 * rows))
        plt.suptitle("패치 품질 시각화 (품질 점수 내림차순)", fontsize=16)
        
        for i in range(n_patches):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(sorted_patches[i])
            plt.title(f"Score: {sorted_scores[i]:.1f}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def compare_enhancement(self, original_patches, output_path=None):
        """
        원본 패치와 향상된 패치 비교 시각화
        
        Args:
            original_patches (list): 원본 패치 리스트
            output_path (str): 출력 파일 경로 (None이면 화면에 표시)
        """
        n_patches = len(original_patches)
        if n_patches == 0:
            print("시각화할 패치가 없습니다.")
            return
        
        # 최대 5개 패치만 표시
        n_display = min(5, n_patches)
        
        # 향상된 패치 생성
        enhanced_patches = [self.apply_adaptive_enhancement(patch) for patch in original_patches[:n_display]]
        centered_patches = [self.center_text_in_patch(patch) for patch in original_patches[:n_display]]
        
        # 품질 점수 계산
        original_scores = [self.calculate_quality_score(patch) for patch in original_patches[:n_display]]
        enhanced_scores = [self.calculate_quality_score(patch) for patch in enhanced_patches]
        centered_scores = [self.calculate_quality_score(patch) for patch in centered_patches]
        
        # 시각화
        plt.figure(figsize=(15, 3 * n_display))
        plt.suptitle("패치 품질 향상 비교", fontsize=16)
        
        for i in range(n_display):
            # 원본 패치
            plt.subplot(n_display, 3, i*3 + 1)
            plt.imshow(original_patches[i])
            plt.title(f"원본 (점수: {original_scores[i]:.1f})")
            plt.axis('off')
            
            # 향상된 패치
            plt.subplot(n_display, 3, i*3 + 2)
            plt.imshow(enhanced_patches[i])
            plt.title(f"품질 향상 (점수: {enhanced_scores[i]:.1f})")
            plt.axis('off')
            
            # 중앙 정렬 패치
            plt.subplot(n_display, 3, i*3 + 3)
            plt.imshow(centered_patches[i])
            plt.title(f"중앙 정렬 (점수: {centered_scores[i]:.1f})")
            plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()


def process_patches_with_quality_validation(patches, output_dir=None, visualize=True):
    """
    패치에 품질 검증 및 향상 적용
    
    Args:
        patches (list): 패치 리스트
        output_dir (str): 출력 디렉토리 (None이면 저장하지 않음)
        visualize (bool): 시각화 여부
    """
    # 패치 품질 검증기 초기화
    validator = PatchQualityValidator()
    
    # 품질 점수 계산
    scores = [validator.calculate_quality_score(patch) for patch in patches]
    
    # 단일 단어 검증
    is_single_word = [validator.validate_single_word(patch) for patch in patches]
    
    # 유효한 패치 필터링
    valid_patches = []
    valid_scores = []
    
    for i, patch in enumerate(patches):
        if scores[i] >= validator.min_quality_score and is_single_word[i]:
            valid_patches.append(patch)
            valid_scores.append(scores[i])
    
    print(f"총 패치: {len(patches)}개")
    print(f"유효한 패치: {len(valid_patches)}개 ({len(valid_patches)/len(patches)*100:.1f}%)")
    
    # 품질 향상 적용
    enhanced_patches = [validator.apply_adaptive_enhancement(patch) for patch in valid_patches]
    
    # 결과 저장
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 원본 패치 저장
        original_dir = os.path.join(output_dir, "original")
        os.makedirs(original_dir, exist_ok=True)
        
        for i, patch in enumerate(valid_patches):
            patch_path = os.path.join(original_dir, f"patch_{i:04d}.jpg")
            cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        
        # 향상된 패치 저장
        enhanced_dir = os.path.join(output_dir, "enhanced")
        os.makedirs(enhanced_dir, exist_ok=True)
        
        for i, patch in enumerate(enhanced_patches):
            patch_path = os.path.join(enhanced_dir, f"patch_{i:04d}.jpg")
            cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    
    # 시각화
    if visualize and valid_patches:
        # 품질 점수 시각화
        if output_dir:
            validator.visualize_patch_quality(valid_patches, valid_scores, 
                                             os.path.join(output_dir, "quality_scores.jpg"))
        else:
            validator.visualize_patch_quality(valid_patches, valid_scores)
        
        # 품질 향상 비교 시각화
        if output_dir:
            validator.compare_enhancement(valid_patches[:5], 
                                         os.path.join(output_dir, "enhancement_comparison.jpg"))
        else:
            validator.compare_enhancement(valid_patches[:5])
    
    return valid_patches, enhanced_patches


if __name__ == "__main__":
    import argparse
    import os
    
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(description="Craft 기반 패치 품질 검증 모듈")
    parser.add_argument("--input", "-i", required=True, help="입력 패치 디렉토리")
    parser.add_argument("--output", "-o", help="출력 디렉토리 (지정하지 않으면 저장하지 않음)")
    
    args = parser.parse_args()
    
    # 패치 로드
    patches = []
    for file in os.listdir(args.input):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            patch_path = os.path.join(args.input, file)
            patch = cv2.imread(patch_path)
            if patch is not None:
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patches.append(patch)
    
    # 패치 품질 검증 및 향상 적용
    process_patches_with_quality_validation(patches, args.output)
