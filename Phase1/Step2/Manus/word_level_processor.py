#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Craft 기반 단어 수준 처리 모듈

이 모듈은 CRAFT 텍스트 감지 결과를 단어 수준으로 처리하는 
고급 알고리즘을 제공합니다.
"""

import cv2
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt

class WordLevelProcessor:
    """단어 수준 처리 클래스"""
    
    def __init__(self,
                 max_width=180,       # 100에서 증가
                 max_height=70,       # 40에서 증가
                 force_split_ratio=4.0,  # 3.0에서 증가
                 min_gap_width=5,
                 enable_merge=True,
                 max_gap=18,          # 10에서 증가
                 max_height_diff=18,  # 10에서 증가
                 max_angle_diff=15):
        """
        단어 수준 처리기 초기화
        
        Args:
            max_width (int): 분할 없이 허용되는 최대 너비
            max_height (int): 분할 없이 허용되는 최대 높이
            force_split_ratio (float): 강제 분할을 위한 비율 임계값
            min_gap_width (int): 단어 간 최소 간격 너비
            enable_merge (bool): 병합 기능 활성화 여부
            max_gap (int): 병합 시 최대 간격
            max_height_diff (int): 병합 시 최대 높이 차이
            max_angle_diff (int): 병합 시 최대 각도 차이
        """
        self.max_width = max_width
        self.max_height = max_height
        self.force_split_ratio = force_split_ratio
        self.min_gap_width = min_gap_width
        self.enable_merge = enable_merge
        self.max_gap = max_gap
        self.max_height_diff = max_height_diff
        self.max_angle_diff = max_angle_diff
    
    def segment_to_words(self, image, bboxes, overlap_ratio=0.2):
        """
        텍스트 영역을 단어 수준으로 세분화
        
        Args:
            image (numpy.ndarray): 입력 이미지
            bboxes (list): 경계 상자 리스트 [x, y, width, height]
            overlap_ratio (float): 분할 시 오버랩 비율
            
        Returns:
            list: 단어 수준 경계 상자 리스트
        """
        word_bboxes = []
        
        for bbox in bboxes:
            x, y, w, h = bbox
            
            # 크기 기반 분할 결정
            if w > self.max_width or h > self.max_height or w/h > self.force_split_ratio or h/w > self.force_split_ratio:
                # 큰 텍스트 영역 또는 비정상적인 비율 -> 분할 필요
                
                # 텍스트 영역 추출
                text_region = image[y:y+h, x:x+w]
                
                # 그레이스케일 변환
                if len(text_region.shape) == 3:
                    gray = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
                else:
                    gray = text_region
                
                # 이진화
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # 수평 프로젝션 프로필 계산 (단어 간 간격 감지)
                h_proj = np.sum(binary, axis=0)
                
                # 간격 감지 (값이 낮은 열이 연속으로 나타나는 구간)
                gaps = self._detect_gaps(h_proj)
                
                # 간격을 기준으로 단어 분할
                if gaps:
                    word_bboxes.extend(self._split_by_gaps(x, y, w, h, gaps, overlap_ratio))
                else:
                    # 간격이 감지되지 않으면 크기 기반 분할
                    word_bboxes.extend(self._split_by_size(x, y, w, h, overlap_ratio))
            else:
                # 작은 텍스트 영역 -> 이미 단어 수준으로 간주
                word_bboxes.append(bbox)
        
        # 인접한 텍스트 상자 병합 (활성화된 경우에만)
        if self.enable_merge:
            word_bboxes = self.merge_adjacent_bboxes(word_bboxes)
        
        return word_bboxes
    
    def _detect_gaps(self, projection, threshold_factor=0.2):
        """
        프로젝션 프로필에서 간격 감지
        
        Args:
            projection (numpy.ndarray): 프로젝션 프로필
            threshold_factor (float): 임계값 계수
            
        Returns:
            list: 감지된 간격 리스트 [(시작, 끝), ...]
        """
        # 프로젝션 값의 평균 계산
        mean_value = np.mean(projection)
        
        # 임계값 계산 (평균의 일정 비율)
        threshold = mean_value * threshold_factor
        
        # 간격 감지
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, val in enumerate(projection):
            if val <= threshold and not in_gap:
                in_gap = True
                gap_start = i
            elif (val > threshold or i == len(projection) - 1) and in_gap:
                in_gap = False
                gap_end = i
                if gap_end - gap_start >= self.min_gap_width:  # 최소 간격 크기
                    gaps.append((gap_start, gap_end))
        
        return gaps
    
    def _split_by_gaps(self, x, y, w, h, gaps, overlap_ratio):
        """
        감지된 간격을 기준으로 텍스트 영역 분할
        
        Args:
            x, y, w, h (int): 원본 경계 상자 좌표
            gaps (list): 감지된 간격 리스트 [(시작, 끝), ...]
            overlap_ratio (float): 분할 시 오버랩 비율
            
        Returns:
            list: 분할된 경계 상자 리스트
        """
        split_bboxes = []
        prev_end = 0
        
        for gap_start, gap_end in gaps:
            if gap_start > prev_end:
                # 간격 이전 영역을 단어로 추출
                word_x = x + prev_end
                word_w = gap_start - prev_end
                
                # 오버랩 추가
                overlap = int(word_w * overlap_ratio)
                word_w += overlap
                
                split_bboxes.append([word_x, y, word_w, h])
            
            prev_end = gap_end
        
        # 마지막 간격 이후 영역
        if prev_end < w:
            word_x = x + prev_end
            word_w = w - prev_end
            split_bboxes.append([word_x, y, word_w, h])
        
        return split_bboxes
    
    def _split_by_size(self, x, y, w, h, overlap_ratio):
        """
        크기 기반 텍스트 영역 분할
        
        Args:
            x, y, w, h (int): 원본 경계 상자 좌표
            overlap_ratio (float): 분할 시 오버랩 비율
            
        Returns:
            list: 분할된 경계 상자 리스트
        """
        # 분할 수 계산
        if w > h:
            # 가로로 긴 텍스트 영역 -> 가로 분할
            n_splits = math.ceil(w / self.max_width)
            split_width = w / n_splits
            overlap = int(split_width * overlap_ratio)
            
            split_bboxes = []
            for i in range(n_splits):
                split_x = x + int(i * split_width)
                # 마지막 분할이 아니면 오버랩 추가
                if i < n_splits - 1:
                    split_w = int(split_width) + overlap
                else:
                    split_w = w - int(i * split_width)
                
                split_bboxes.append([split_x, y, split_w, h])
        else:
            # 세로로 긴 텍스트 영역 -> 세로 분할
            n_splits = math.ceil(h / self.max_height)
            split_height = h / n_splits
            overlap = int(split_height * overlap_ratio)
            
            split_bboxes = []
            for i in range(n_splits):
                split_y = y + int(i * split_height)
                # 마지막 분할이 아니면 오버랩 추가
                if i < n_splits - 1:
                    split_h = int(split_height) + overlap
                else:
                    split_h = h - int(i * split_height)
                
                split_bboxes.append([x, split_y, w, split_h])
        
        return split_bboxes
    
    def merge_adjacent_bboxes(self, bboxes):
        """
        인접한 텍스트 상자 병합
        
        Args:
            bboxes (list): 경계 상자 리스트 [x, y, width, height]
            
        Returns:
            list: 병합된 경계 상자 리스트
        """
        if len(bboxes) <= 1:
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
            is_angle_similar = abs(curr_angle - next_angle) <= self.max_angle_diff
            
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
    
    def cluster_by_line(self, bboxes, max_y_diff_ratio=0.5):
        """
        텍스트 상자를 라인별로 클러스터링
        
        Args:
            bboxes (list): 경계 상자 리스트 [x, y, width, height]
            max_y_diff_ratio (float): 최대 y 차이 비율
            
        Returns:
            list: 라인별로 클러스터링된 경계 상자 리스트
        """
        if not bboxes:
            return []
        
        # 평균 높이 계산
        avg_height = np.mean([h for _, _, _, h in bboxes])
        max_y_diff = avg_height * max_y_diff_ratio
        
        # y 좌표 기준으로 정렬
        sorted_by_y = sorted(bboxes, key=lambda box: box[1])
        
        # 라인 클러스터링
        lines = []
        current_line = [sorted_by_y[0]]
        current_y_center = sorted_by_y[0][1] + sorted_by_y[0][3] / 2
        
        for bbox in sorted_by_y[1:]:
            y_center = bbox[1] + bbox[3] / 2
            
            if abs(y_center - current_y_center) <= max_y_diff:
                # 같은 라인에 추가
                current_line.append(bbox)
                # 라인 중심 y 좌표 업데이트
                current_y_center = np.mean([b[1] + b[3]/2 for b in current_line])
            else:
                # 새 라인 시작
                lines.append(current_line)
                current_line = [bbox]
                current_y_center = y_center
        
        # 마지막 라인 추가
        if current_line:
            lines.append(current_line)
        
        # 각 라인 내에서 x 좌표로 정렬
        for i in range(len(lines)):
            lines[i] = sorted(lines[i], key=lambda box: box[0])
        
        return lines
    
    def validate_word_completeness(self, image, bbox):
        """
        단어 완전성 검증
        
        Args:
            image (numpy.ndarray): 입력 이미지
            bbox (list): 경계 상자 [x, y, width, height]
            
        Returns:
            float: 완전성 점수 (0-1)
        """
        x, y, w, h = bbox
        
        # 경계 조건 확인
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w-1))
        y = max(0, min(y, img_h-1))
        w = max(1, min(w, img_w-x))
        h = max(1, min(h, img_h-y))
        
        # 텍스트 영역 추출
        text_region = image[y:y+h, x:x+w]
        
        # 그레이스케일 변환
        if len(text_region.shape) == 3:
            gray = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = text_region
        
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 수평 프로젝션 프로필 계산
        h_proj = np.sum(binary, axis=0)
        
        # 왼쪽과 오른쪽 가장자리 텍스트 밀도 계산
        edge_width = max(3, int(w * 0.1))  # 가장자리 너비 (최소 3픽셀)
        
        left_density = np.sum(h_proj[:edge_width]) / (edge_width * h)
        right_density = np.sum(h_proj[-edge_width:]) / (edge_width * h)
        
        # 전체 텍스트 밀도 계산
        total_density = np.sum(h_proj) / (w * h)
        
        # 가장자리 밀도가 전체 밀도보다 현저히 높으면 잘린 단어일 가능성이 높음
        edge_ratio_threshold = 1.5
        
        left_edge_ratio = left_density / max(0.001, total_density)
        right_edge_ratio = right_density / max(0.001, total_density)
        
        # 완전성 점수 계산 (1에 가까울수록 완전한 단어)
        if left_edge_ratio > edge_ratio_threshold or right_edge_ratio > edge_ratio_threshold:
            # 가장자리 밀도가 높으면 잘린 단어일 가능성이 높음
            completeness = 1.0 - min(1.0, max(left_edge_ratio, right_edge_ratio) / (edge_ratio_threshold * 2))
        else:
            # 가장자리 밀도가 낮으면 완전한 단어일 가능성이 높음
            completeness = 1.0
        
        return completeness
    
    def visualize_word_segmentation(self, image, bboxes, lines=None, output_path=None):
        """
        단어 세분화 결과 시각화
        
        Args:
            image (numpy.ndarray): 입력 이미지
            bboxes (list): 경계 상자 리스트 [x, y, width, height]
            lines (list): 라인별로 클러스터링된 경계 상자 리스트
            output_path (str): 출력 파일 경로 (None이면 화면에 표시)
        """
        # 원본 이미지 복사
        vis_image = image.copy()
        
        if lines:
            # 라인별로 다른 색상 사용
            colors = [
                (0, 255, 0),    # 녹색
                (255, 0, 0),    # 파란색
                (0, 0, 255),    # 빨간색
                (255, 255, 0),  # 청록색
                (255, 0, 255),  # 자홍색
                (0, 255, 255),  # 노란색
            ]
            
            for i, line in enumerate(lines):
                color = colors[i % len(colors)]
                
                for bbox in line:
                    x, y, w, h = bbox
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        else:
            # 모든 경계 상자를 녹색으로 표시
            for bbox in bboxes:
                x, y, w, h = bbox
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 결과 표시 또는 저장
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        else:
            plt.figure(figsize=(12, 10))
            plt.imshow(vis_image)
            plt.axis('off')
            plt.title(f"단어 세분화 결과 ({len(bboxes)}개 단어)")
            plt.tight_layout()
            plt.show()


def process_image_with_word_segmentation(image_path, output_path=None, visualize=True):
    """
    이미지에 단어 수준 세분화 적용
    
    Args:
        image_path (str): 입력 이미지 경로
        output_path (str): 출력 이미지 경로 (None이면 화면에 표시)
        visualize (bool): 시각화 여부
    """
    # 필요한 모듈 임포트
    from craft_word_patch_pipeline import CraftTextDetector
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없음: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # CRAFT 텍스트 감지기 초기화
    craft_detector = CraftTextDetector()
    
    # 텍스트 영역 감지
    prediction_result = craft_detector.detect_text_regions(image)
    
    # 다각형을 경계 상자로 변환
    bboxes = craft_detector.convert_polys_to_bboxes(prediction_result["boxes"])
    
    print(f"감지된 텍스트 영역: {len(bboxes)}개")
    
    # 단어 수준 처리기 초기화
    word_processor = WordLevelProcessor()
    
    # 단어 수준 세분화
    word_bboxes = word_processor.segment_to_words(image, bboxes)
    
    print(f"단어 수준 세분화 결과: {len(word_bboxes)}개 단어")
    
    # 라인별 클러스터링
    lines = word_processor.cluster_by_line(word_bboxes)
    
    print(f"감지된 텍스트 라인: {len(lines)}개")
    
    # 시각화
    if visualize:
        word_processor.visualize_word_segmentation(image, word_bboxes, lines, output_path)
    
    return word_bboxes, lines


if __name__ == "__main__":
    import argparse
    
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(description="Craft 기반 단어 수준 처리 모듈")
    parser.add_argument("--input", "-i", required=True, help="입력 이미지 파일")
    parser.add_argument("--output", "-o", help="출력 이미지 파일 (지정하지 않으면 화면에 표시)")
    
    args = parser.parse_args()
    
    # 단어 수준 세분화 실행
    process_image_with_word_segmentation(args.input, args.output)
