# 1번

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# CRAFT 텍스트 감지 라이브러리 설치가 필요합니다
# 터미널에서 다음 명령어를 실행하세요: pip install craft-text-detector

from craft_text_detector import Craft

class CraftTextDetector:
    """CRAFT 텍스트 감지기 클래스"""
    
    def __init__(self, weights_path=None, refiner_weights_path=None, device='cpu', text_threshold=0.7, link_threshold=0.4, low_text=0.4):
        """
        CRAFT 텍스트 감지기 초기화
        
        Args:
            weights_path: CRAFT 모델 가중치 경로 (None이면 기본 가중치 사용)
            refiner_weights_path: 정제 모델 가중치 경로 (None이면 사용 안 함)
            device: 'cuda' 또는 'cpu'
            text_threshold: 텍스트 감지 신뢰도 임계값
            link_threshold: 텍스트 링크 임계값
            low_text: 낮은 신뢰도 텍스트 임계값
        """
        from craft_text_detector import Craft
        
        # 단어 단위 감지를 위한 매개변수 최적화
        # - text_threshold를 더 낮춰 작은 글자도 감지 (0.55 → 0.45)
        # - link_threshold를 크게 높여 문자 간 연결을 확실히 끊음 (0.6 → 0.75)
        # - long_size 증가로 세밀한 감지 (1536 → 1920)
        self.craft = Craft(
            output_dir=None,
            crop_type="poly",
            cuda=device == 'cuda',
            long_size=1920,       # 더 세밀한 감지를 위해 추가 증가
            text_threshold=0.45,  # 더 작은 글자도 감지
            link_threshold=0.75,  # 단어 간 연결을 확실히 끊기
            low_text=0.3          # 낮은 신뢰도 텍스트도 포함
        )
        
    def detect_text_regions(self, image, text_threshold=None, link_threshold=None, low_text=None):
        """
        이미지에서 텍스트 영역 감지
        
        Args:
            image: 이미지 파일 경로 또는 NumPy 배열
            text_threshold: 사용되지 않음 (하위 호환성을 위해 유지)
            link_threshold: 사용되지 않음 (하위 호환성을 위해 유지)
            low_text: 사용되지 않음 (하위 호환성을 위해 유지)
            
        Returns:
            dict: 감지 결과
        """
        # 이미지가 NumPy 배열인 경우 임시 파일로 저장
        if isinstance(image, np.ndarray):
            import tempfile
            import cv2
            
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, 'image.jpg')
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # 이미지 감지 (매개변수 제거)
            prediction_result = self.craft.detect_text(temp_path)
            
            # 임시 파일 삭제
            os.remove(temp_path)
            os.rmdir(temp_dir)
        else:
            # 이미지 경로가 주어진 경우
            prediction_result = self.craft.detect_text(image)
        
        return prediction_result
    
    def get_text_boxes(self, image):
        """
        이미지에서 텍스트 영역의 경계 상자만 반환
        
        Args:
            image: NumPy 배열, PIL 이미지 또는 이미지 파일 경로
            
        Returns:
            list: 경계 상자 목록 (각 상자는 (x1, y1, x2, y2, x3, y3, x4, y4) 형식)
        """
        prediction_result = self.detect_text_regions(image)
        return prediction_result["boxes"]
    
    def convert_polys_to_bboxes(self, polys):
        """
        다각형 좌표를 직사각형 경계 상자로 변환
        
        Args:
            polys: 다각형 좌표 리스트 (각 다각형은 (x1, y1, x2, y2, x3, y3, x4, y4) 형식)
            
        Returns:
            list: 경계 상자 목록 (각 상자는 [x, y, w, h] 형식)
        """
        bboxes = []
        for poly in polys:
            # 다각형 좌표 변환 (x1, y1, x2, y2, x3, y3, x4, y4) -> [[x1, y1], [x2, y2], ...]
            poly_points = np.array(poly).reshape(-1, 2)
            
            # 최소/최대 x, y 값 찾기
            x_min = np.min(poly_points[:, 0])
            y_min = np.min(poly_points[:, 1])
            x_max = np.max(poly_points[:, 0])
            y_max = np.max(poly_points[:, 1])
            
            # [x, y, width, height] 형식의 경계 상자
            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            bboxes.append(bbox)
        
        return bboxes
    
    def visualize_detection(self, image, prediction_result=None, output_path=None):
        """
        텍스트 감지 결과 시각화
        
        Args:
            image: 원본 이미지 (NumPy 배열)
            prediction_result: detect_text_regions()의 결과 (없으면 새로 감지)
            output_path: 결과 이미지 저장 경로 (없으면 표시만 함)
            
        Returns:
            NumPy 배열: 시각화된 이미지
        """
        # 이미지 로드 방식 처리
        if isinstance(image, str) or isinstance(image, Path):
            image_path = str(image)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # prediction_result가 없으면 새로 감지
        if prediction_result is None:
            prediction_result = self.detect_text_regions(image)
            
        # 텍스트 영역 시각화 (직접 구현)
        visualization = image.copy()
        
        # prediction_result에서 boxes 가져오기
        if "boxes" in prediction_result:
            boxes = prediction_result["boxes"]
            bboxes = self.convert_polys_to_bboxes(boxes)
            
            # 히트맵 생성 (단순한 흑백 이미지)
            h, w = image.shape[:2]
            heatmap = np.zeros((h, w), dtype=np.uint8)
            
            # 박스 그리기
            for bbox in bboxes:
                x, y, width, height = bbox
                cv2.rectangle(visualization, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # 히트맵에 영역 추가
                cv2.rectangle(heatmap, (x, y), (x + width, y + height), 255, -1)
            
            # 컬러 히트맵으로 변환
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # 알파 블렌딩
            alpha = 0.5
            visualization = cv2.addWeighted(visualization, 1-alpha, heatmap_colored, alpha, 0)
        
        # 결과 저장
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            
        return visualization
