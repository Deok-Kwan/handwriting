#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Craft 기반 단어 패치화 파이프라인 매개변수 최적화 도구

이 스크립트는 CRAFT 텍스트 감지 알고리즘의 매개변수를 최적화하여
단어 단위 패치 추출 성능을 향상시킵니다.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def check_dependencies():
    """필요한 의존성 패키지 확인 및 설치"""
    required_packages = [
        'numpy',
        'opencv-python',
        'torch',
        'matplotlib',
        'tqdm',
        'craft-text-detector'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("다음 패키지를 설치해야 합니다:")
        for package in missing_packages:
            print(f"  - {package}")
        
        install = input("자동으로 설치하시겠습니까? (y/n): ")
        if install.lower() == 'y':
            import subprocess
            for package in missing_packages:
                print(f"{package} 설치 중...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("모든 패키지가 설치되었습니다.")
        else:
            print("다음 명령으로 필요한 패키지를 설치하세요:")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)

def analyze_image_quality(image):
    """
    이미지 품질 분석
    
    Args:
        image (numpy.ndarray): 입력 이미지
        
    Returns:
        dict: 이미지 품질 정보
    """
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 대비 계산
    contrast = np.std(gray)
    
    # 선명도 계산
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 노이즈 수준 추정
    h, w = gray.shape
    noise_levels = []
    
    for i in range(3):
        for j in range(3):
            block = gray[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
            noise_levels.append(np.std(block))
    
    noise_levels.sort()
    noise_level = np.mean(noise_levels[:3])
    
    # 밝기 계산
    brightness = np.mean(gray)
    
    return {
        'contrast': contrast,
        'sharpness': laplacian_var,
        'noise_level': noise_level,
        'brightness': brightness
    }

def adaptive_text_detection_params(image):
    """
    이미지 특성에 따라 텍스트 감지 매개변수를 적응적으로 조정
    
    Args:
        image (numpy.ndarray): 입력 이미지
        
    Returns:
        dict: 최적화된 텍스트 감지 매개변수
    """
    # 이미지 품질 분석
    quality = analyze_image_quality(image)
    
    # 이미지 해상도 분석
    h, w = image.shape[:2]
    resolution = h * w
    
    # 기본 매개변수
    params = {
        'text_threshold': 0.55,
        'link_threshold': 0.85,
        'low_text': 0.35,
        'long_size': 1536
    }
    
    # 해상도에 따른 long_size 조정
    if resolution > 4000000:  # 4MP 이상
        params['long_size'] = 1536
    elif resolution > 1000000:  # 1MP 이상
        params['long_size'] = 1280
    else:
        params['long_size'] = 1024
    
    # 대비에 따른 text_threshold 조정
    if quality['contrast'] < 40:  # 낮은 대비
        params['text_threshold'] = 0.45  # 낮은 임계값
    else:
        params['text_threshold'] = 0.55  # 높은 임계값
    
    # 선명도에 따른 link_threshold 조정
    if quality['sharpness'] < 100:  # 흐린 이미지
        params['link_threshold'] = 0.75  # 낮은 임계값
    else:
        params['link_threshold'] = 0.85  # 높은 임계값
    
    # 노이즈 수준에 따른 low_text 조정
    if quality['noise_level'] > 10:  # 노이즈가 많은 이미지
        params['low_text'] = 0.4  # 높은 임계값
    else:
        params['low_text'] = 0.35  # 기본 임계값
    
    return params

def optimize_split_overlap(bbox_width, bbox_height):
    """
    경계 상자 크기에 따라 최적의 오버랩 비율 계산
    
    Args:
        bbox_width (int): 경계 상자 너비
        bbox_height (int): 경계 상자 높이
        
    Returns:
        float: 최적의 오버랩 비율
    """
    # 기본 오버랩 비율
    base_overlap_ratio = 0.2  # 현재 0.1에서 증가
    
    # 너비에 따른 오버랩 조정 (넓을수록 오버랩 증가)
    if bbox_width > 150:
        width_factor = 1.2
    elif bbox_width > 100:
        width_factor = 1.1
    else:
        width_factor = 1.0
    
    # 높이에 따른 오버랩 조정 (높을수록 오버랩 증가)
    if bbox_height > 60:
        height_factor = 1.2
    elif bbox_height > 40:
        height_factor = 1.1
    else:
        height_factor = 1.0
    
    # 최종 오버랩 비율 계산
    overlap_ratio = base_overlap_ratio * width_factor * height_factor
    
    # 최대 0.35로 제한
    return min(0.35, overlap_ratio)

def optimize_patch_size(word_bbox):
    """
    단어 크기에 따라 최적의 패치 크기 결정
    
    Args:
        word_bbox (list): 단어 경계 상자 [x, y, width, height]
        
    Returns:
        tuple: 최적의 패치 크기 (너비, 높이)
    """
    _, _, w, h = word_bbox
    
    # 기본 패치 크기
    base_width, base_height = 224, 224
    
    # 단어 비율 계산
    aspect_ratio = w / max(1, h)
    
    # 비율에 따른 패치 크기 조정
    if aspect_ratio > 3.0:  # 매우 넓은 단어
        return (320, 160)  # 넓은 패치
    elif aspect_ratio > 2.0:  # 넓은 단어
        return (288, 192)  # 약간 넓은 패치
    elif aspect_ratio > 1.5:  # 약간 넓은 단어
        return (256, 224)  # 표준보다 약간 넓은 패치
    elif aspect_ratio < 0.5:  # 매우 세로로 긴 단어
        return (160, 320)  # 세로로 긴 패치
    elif aspect_ratio < 0.75:  # 세로로 긴 단어
        return (192, 288)  # 약간 세로로 긴 패치
    else:  # 표준 비율 단어
        return (base_width, base_height)  # 표준 패치

def run_parameter_optimization(image_path, output_dir):
    """
    이미지에 대한 매개변수 최적화 실행
    
    Args:
        image_path (str): 입력 이미지 경로
        output_dir (str): 출력 디렉토리
    """
    # 필요한 모듈 임포트
    from craft_word_patch_pipeline import CraftTextDetector, OptimizedWordPatchExtractor
    import torch
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없음: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지 품질 분석
    quality = analyze_image_quality(image)
    print("이미지 품질 분석 결과:")
    print(f"  대비: {quality['contrast']:.2f}")
    print(f"  선명도: {quality['sharpness']:.2f}")
    print(f"  노이즈 수준: {quality['noise_level']:.2f}")
    print(f"  밝기: {quality['brightness']:.2f}")
    
    # 적응형 텍스트 감지 매개변수 계산
    detection_params = adaptive_text_detection_params(image)
    print("\n최적화된 텍스트 감지 매개변수:")
    print(f"  text_threshold: {detection_params['text_threshold']}")
    print(f"  link_threshold: {detection_params['link_threshold']}")
    print(f"  low_text: {detection_params['low_text']}")
    print(f"  long_size: {detection_params['long_size']}")
    
    # 장치 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n장치: {device}")
    
    # CRAFT 텍스트 감지기 초기화
    craft_detector = CraftTextDetector(
        text_threshold=detection_params['text_threshold'],
        link_threshold=detection_params['link_threshold'],
        low_text=detection_params['low_text'],
        long_size=detection_params['long_size'],
        device=device
    )
    
    # 텍스트 영역 감지
    prediction_result = craft_detector.detect_text_regions(image)
    
    # 다각형을 경계 상자로 변환
    bboxes = craft_detector.convert_polys_to_bboxes(prediction_result["boxes"])
    
    print(f"\n감지된 텍스트 영역: {len(bboxes)}개")
    
    # 최적화된 패치 크기 계산
    if bboxes:
        print("\n단어 크기에 따른 최적 패치 크기 예시:")
        for i, bbox in enumerate(bboxes[:5]):  # 처음 5개만 표시
            optimal_size = optimize_patch_size(bbox)
            print(f"  단어 {i+1} ({bbox[2]}x{bbox[3]}): {optimal_size[0]}x{optimal_size[1]}")
    
    # 패치 추출기 초기화 (기본 매개변수)
    default_extractor = OptimizedWordPatchExtractor(
        device=device,
        craft_detector=craft_detector
    )
    
    # 최적화된 패치 추출기 초기화
    optimized_extractor = OptimizedWordPatchExtractor(
        padding_ratio=0.15,
        min_text_area=45,
        overlap_threshold=0.22,
        max_scale=1.5,
        enable_merge=True,
        max_gap=18,
        max_height_diff=18,
        min_quality_score=60,
        device=device,
        craft_detector=craft_detector
    )
    
    # 기본 매개변수로 패치 추출
    print("\n기본 매개변수로 패치 추출 중...")
    start_time = time.time()
    default_patches, default_bboxes = default_extractor.extract_patches(image)
    default_time = time.time() - start_time
    
    # 최적화된 매개변수로 패치 추출
    print("최적화된 매개변수로 패치 추출 중...")
    start_time = time.time()
    optimized_patches, optimized_bboxes = optimized_extractor.extract_patches(image)
    optimized_time = time.time() - start_time
    
    print("\n결과 비교:")
    print(f"  기본 매개변수: {len(default_patches)}개 패치 추출 (소요 시간: {default_time:.2f}초)")
    print(f"  최적화된 매개변수: {len(optimized_patches)}개 패치 추출 (소요 시간: {optimized_time:.2f}초)")
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    
    # 매개변수 정보 저장
    params_info = {
        "image_quality": quality,
        "detection_params": detection_params,
        "default_result": {
            "patch_count": len(default_patches),
            "processing_time": default_time
        },
        "optimized_result": {
            "patch_count": len(optimized_patches),
            "processing_time": optimized_time
        }
    }
    
    params_path = os.path.join(output_dir, "optimization_params.json")
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(params_info, f, indent=4)
    
    # 기본 패치 저장
    default_dir = os.path.join(output_dir, "default_patches")
    os.makedirs(default_dir, exist_ok=True)
    
    for i, patch in enumerate(default_patches):
        patch_path = os.path.join(default_dir, f"patch_{i:04d}.jpg")
        cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    
    # 최적화된 패치 저장
    optimized_dir = os.path.join(output_dir, "optimized_patches")
    os.makedirs(optimized_dir, exist_ok=True)
    
    for i, patch in enumerate(optimized_patches):
        patch_path = os.path.join(optimized_dir, f"patch_{i:04d}.jpg")
        cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    
    # 시각화
    # 원본 이미지에 경계 상자 표시 (기본)
    default_image = image.copy()
    for i, bbox in enumerate(default_bboxes):
        x, y, w, h = bbox
        cv2.rectangle(default_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 원본 이미지에 경계 상자 표시 (최적화)
    optimized_image = image.copy()
    for i, bbox in enumerate(optimized_bboxes):
        x, y, w, h = bbox
        cv2.rectangle(optimized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 시각화 이미지 저장
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    plt.imshow(default_image)
    plt.title(f"기본 매개변수 ({len(default_bboxes)}개 영역)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(optimized_image)
    plt.title(f"최적화된 매개변수 ({len(optimized_bboxes)}개 영역)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.jpg"))
    
    # 패치 그리드 시각화
    if default_patches and optimized_patches:
        # 기본 패치 그리드
        n_default = min(len(default_patches), 20)  # 최대 20개만 표시
        cols = min(5, n_default)
        rows = (n_default + cols - 1) // cols
        
        plt.figure(figsize=(15, 3 * rows))
        plt.suptitle("기본 매개변수로 추출한 패치", fontsize=16)
        
        for i in range(n_default):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(default_patches[i])
            plt.title(f"Patch {i+1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(output_dir, "default_patches_grid.jpg"))
        
        # 최적화된 패치 그리드
        n_optimized = min(len(optimized_patches), 20)  # 최대 20개만 표시
        cols = min(5, n_optimized)
        rows = (n_optimized + cols - 1) // cols
        
        plt.figure(figsize=(15, 3 * rows))
        plt.suptitle("최적화된 매개변수로 추출한 패치", fontsize=16)
        
        for i in range(n_optimized):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(optimized_patches[i])
            plt.title(f"Patch {i+1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(output_dir, "optimized_patches_grid.jpg"))
    
    print(f"\n최적화 결과가 {output_dir}에 저장되었습니다.")

def main():
    """메인 함수"""
    # 의존성 확인
    check_dependencies()
    
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(description="Craft 기반 단어 패치화 파이프라인 매개변수 최적화 도구")
    parser.add_argument("--input", "-i", required=True, help="입력 이미지 파일")
    parser.add_argument("--output", "-o", required=True, help="출력 디렉토리")
    
    args = parser.parse_args()
    
    # 매개변수 최적화 실행
    run_parameter_optimization(args.input, args.output)


if __name__ == "__main__":
    main()
