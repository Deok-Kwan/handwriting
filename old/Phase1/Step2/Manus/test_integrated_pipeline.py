#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Craft 기반 단어 패치화 파이프라인 통합 테스트 스크립트

이 스크립트는 최적화된 Craft 기반 패치화 파이프라인의 모든 구성 요소를
통합하여 테스트합니다.
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

# 모듈 경로 추가
module_path = os.path.abspath(os.path.dirname(__file__))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

def check_dependencies():
    """필요한 의존성 패키지 확인 및 설치"""
    required_packages = [
        'numpy',
        'opencv-python',
        'torch',
        'matplotlib',
        'tqdm',
        'craft-text-detector',
        'scikit-image'
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

def run_integrated_pipeline(image_path, output_dir, patch_size=(224, 224), visualize=True, use_gpu=False):
    """
    통합 파이프라인 실행
    
    Args:
        image_path (str): 입력 이미지 경로
        output_dir (str): 출력 디렉토리
        patch_size (tuple): 패치 크기 (너비, 높이)
        visualize (bool): 시각화 여부
        use_gpu (bool): GPU 사용 여부
    """
    # 필요한 모듈 임포트
    from craft_word_patch_pipeline import CraftTextDetector, OptimizedWordPatchExtractor
    from word_level_processor import WordLevelProcessor
    from patch_quality_validator import PatchQualityValidator
    import torch
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없음: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 장치 설정
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"장치: {device}")
    
    # 1. 이미지 품질 분석 및 적응형 매개변수 설정
    from optimize_parameters import analyze_image_quality, adaptive_text_detection_params
    
    print("이미지 품질 분석 중...")
    quality = analyze_image_quality(image)
    detection_params = adaptive_text_detection_params(image)
    
    print("이미지 품질 분석 결과:")
    print(f"  대비: {quality['contrast']:.2f}")
    print(f"  선명도: {quality['sharpness']:.2f}")
    print(f"  노이즈 수준: {quality['noise_level']:.2f}")
    print(f"  밝기: {quality['brightness']:.2f}")
    
    print("\n최적화된 텍스트 감지 매개변수:")
    print(f"  text_threshold: {detection_params['text_threshold']}")
    print(f"  link_threshold: {detection_params['link_threshold']}")
    print(f"  low_text: {detection_params['low_text']}")
    print(f"  long_size: {detection_params['long_size']}")
    
    # 2. CRAFT 텍스트 감지기 초기화
    print("\nCRAFT 텍스트 감지기 초기화 중...")
    craft_detector = CraftTextDetector(
        text_threshold=detection_params['text_threshold'],
        link_threshold=detection_params['link_threshold'],
        low_text=detection_params['low_text'],
        long_size=detection_params['long_size'],
        device=device
    )
    
    # 3. 텍스트 영역 감지
    print("텍스트 영역 감지 중...")
    start_time = time.time()
    prediction_result = craft_detector.detect_text_regions(image)
    detection_time = time.time() - start_time
    
    # 다각형을 경계 상자로 변환
    bboxes = craft_detector.convert_polys_to_bboxes(prediction_result["boxes"])
    
    print(f"감지된 텍스트 영역: {len(bboxes)}개 (소요 시간: {detection_time:.2f}초)")
    
    # 4. 단어 수준 처리
    print("\n단어 수준 처리 중...")
    word_processor = WordLevelProcessor()
    
    start_time = time.time()
    word_bboxes = word_processor.segment_to_words(image, bboxes)
    word_processing_time = time.time() - start_time
    
    print(f"단어 수준 세분화 결과: {len(word_bboxes)}개 단어 (소요 시간: {word_processing_time:.2f}초)")
    
    # 라인별 클러스터링
    lines = word_processor.cluster_by_line(word_bboxes)
    print(f"감지된 텍스트 라인: {len(lines)}개")
    
    # 5. 패치 추출
    print("\n패치 추출 중...")
    extractor = OptimizedWordPatchExtractor(
        patch_size=patch_size,
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
    
    start_time = time.time()
    patches, final_bboxes = extractor.extract_patches(image)
    extraction_time = time.time() - start_time
    
    print(f"추출된 패치: {len(patches)}개 (소요 시간: {extraction_time:.2f}초)")
    
    # 6. 패치 품질 검증 및 향상
    print("\n패치 품질 검증 및 향상 중...")
    validator = PatchQualityValidator()
    
    # 품질 점수 계산
    scores = [validator.calculate_quality_score(patch) for patch in patches]
    
    # 단일 단어 검증
    is_single_word = [validator.validate_single_word(patch) for patch in patches]
    
    # 유효한 패치 필터링
    valid_patches = []
    valid_scores = []
    valid_bboxes = []
    
    for i, patch in enumerate(patches):
        if scores[i] >= validator.min_quality_score and is_single_word[i]:
            valid_patches.append(patch)
            valid_scores.append(scores[i])
            valid_bboxes.append(final_bboxes[i])
    
    print(f"유효한 패치: {len(valid_patches)}개 ({len(valid_patches)/max(1, len(patches))*100:.1f}%)")
    
    # 품질 향상 적용
    enhanced_patches = [validator.apply_adaptive_enhancement(patch) for patch in valid_patches]
    
    # 7. 결과 저장
    print("\n결과 저장 중...")
    
    # 원본 패치 저장
    original_dir = os.path.join(output_dir, "original_patches")
    os.makedirs(original_dir, exist_ok=True)
    
    for i, patch in enumerate(valid_patches):
        patch_path = os.path.join(original_dir, f"patch_{i:04d}.jpg")
        cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    
    # 향상된 패치 저장
    enhanced_dir = os.path.join(output_dir, "enhanced_patches")
    os.makedirs(enhanced_dir, exist_ok=True)
    
    for i, patch in enumerate(enhanced_patches):
        patch_path = os.path.join(enhanced_dir, f"patch_{i:04d}.jpg")
        cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    
    # 경계 상자 정보 저장
    boxes_data = []
    for i, bbox in enumerate(valid_bboxes):
        boxes_data.append({
            "id": i,
            "x": int(bbox[0]),
            "y": int(bbox[1]),
            "width": int(bbox[2]),
            "height": int(bbox[3]),
            "score": float(valid_scores[i]),
            "patch_file": f"patch_{i:04d}.jpg"
        })
    
    boxes_path = os.path.join(output_dir, "boxes.json")
    with open(boxes_path, 'w', encoding='utf-8') as f:
        json.dump(boxes_data, f, indent=4)
    
    # 8. 시각화
    if visualize:
        print("\n결과 시각화 중...")
        
        # 원본 이미지에 경계 상자 표시
        vis_image = image.copy()
        for i, bbox in enumerate(valid_bboxes):
            x, y, w, h = bbox
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_image, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 시각화 이미지 저장
        vis_path = os.path.join(output_dir, "visualization.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # 단어 세분화 시각화
        word_vis_path = os.path.join(output_dir, "word_segmentation.jpg")
        word_processor.visualize_word_segmentation(image, word_bboxes, lines, word_vis_path)
        
        # 패치 품질 시각화
        if valid_patches:
            validator.visualize_patch_quality(valid_patches, valid_scores, 
                                             os.path.join(output_dir, "quality_scores.jpg"))
            
            # 품질 향상 비교 시각화
            validator.compare_enhancement(valid_patches[:5], 
                                         os.path.join(output_dir, "enhancement_comparison.jpg"))
    
    # 9. 성능 요약
    performance = {
        "image_path": image_path,
        "image_size": f"{image.shape[1]}x{image.shape[0]}",
        "device": device,
        "detection_time": detection_time,
        "word_processing_time": word_processing_time,
        "extraction_time": extraction_time,
        "total_time": detection_time + word_processing_time + extraction_time,
        "detected_regions": len(bboxes),
        "word_regions": len(word_bboxes),
        "extracted_patches": len(patches),
        "valid_patches": len(valid_patches),
        "valid_ratio": len(valid_patches) / max(1, len(patches))
    }
    
    performance_path = os.path.join(output_dir, "performance.json")
    with open(performance_path, 'w', encoding='utf-8') as f:
        json.dump(performance, f, indent=4)
    
    print("\n처리 완료!")
    print(f"결과가 {output_dir}에 저장되었습니다.")
    
    return valid_patches, enhanced_patches, performance

def main():
    """메인 함수"""
    # Jupyter 노트북 환경 감지
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            args = argparse.Namespace(
                input='sample.jpg',
                output='results',
                patch_size=(224,224),
                no_visualize=False,
                gpu=False
            )
        else:
            parser = argparse.ArgumentParser(description="Craft 기반 단어 패치화 파이프라인 통합 테스트")
            parser.add_argument("--input", "-i", required=True, help="입력 이미지 파일")
            parser.add_argument("--output", "-o", required=True, help="출력 디렉토리")
            parser.add_argument("--patch-size", type=int, nargs=2, default=[224, 224], help="패치 크기 (너비 높이)")
            parser.add_argument("--no-visualize", action="store_true", help="시각화 비활성화")
            parser.add_argument("--gpu", action="store_true", help="GPU 사용 (가능한 경우)")
            args = parser.parse_args()
    except ImportError:
        # IPython 모듈이 없으면 일반 실행
        parser = argparse.ArgumentParser(description="Craft 기반 단어 패치화 파이프라인 통합 테스트")
        parser.add_argument("--input", "-i", required=True, help="입력 이미지 파일")
        parser.add_argument("--output", "-o", required=True, help="출력 디렉토리")
        parser.add_argument("--patch-size", type=int, nargs=2, default=[224, 224], help="패치 크기 (너비 높이)")
        parser.add_argument("--no-visualize", action="store_true", help="시각화 비활성화")
        parser.add_argument("--gpu", action="store_true", help="GPU 사용 (가능한 경우)")
        args = parser.parse_args()

    # 의존성 확인
    check_dependencies()
    
    # 통합 파이프라인 실행
    run_integrated_pipeline(
        args.input, args.output, args.patch_size,
        not args.no_visualize, args.gpu
    )


if __name__ == "__main__":
    main()
