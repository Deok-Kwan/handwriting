#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ㅛ
Craft 기반 단어 패치화 파이프라인 실행 스크립트

이 스크립트는 CRAFT 텍스트 감지 알고리즘을 사용하여 문서 이미지에서
단어 단위 패치를 추출하는 최적화된 파이프라인을 실행합니다.
"""

import os
import sys
import argparse
from pathlib import Path

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

def main():
    """메인 함수"""
    # Jupyter 노트북 환경 감지
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            # Jupyter Notebook 환경에서 실행 시 기본 인수 설정
            print("Jupyter Notebook 환경에서 실행 중입니다.")
            print("기본 테스트 이미지와 출력 경로를 사용합니다.")
            
            # 테스트 이미지 경로 설정
            default_image = "/home/dk926/workspace/MIL/mil_dataset/single_author/writer_w0001/images/synth_w0001_single_0_20250331000323789855_img.png"
            default_output = "/home/dk926/workspace/MIL/results/craft_output"
            
            class Args:
                def __init__(self):
                    self.input = default_image
                    self.output = default_output
                    self.patch_size = [224, 224]
                    self.padding_ratio = 0.15
                    self.min_quality = 35
                    self.visualize = True
                    self.no_merge = False
                    self.gpu = False
            
            args = Args()
            print(f"입력 이미지: {args.input}")
            print(f"출력 디렉토리: {args.output}")
        else:
            # 일반 명령줄 환경에서 실행
            parser = argparse.ArgumentParser(description="Craft 기반 단어 패치화 파이프라인")
            parser.add_argument("--input", "-i", required=True, help="입력 이미지 파일 또는 디렉토리")
            parser.add_argument("--output", "-o", required=True, help="출력 디렉토리")
            parser.add_argument("--patch-size", type=int, nargs=2, default=[224, 224], help="패치 크기 (너비 높이)")
            parser.add_argument("--padding-ratio", type=float, default=0.15, help="패딩 비율 (기본값: 0.15)")
            parser.add_argument("--min-quality", type=int, default=60, help="최소 품질 점수 (0-100, 기본값: 60)")
            parser.add_argument("--visualize", action="store_true", help="결과 시각화")
            parser.add_argument("--no-merge", action="store_true", help="인접 텍스트 상자 병합 비활성화")
            parser.add_argument("--gpu", action="store_true", help="GPU 사용 (가능한 경우)")
            args = parser.parse_args()
    except ImportError:
        # IPython 모듈이 없는 경우 일반 명령줄 실행
        parser = argparse.ArgumentParser(description="Craft 기반 단어 패치화 파이프라인")
        parser.add_argument("--input", "-i", required=True, help="입력 이미지 파일 또는 디렉토리")
        parser.add_argument("--output", "-o", required=True, help="출력 디렉토리")
        parser.add_argument("--patch-size", type=int, nargs=2, default=[224, 224], help="패치 크기 (너비 높이)")
        parser.add_argument("--padding-ratio", type=float, default=0.15, help="패딩 비율 (기본값: 0.15)")
        parser.add_argument("--min-quality", type=int, default=60, help="최소 품질 점수 (0-100, 기본값: 60)")
        parser.add_argument("--visualize", action="store_true", help="결과 시각화")
        parser.add_argument("--no-merge", action="store_true", help="인접 텍스트 상자 병합 비활성화")
        parser.add_argument("--gpu", action="store_true", help="GPU 사용 (가능한 경우)")
        args = parser.parse_args()
    
    # 의존성 확인
    check_dependencies()
    
    # 이제 필요한 모듈 임포트
    from craft_word_patch_pipeline import OptimizedWordPatchExtractor
    import cv2
    import torch
    import numpy as np
    import json
    import time
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    
    # 장치 설정
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"장치: {device}")
    
    # 패치 추출기 초기화
    extractor = OptimizedWordPatchExtractor(
        patch_size=tuple(args.patch_size),
        padding_ratio=args.padding_ratio,
        min_quality_score=args.min_quality,
        enable_merge=not args.no_merge,
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
        os.makedirs(patch_dir, exist_ok=True)
        
        for i, patch in enumerate(patches):
            patch_path = patch_dir / f"patch_{i:04d}.jpg"
            cv2.imwrite(str(patch_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        
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
            # 원본 이미지에 경계 상자 표시
            image_with_boxes = image.copy()
            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image_with_boxes, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 패치 그리드 생성
            n_patches = len(patches)
            if n_patches > 0:
                import math
                cols = min(8, n_patches)
                rows = math.ceil(n_patches / cols)
                
                plt.figure(figsize=(20, 4 * rows + 4))
                
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
                
                vis_path = output_dir / "visualization.jpg"
                plt.savefig(str(vis_path))
                plt.close()
                
                print(f"시각화 이미지 저장됨: {vis_path}")
            else:
                print("추출된 패치가 없습니다.")
    
    elif input_path.is_dir():
        # 디렉토리 내 모든 이미지 처리
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = [f for f in input_path.glob('**/*') if f.suffix.lower() in image_extensions]
        
        print(f"총 {len(image_files)}개 이미지 처리 예정")
        
        total_patches = 0
        total_time = 0
        
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
            start_time = time.time()
            patches, bboxes = extractor.extract_patches(image)
            elapsed_time = time.time() - start_time
            
            total_patches += len(patches)
            total_time += elapsed_time
            
            # 결과 저장
            os.makedirs(output_subdir, exist_ok=True)
            
            # 패치 저장
            patch_dir = output_subdir / "patches"
            os.makedirs(patch_dir, exist_ok=True)
            
            for i, patch in enumerate(patches):
                patch_path = patch_dir / f"patch_{i:04d}.jpg"
                cv2.imwrite(str(patch_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            
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
                # 원본 이미지에 경계 상자 표시
                image_with_boxes = image.copy()
                for i, bbox in enumerate(bboxes):
                    x, y, w, h = bbox
                    cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 시각화 이미지 저장
                vis_path = output_subdir / "visualization.jpg"
                cv2.imwrite(str(vis_path), cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
        
        print(f"처리 완료: 총 {total_patches}개 패치 추출 (평균 시간: {total_time/len(image_files):.2f}초/이미지)")
    
    else:
        print(f"입력 경로가 존재하지 않음: {args.input}")


if __name__ == "__main__":
    main()
