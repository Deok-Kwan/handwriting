#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIL Dataset Analysis
데이터셋 기본 특성 파악 및 시각화 스크립트
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import Counter
import cv2

# 1. GPU 설정 확인
def check_gpu():
    """GPU 가용성 확인 및 설정"""
    print("===== GPU 설정 확인 =====")
    gpu_available = torch.cuda.is_available()
    print(f"GPU 사용 가능: {gpu_available}")
    
    if gpu_available:
        gpu_count = torch.cuda.device_count()
        print(f"사용 가능한 GPU 수: {gpu_count}")
        print(f"현재 GPU: {torch.cuda.get_device_name(0)}")
        
        # 사용할 GPU 설정 (cuda:0 사용)
        device = torch.device("cuda:0")
        print(f"사용할 장치: {device}")
    else:
        device = torch.device("cpu")
        print(f"GPU를 사용할 수 없어 CPU를 사용합니다: {device}")
        
    return device

# 2. 데이터셋 구조 확인
def explore_dataset_structure(dataset_root):
    """데이터셋 폴더 구조 탐색"""
    print("\n===== 데이터셋 구조 탐색 =====")
    
    # 주요 디렉토리 리스트
    main_dirs = [
        "single_author", "multi_author", "splits", "index", "visualization_results"
    ]
    
    for dir_name in main_dirs:
        dir_path = os.path.join(dataset_root, dir_name)
        if os.path.exists(dir_path):
            print(f"\n{dir_name} 디렉토리:")
            
            if dir_name == "single_author":
                authors = [os.path.basename(d) for d in glob.glob(os.path.join(dir_path, "writer_*"))]
                print(f"  - 단일 작성자 수: {len(authors)}")
                print(f"  - 작성자 예시: {', '.join(authors[:5])}{'...' if len(authors) > 5 else ''}")
                
                # 예시 작성자 폴더 내용 확인
                if authors:
                    sample_author = authors[0]
                    sample_path = os.path.join(dir_path, sample_author)
                    subdirs = os.listdir(sample_path)
                    print(f"  - {sample_author} 폴더 구조: {', '.join(subdirs)}")
                    
                    # 각 폴더 내 파일 수 확인
                    for subdir in subdirs:
                        files = glob.glob(os.path.join(sample_path, subdir, "*"))
                        print(f"    - {subdir}: {len(files)}개 파일")
            
            elif dir_name == "multi_author":
                pairs = [os.path.basename(d) for d in glob.glob(os.path.join(dir_path, "pair_*"))]
                print(f"  - 작성자 쌍 수: {len(pairs)}")
                print(f"  - 쌍 예시: {', '.join(pairs[:3])}{'...' if len(pairs) > 3 else ''}")
                
                # 예시 쌍 폴더 내용 확인
                if pairs:
                    sample_pair = pairs[0]
                    sample_path = os.path.join(dir_path, sample_pair)
                    subdirs = os.listdir(sample_path)
                    print(f"  - {sample_pair} 폴더 구조: {', '.join(subdirs)}")
                    
                    # 각 폴더 내 파일 수 확인
                    for subdir in subdirs:
                        files = glob.glob(os.path.join(sample_path, subdir, "*"))
                        print(f"    - {subdir}: {len(files)}개 파일")
            
            elif dir_name == "splits":
                split_files = os.listdir(dir_path)
                print(f"  - 분할 파일: {', '.join(split_files)}")
                
                # 각 분할 파일 내 샘플 수 확인
                for split_file in split_files:
                    with open(os.path.join(dir_path, split_file), 'r') as f:
                        data = json.load(f)
                        print(f"    - {split_file}: {len(data)}개 샘플")
            
            elif dir_name == "index":
                index_files = os.listdir(dir_path)
                print(f"  - 인덱스 파일: {', '.join(index_files)}")
                
                # CSV 파일 간단 확인
                csv_files = [f for f in index_files if f.endswith('.csv')]
                if csv_files:
                    with open(os.path.join(dir_path, csv_files[0]), 'r') as f:
                        head = [next(f).strip() for _ in range(2)]
                        print(f"    - CSV 헤더: {head[0]}")
                        
            else:
                files = os.listdir(dir_path)
                print(f"  - 총 {len(files)}개 파일")
                
        else:
            print(f"경고: {dir_name} 디렉토리가 존재하지 않습니다.")

# 3. 기본 통계 계산
def compute_dataset_statistics(dataset_root):
    """데이터셋 기본 통계 계산"""
    print("\n===== 데이터셋 기본 통계 =====")
    
    # 분할 파일에서 통계 추출
    splits_dir = os.path.join(dataset_root, "splits")
    all_data = []
    
    for split_file in ["train.json", "val.json", "test.json"]:
        file_path = os.path.join(splits_dir, split_file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
                
    # 문서 통계
    total_docs = len(all_data)
    print(f"전체 문서 수: {total_docs}")
    
    # 단일/다중 작성자 문서 비율
    single_author_docs = [d for d in all_data if not d.get('is_multi_author', False)]
    multi_author_docs = [d for d in all_data if d.get('is_multi_author', False)]
    
    print(f"단일 작성자 문서: {len(single_author_docs)}개 ({len(single_author_docs)/total_docs*100:.1f}%)")
    print(f"다중 작성자 문서: {len(multi_author_docs)}개 ({len(multi_author_docs)/total_docs*100:.1f}%)")
    
    # 작성자 통계
    all_authors = []
    for doc in all_data:
        all_authors.extend(doc.get('authors', []))
        
    unique_authors = sorted(set(all_authors))
    author_counts = Counter(all_authors)
    
    print(f"고유 작성자 수: {len(unique_authors)}")
    print(f"작성자 목록: {', '.join(unique_authors[:10])}{'...' if len(unique_authors) > 10 else ''}")
    
    # 가장 많이 등장하는 작성자 Top 5
    most_common = author_counts.most_common(5)
    print("가장 많이 등장하는 작성자 Top 5:")
    for author, count in most_common:
        print(f"  - {author}: {count}회")
    
    # 메타데이터 샘플 확인
    if all_data:
        sample_doc = all_data[0]
        meta_path = os.path.join(dataset_root, sample_doc.get('metadata_path', ''))
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                print("\n메타데이터 예시 (주요 필드):")
                for key in ['timestamp', 'generation_method', 'parameters']:
                    if key in meta:
                        print(f"  - {key}: {meta[key]}")
                
                if 'bag_labels' in meta:
                    print(f"  - bag_labels: {meta['bag_labels']}")
                
                if 'instance_labels' in meta:
                    print(f"  - instance_labels 수: {len(meta.get('instance_labels', []))}개")

# 4. 샘플 이미지 및 마스크 시각화
def visualize_samples(dataset_root, num_samples=3):
    """샘플 이미지와 마스크 시각화"""
    print("\n===== Sample Images and Masks Visualization =====")
    
    # 단일 작성자 샘플
    single_author_samples = []
    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_root, "splits", f"{split}.json")
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                data = json.load(f)
                single_samples = [d for d in data if not d.get('is_multi_author', False)]
                if single_samples:
                    single_author_samples.extend(single_samples[:1])  # 각 분할에서 1개씩만
        if len(single_author_samples) >= num_samples:
            break
            
    # 다중 작성자 샘플
    multi_author_samples = []
    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_root, "splits", f"{split}.json")
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                data = json.load(f)
                multi_samples = [d for d in data if d.get('is_multi_author', False)]
                if multi_samples:
                    multi_author_samples.extend(multi_samples[:1])  # 각 분할에서 1개씩만
        if len(multi_author_samples) >= num_samples:
            break
    
    # 시각화 함수
    def visualize_doc(doc_info, title, fig_idx):
        img_path = os.path.join(dataset_root, doc_info.get('img_path', ''))
        mask_path = os.path.join(dataset_root, doc_info.get('mask_path', ''))
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            # 이미지 로드
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 마스크 로드
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            # 마스크의 고유값 확인
            unique_vals = np.unique(mask)
            
            plt.figure(fig_idx, figsize=(15, 8))
            
            # 이미지 표시
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"{title} - Image")
            plt.axis('off')
            
            # 마스크 표시
            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='rainbow')
            plt.title(f"{title} - Mask (Unique values: {unique_vals})")
            plt.axis('off')
            
            plt.tight_layout()
            
            return True
        else:
            print(f"Warning: Image or mask file not found - {img_path}, {mask_path}")
            return False
    
    # 단일 작성자 문서 시각화
    if single_author_samples:
        sample = single_author_samples[0]
        if visualize_doc(sample, f"Single Author Document (Author: {', '.join(sample.get('authors', []))})", 1):
            print("Single author document visualization completed")
    
    # 다중 작성자 문서 시각화
    if multi_author_samples:
        sample = multi_author_samples[0]
        if visualize_doc(sample, f"Multi Author Document (Authors: {', '.join(sample.get('authors', []))})", 2):
            print("Multi author document visualization completed")
    
    plt.show()

# 메인 함수
def main():
    """메인 함수"""
    # 데이터셋 경로
    dataset_root = "/home/dk926/workspace/MIL/mil_dataset"
    
    # 1. GPU 설정
    device = check_gpu()
    
    # 2. 데이터셋 구조 탐색
    explore_dataset_structure(dataset_root)
    
    # 3. 기본 통계 계산
    compute_dataset_statistics(dataset_root)
    
    # 4. 샘플 시각화
    visualize_samples(dataset_root)
    
    print("\n===== 분석 완료 =====")

if __name__ == "__main__":
    main() 