# 데이터셋 파일 구조화 도구
# 파일 이동, 디렉토리 생성, 이름 변경 등의 작업

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
import sqlite3
import pandas as pd
import random
from pathlib import Path
import argparse
import sys

def parse_arguments():
    # Jupyter 환경에서 실행될 때 sys.argv 정리
    if any(x.startswith('-f') for x in sys.argv):
        sys.argv = [sys.argv[0]]  # 다른 인자들 제거
        
    parser = argparse.ArgumentParser(description='MIL 데이터셋 파일 구조화 도구')
    parser.add_argument('--source_dir', type=str, default='./synth_output_mil_test/', 
                        help='Data2.py로 생성된 결과물이 있는 원본 디렉토리')
    parser.add_argument('--target_dir', type=str, default='./mil_dataset/', 
                        help='구조화된 파일을 저장할 대상 디렉토리')
    parser.add_argument('--split', action='store_true', 
                        help='데이터셋을 학습/검증/테스트로 분할할지 여부')
    parser.add_argument('--train_ratio', type=float, default=0.7, 
                        help='학습 데이터 비율 (기본값: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15, 
                        help='검증 데이터 비율 (기본값: 0.15)')
    parser.add_argument('--create_index', action='store_true', 
                        help='검색 가능한 메타데이터 인덱스를 생성할지 여부')
    return parser.parse_args()

def organize_files(source_dir, target_dir, create_split=False, train_ratio=0.7, val_ratio=0.15, create_index=False):
    """
    Data2.py로 생성된 파일들을 구조화된 형태로 정리합니다.
    """
    # 대상 디렉토리 생성
    os.makedirs(target_dir, exist_ok=True)
    
    # 로그 디렉토리(메타데이터) 경로 구성
    source_logs_dir = os.path.join(source_dir, 'logs')
    
    # 데이터 파일 매핑을 위한 딕셔너리
    all_files = []
    single_author_files = []
    multi_author_files = []
    
    # 모든 로그 파일을 순회하며 처리
    print(f"원본 디렉토리 {source_dir}에서 파일 분석 중...")
    
    # 로그 디렉토리 존재 확인
    if not os.path.exists(source_logs_dir):
        source_logs_dir = source_dir  # 로그 디렉토리가 없으면 원본 디렉토리에서 직접 검색
        
    # 로그 파일 수집
    log_files = [f for f in os.listdir(source_logs_dir) if f.endswith('_log.json')]
    total_files = len(log_files)
    print(f"총 {total_files}개의 메타데이터 파일을 찾았습니다.")
    
    for i, log_filename in enumerate(log_files):
        if i % 100 == 0:
            print(f"진행 중: {i}/{total_files} 파일 처리됨")
            
        log_path = os.path.join(source_logs_dir, log_filename)
        
        try:
            # 메타데이터 로드
            with open(log_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 이미지와 마스크 파일 경로 확인
            img_path = metadata.get('output_image_file')
            mask_path = metadata.get('output_mask_file')
            
            if not img_path or not mask_path:
                print(f"경고: {log_filename}에 이미지/마스크 경로가 없습니다. 건너뜁니다.")
                continue
                
            # 상대 경로로 변환된 경우 처리
            if not os.path.isabs(img_path):
                img_path = os.path.join(source_dir, os.path.basename(img_path))
            if not os.path.isabs(mask_path):
                mask_path = os.path.join(source_dir, os.path.basename(mask_path))
                
            # 파일 존재 여부 확인
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"경고: {log_filename}에 참조된 이미지/마스크가 존재하지 않습니다. 건너뜁니다.")
                continue
                
            # 문서 유형(단일/다중 작성자) 확인
            is_multi_author = metadata.get('bag_labels', {}).get('is_multi_author', False)
            authors = metadata.get('bag_labels', {}).get('authors_present_str', [])
            
            if not authors:
                print(f"경고: {log_filename}에 작성자 정보가 없습니다. 건너뜁니다.")
                continue
                
            # 파일 정보 수집
            file_info = {
                'metadata_path': log_path,
                'img_path': img_path, 
                'mask_path': mask_path,
                'is_multi_author': is_multi_author,
                'authors': authors,
                'log_filename': log_filename,
                'img_filename': os.path.basename(img_path),
                'mask_filename': os.path.basename(mask_path)
            }
            
            all_files.append(file_info)
            
            if is_multi_author:
                multi_author_files.append(file_info)
            else:
                single_author_files.append(file_info)
                
        except Exception as e:
            print(f"오류: {log_filename} 처리 중 문제 발생: {e}")
    
    print(f"분석 완료: 단일 작성자 문서 {len(single_author_files)}개, 다중 작성자 문서 {len(multi_author_files)}개")
    
    # 파일 복사 및 구조화
    copy_files_to_structured_dirs(single_author_files, multi_author_files, target_dir)
    
    # 데이터셋 분할 생성
    if create_split:
        create_dataset_splits(all_files, target_dir, train_ratio, val_ratio)
    
    # 메타데이터 인덱스 생성
    if create_index:
        create_metadata_index(all_files, target_dir)
    
    return len(all_files)

def copy_files_to_structured_dirs(single_author_files, multi_author_files, target_dir):
    """파일을 구조화된 디렉토리로 복사합니다."""
    print("파일을 구조화된 디렉토리로 복사 중...")
    
    # 단일 작성자 문서 처리
    single_dir = os.path.join(target_dir, 'single_author')
    os.makedirs(single_dir, exist_ok=True)
    
    for file_info in single_author_files:
        author = file_info['authors'][0]  # 단일 작성자는 첫 번째(유일) 작성자
        
        # 작성자별 디렉토리 생성
        author_dir = os.path.join(single_dir, f"writer_{author}")
        images_dir = os.path.join(author_dir, 'images')
        masks_dir = os.path.join(author_dir, 'masks')
        metadata_dir = os.path.join(author_dir, 'metadata')
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        
        # 파일 복사
        shutil.copy2(file_info['img_path'], os.path.join(images_dir, file_info['img_filename']))
        shutil.copy2(file_info['mask_path'], os.path.join(masks_dir, file_info['mask_filename']))
        shutil.copy2(file_info['metadata_path'], os.path.join(metadata_dir, file_info['log_filename']))
    
    # 다중 작성자 문서 처리
    multi_dir = os.path.join(target_dir, 'multi_author')
    os.makedirs(multi_dir, exist_ok=True)
    
    for file_info in multi_author_files:
        authors = file_info['authors']
        authors.sort()  # 일관된 폴더명을 위해 정렬
        
        # 작성자 쌍별 디렉토리 생성
        pair_name = f"pair_{'_'.join(authors)}"
        pair_dir = os.path.join(multi_dir, pair_name)
        images_dir = os.path.join(pair_dir, 'images')
        masks_dir = os.path.join(pair_dir, 'masks')
        metadata_dir = os.path.join(pair_dir, 'metadata')
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        
        # 파일 복사
        shutil.copy2(file_info['img_path'], os.path.join(images_dir, file_info['img_filename']))
        shutil.copy2(file_info['mask_path'], os.path.join(masks_dir, file_info['mask_filename']))
        shutil.copy2(file_info['metadata_path'], os.path.join(metadata_dir, file_info['log_filename']))
    
    print("파일 복사 완료")

def create_dataset_splits(all_files, target_dir, train_ratio, val_ratio):
    """데이터셋을 학습/검증/테스트로 분할합니다."""
    print("데이터셋 분할 생성 중...")
    
    # 랜덤하게 섞기
    random.seed(42)  # 재현성을 위한 시드 설정
    random.shuffle(all_files)
    
    total = len(all_files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size+val_size]
    test_files = all_files[train_size+val_size:]
    
    splits_dir = os.path.join(target_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    # 상대 경로로 변환
    base_path = Path(target_dir)
    
    def get_relative_paths(file_list):
        result = []
        for f in file_list:
            # 구조화된 새 경로로 변환
            if f['is_multi_author']:
                authors = sorted(f['authors'])
                pair_name = f"pair_{'_'.join(authors)}"
                img_rel_path = f"multi_author/{pair_name}/images/{f['img_filename']}"
                mask_rel_path = f"multi_author/{pair_name}/masks/{f['mask_filename']}"
                meta_rel_path = f"multi_author/{pair_name}/metadata/{f['log_filename']}"
            else:
                author = f['authors'][0]
                img_rel_path = f"single_author/writer_{author}/images/{f['img_filename']}"
                mask_rel_path = f"single_author/writer_{author}/masks/{f['mask_filename']}"
                meta_rel_path = f"single_author/writer_{author}/metadata/{f['log_filename']}"
            
            result.append({
                'img_path': img_rel_path,
                'mask_path': mask_rel_path,
                'metadata_path': meta_rel_path,
                'is_multi_author': f['is_multi_author'],
                'authors': f['authors']
            })
        return result
    
    # 분할 정보 저장
    with open(os.path.join(splits_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(get_relative_paths(train_files), f, indent=2)
    
    with open(os.path.join(splits_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(get_relative_paths(val_files), f, indent=2)
    
    with open(os.path.join(splits_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(get_relative_paths(test_files), f, indent=2)
    
    print(f"데이터셋 분할 완료: 학습 {len(train_files)}개, 검증 {len(val_files)}개, 테스트 {len(test_files)}개")

def create_metadata_index(all_files, target_dir):
    """검색 가능한 메타데이터 인덱스를 생성합니다."""
    print("메타데이터 인덱스 생성 중...")
    
    index_dir = os.path.join(target_dir, 'index')
    os.makedirs(index_dir, exist_ok=True)
    
    # CSV 인덱스 생성
    df = pd.DataFrame(all_files)
    csv_path = os.path.join(index_dir, 'all_files.csv')
    df.to_csv(csv_path, index=False)
    
    # SQLite 인덱스 생성
    db_path = os.path.join(index_dir, 'metadata_index.db')
    
    # 기존 DB 삭제
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    
    # 기본 파일 정보 테이블 생성
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE files (
        id INTEGER PRIMARY KEY,
        metadata_path TEXT,
        img_path TEXT,
        mask_path TEXT,
        is_multi_author INTEGER,
        log_filename TEXT,
        img_filename TEXT,
        mask_filename TEXT
    )
    ''')
    
    # 작성자 정보 테이블 생성
    cursor.execute('''
    CREATE TABLE author_document (
        id INTEGER PRIMARY KEY,
        file_id INTEGER,
        author_id TEXT,
        FOREIGN KEY (file_id) REFERENCES files (id)
    )
    ''')
    
    # 데이터 삽입
    for i, f in enumerate(all_files):
        cursor.execute('''
        INSERT INTO files (id, metadata_path, img_path, mask_path, is_multi_author, log_filename, img_filename, mask_filename)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (i, f['metadata_path'], f['img_path'], f['mask_path'], 
              1 if f['is_multi_author'] else 0, f['log_filename'], f['img_filename'], f['mask_filename']))
        
        # 작성자 관계 삽입
        for author in f['authors']:
            cursor.execute('''
            INSERT INTO author_document (file_id, author_id)
            VALUES (?, ?)
            ''', (i, author))
    
    # 검색 최적화를 위한 인덱스 생성
    cursor.execute('CREATE INDEX idx_is_multi_author ON files (is_multi_author)')
    cursor.execute('CREATE INDEX idx_author_id ON author_document (author_id)')
    
    conn.commit()
    conn.close()
    
    print(f"메타데이터 인덱스 생성 완료: {csv_path} 및 {db_path}")

def process_dataset(source_dir='./synth_output_mil_test/', target_dir='./mil_dataset/', create_split=True, train_ratio=0.7, val_ratio=0.15, create_index=True):
    """
    Jupyter Notebook 등에서 직접 호출할 수 있는 인터페이스 함수
    """
    print("=" * 60)
    print("MIL 데이터셋 파일 구조화 도구")
    print("=" * 60)
    print(f"원본 디렉토리: {source_dir}")
    print(f"대상 디렉토리: {target_dir}")
    print(f"데이터셋 분할: {'예' if create_split else '아니오'}")
    print(f"인덱스 생성: {'예' if create_index else '아니오'}")
    print("=" * 60)
    
    total_files = organize_files(
        source_dir, 
        target_dir, 
        create_split=create_split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        create_index=create_index
    )
    
    print("=" * 60)
    print(f"처리 완료: 총 {total_files}개 파일이 구조화되었습니다.")
    print(f"결과물 위치: {target_dir}")
    print("=" * 60)
    
    return total_files

def main():
    args = parse_arguments()
    
    print("=" * 60)
    print("MIL 데이터셋 파일 구조화 도구")
    print("=" * 60)
    print(f"원본 디렉토리: {args.source_dir}")
    print(f"대상 디렉토리: {args.target_dir}")
    print(f"데이터셋 분할: {'예' if args.split else '아니오'}")
    print(f"인덱스 생성: {'예' if args.create_index else '아니오'}")
    print("=" * 60)
    
    total_files = organize_files(
        args.source_dir, 
        args.target_dir, 
        create_split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        create_index=args.create_index
    )
    
    print("=" * 60)
    print(f"처리 완료: 총 {total_files}개 파일이 구조화되었습니다.")
    print(f"결과물 위치: {args.target_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main() 