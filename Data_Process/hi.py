#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 필요한 라이브러리 및 패키지 임포트
import os
import cv2
import numpy as np
import random
import pandas as pd
import json
import datetime
from skimage.exposure import match_histograms
import traceback




# 경로 설정
# 디렉토리 경로 설정
BACKGROUND_DIR = "/home/dk926/workspace/Hand/original"
HANDWRITING_DIR = "/home/dk926/workspace/MIL/line_segments"
OUTPUT_DIR = './synth_output/'
LOG_DIR = './synth_output/logs/'

# 특정 파일 경로 설정 (예시 파일)
BACKGROUND_EXAMPLE = os.path.join(BACKGROUND_DIR, "w0001_s01_pLND_r01.png")
HANDWRITING_EXAMPLE = os.path.join(HANDWRITING_DIR, "w0001_pWOZ_line_0.png")

# 데이터 로딩
def load_data():
    """
    배경 이미지와 손글씨 이미지 데이터를 로드합니다.
    """
    # 1. 파일 리스트 수집
    background_files = [os.path.join(BACKGROUND_DIR, f) for f in os.listdir(BACKGROUND_DIR) if f.endswith('.png')]
    handwriting_files = [os.path.join(HANDWRITING_DIR, f) for f in os.listdir(HANDWRITING_DIR) if f.endswith('.png')]
    
    print(f"배경 이미지 파일 수: {len(background_files)}")
    print(f"손글씨 이미지 파일 수: {len(handwriting_files)}")
    
    return background_files, handwriting_files

# 샘플 선택
def select_samples(background_files, handwriting_files, num_background_samples=10, num_handwriting_samples=50):
    """
    효율적 처리를 위해 일부 샘플만 선택합니다.
    """
    selected_backgrounds = random.sample(background_files, min(num_background_samples, len(background_files)))
    selected_handwritings = random.sample(handwriting_files, min(num_handwriting_samples, len(handwriting_files)))
    
    return selected_backgrounds, selected_handwritings

# 레이아웃 분석 함수
def analyze_layout(doc_image):
    """
    문서 이미지에서 채울 수 있는 영역을 식별합니다.
    doc_image: numpy array (BGR or grayscale)
    returns: list of bounding boxes (x, y, w, h) for 'fillable regions'
    """
    # 데모용: 하단 영역을 채울 영역으로 반환
    h, w = doc_image.shape[:2]
    region = (0, int(h*0.7), w, int(h*0.3))  # (x=0, y=70%, width=w, height=30%)
    return [region]  # 바운딩 박스 리스트

# 이미지 처리 함수
def apply_histogram_matching(src_img, ref_img):
    """
    히스토그램 매칭을 적용합니다.
    """
    # 소스 및 참조 이미지 형식 확인
    if len(src_img.shape) == 3 and len(ref_img.shape) == 3:
        matched = match_histograms(src_img, ref_img, channel_axis=-1)
    else:
        matched = match_histograms(src_img, ref_img)
    return matched

def seamless_clone(src_img, dst_img, mask, center):
    """
    이미지를 자연스럽게 합성합니다.
    """
    output = cv2.seamlessClone(src_img, dst_img, mask, center, cv2.NORMAL_CLONE)
    return output

# 파일 이름 추출 함수
def extract_filename(path):
    """전체 경로에서 파일 이름만 추출합니다."""
    return os.path.basename(path)

# 로그 파일 생성 함수
def create_log_file(result_info, filename):
    """합성 정보를 로그 파일로 저장합니다."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, filename)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(result_info, f, ensure_ascii=False, indent=2)
    
    return log_path

# 이미지에 텍스트 추가 함수
def add_text_to_image(image, text, position=(10, 30), font_scale=0.7, color=(0, 0, 255), thickness=2):
    """이미지에 텍스트를 추가합니다."""
    # 긴 텍스트를 여러 줄로 나누기
    max_width = 80
    lines = []
    for t in text:
        if len(t) > max_width:
            # 긴 줄은 여러 줄로 나누기
            for i in range(0, len(t), max_width):
                lines.append(t[i:i+max_width])
        else:
            lines.append(t)
    
    # 이미지에 텍스트 추가
    result = image.copy()
    y_offset = position[1]
    for line in lines:
        cv2.putText(result, line, (position[0], y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y_offset += 30
    
    return result

# 시나리오 1: 기존 문서 + 아래에 다른 Writer
def insert_writer_into_doc(doc_path, writer_line_path):
    """
    1) 문서 로드
    2) 레이아웃 분석(또는 하단 여백 바운딩 박스)
    3) Writer 라인 로드 후 히스토그램 매칭 + seamlessClone 적용
    """
    doc_img = cv2.imread(doc_path)
    if doc_img is None:
        raise FileNotFoundError(f"문서 파일을 찾을 수 없습니다: {doc_path}")

    # 레이아웃 분석하여 "삽입할 바운딩 박스" 찾기
    regions = analyze_layout(doc_img)
    if not regions:
        return doc_img  # 영역을 찾지 못한 경우 원본 반환

    x, y, w, h = regions[0]  # 첫 번째 영역 선택
    
    # Writer 라인 로드
    src_line = cv2.imread(writer_line_path)
    if src_line is None:
        raise FileNotFoundError(f"손글씨 파일을 찾을 수 없습니다: {writer_line_path}")

    # 히스토그램 매칭 적용
    matched_line = apply_histogram_matching(src_line, doc_img)
    
    # 마스크 생성 (손글씨 영역)
    mask = np.where(matched_line < 250, 255, 0).astype(np.uint8)
    
    # 중심점 계산
    center = (x + matched_line.shape[1]//2, y + matched_line.shape[0]//2)
    
    # Seamless 클론 적용
    result = seamless_clone(matched_line, doc_img, mask, center)
    
    # 합성 정보 추가
    doc_name = extract_filename(doc_path)
    writer_name = extract_filename(writer_line_path)
    info_text = [
        f"배경 문서: {doc_name}",
        f"손글씨: {writer_name}"
    ]
    result = add_text_to_image(result, info_text)
    
    return result, {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "background_document": doc_path,
        "handwriting_document": writer_line_path,
        "method": "insert_writer_into_doc",
        "region": f"x={x}, y={y}, w={w}, h={h}"
    }

# 시나리오 2: 새 문서(빈 배경) + 최대 2명의 Writers
def create_multi_doc(lines_list, doc_width=1000, line_spacing=20, max_writers=2):
    """
    최대 max_writers명의 작성자의 손글씨 라인을 하나의 문서로 만듭니다.
    각 라인에 대한 인스턴스 라벨(bbox, author_id)을 생성합니다.
    
    lines_list: 경로 리스트(각각 다른 작성자)
    max_writers: 최대 사용할 작성자 수 (기본값: 2)
    
    returns:
        final_img: 합성된 이미지 (주석 없음)
        multi_info: 메타데이터 (MIL 라벨, 인스턴스 라벨 포함)
    """
    # 최대 작성자 수 제한
    if len(lines_list) > max_writers:
        print(f"경고: 최대 {max_writers}명의 필적만 사용합니다. {len(lines_list) - max_writers}개 필적이 무시됩니다.")
        lines_list = lines_list[:max_writers]
    
    # 흰색 문서 초기화
    final_img = np.full((0, doc_width, 3), 255, dtype=np.uint8)
    
    # 추적 정보
    used_files = []
    y_positions = []
    instances = []        # 인스턴스 정보 저장할 리스트
    current_y = 0
    writer_count = 0  # 작성자 수 추적
    
    # 필적 이미지 로드
    loaded_images = []
    valid_paths = []
    
    for line_path in lines_list:
        # 최대 작성자 수 확인
        if writer_count >= max_writers:
            break
            
        src_line = cv2.imread(line_path)
        if src_line is None:
            print(f"경고: 파일을 찾을 수 없습니다: {line_path}")
            continue
        
        loaded_images.append(src_line)
        valid_paths.append(line_path)
        writer_count += 1
    
    # 유효한 이미지가 없으면 빈 이미지와 정보 반환
    if not loaded_images:
        print("경고: 유효한 필적 이미지가 없습니다.")
        return final_img, {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "handwriting_files": [],
            "writer_count": 0,
            "method": "create_multi_doc",
            "doc_width": doc_width,
            "line_spacing": line_spacing,
            "y_positions": [],
            "max_writers_allowed": max_writers,
            "MIL_label": "single_author",
            "instances": [],
            "normalized": False
        }
    
    # 참조 이미지 선택 (첫 번째 이미지로 설정)
    reference_img = loaded_images[0]
    
    # 모든 이미지 처리 및 최종 이미지 합성
    writer_count = 0
    current_y = 0
    
    for i, (img, line_path) in enumerate(zip(loaded_images, valid_paths)):
        # 비정규화 모드: 단순 크기 조정만 수행
        scale = doc_width / img.shape[1]
        new_height = int(img.shape[0] * scale)
        
        # 이미지 리사이즈
        resized = cv2.resize(img, (doc_width, new_height))
        
        instance_info = {
            "author_id": extract_filename(line_path).split("_")[0],
            "file_path": line_path,
            "position_index": writer_count,
            "normalized_height": new_height,
            "normalized_width": doc_width,
            "original_width": img.shape[1],
            "original_height": img.shape[0],
            "scaling_factor": scale
        }
        
        # 여백 추가
        if final_img.shape[0] > 0:
            space = np.full((line_spacing, doc_width, 3), 255, dtype=np.uint8)
            final_img = np.concatenate((final_img, space), axis=0)
            current_y += line_spacing
            
        # 현재 높이 기록
        y_positions.append(current_y)
        
        # 세로 합성
        final_img = np.concatenate((final_img, resized), axis=0)
        
        # 인스턴스 라벨(bbox) 계산
        instance_info["bbox"] = [0, current_y, doc_width, resized.shape[0]]
        
        # 인스턴스 라벨 추가
        instances.append(instance_info)
        
        current_y += resized.shape[0]
        used_files.append(line_path)
        writer_count += 1
    
    # MIL 라벨: writer_count=1 → single_author, >1 → multi_author
    mil_label = "single_author" if writer_count <= 1 else "multi_author"
    
    result_info = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "handwriting_files": used_files,
        "writer_count": writer_count,
        "method": "create_multi_doc",
        "doc_width": doc_width,
        "line_spacing": line_spacing,
        "y_positions": y_positions,
        "max_writers_allowed": max_writers,
        "MIL_label": mil_label,
        "instances": instances,
        "normalized": False
    }
    
    return final_img, result_info

# 메인 실행 코드
def main():
    """
    주요 실행 함수 - 비정규화 버전만 생성
    """
    # 데이터 로드
    background_files, handwriting_files = load_data()
    
    # 샘플 선택
    _, selected_handwritings = select_samples(
        background_files, handwriting_files, 
        num_background_samples=0,  # 배경 이미지는 불필요
        num_handwriting_samples=10
    )
    
    # 출력 디렉토리 확인 및 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 시나리오: 최대 2명의 필적 합성
    if len(selected_handwritings) >= 2:
        # 최대 2명의 필적만 사용
        lines_list = selected_handwritings[:2]
        
        try:
            # 두 필적 ID 추출
            author_ids = [extract_filename(path).split("_")[0] for path in lines_list[:2]]
            author_prefix = f"{author_ids[0]}_{author_ids[1]}"
            
            print(f"문서 합성 실행: {len(lines_list)}개 필적 사용 (최대 2개)")
            print("비정규화 버전 (단순 배치)")
            
            # 비정규화 버전 생성 (단순 배치)
            multi_img_raw, multi_info_raw = create_multi_doc(
                lines_list, 
                doc_width=1200, 
                line_spacing=30, 
                max_writers=2
            )
            
            # 비정규화 버전 저장
            raw_filename = f"raw_{author_prefix}_{timestamp}.png"
            raw_path = os.path.join(OUTPUT_DIR, raw_filename)
            cv2.imwrite(raw_path, multi_img_raw)
            
            # 로그 파일 생성
            raw_log_filename = f"raw_log_{timestamp}.json"
            multi_info_raw["output_file"] = raw_path
            raw_log_path = create_log_file(multi_info_raw, raw_log_filename)
            
            print(f"비정규화 버전 저장됨: {raw_path}")
            print(f"비정규화 로그 저장됨: {raw_log_path}")
            
        except Exception as e:
            print(f"문서 합성 중 오류 발생: {e}")
            traceback.print_exc()
    else:
        print(f"오류: 필적 합성에는 최소 2개의 필적이 필요합니다. 현재 {len(selected_handwritings)}개만 사용 가능합니다.")
    
    print("합성 문서 생성 완료")

# 스크립트로 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()


