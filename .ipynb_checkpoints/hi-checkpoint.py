#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 기본 모듈 임포트
import os
import sys
import traceback
import json
import datetime
import random

print("=== 스크립트 시작 ===")
print(f"Python 버전: {sys.version}")
print(f"실행 경로: {os.path.abspath(__file__)}")
print(f"현재 작업 디렉토리: {os.getcwd()}")

# 필수 패키지 임포트 (예외 처리 포함)
try:
    print("NumPy 임포트 중...")
    import numpy as np
    print(f"NumPy 버전: {np.__version__}")
except ImportError:
    print("오류: NumPy 패키지를 찾을 수 없습니다. 다음 명령어로 설치하세요: pip install numpy")
    sys.exit(1)
except Exception as e:
    print(f"NumPy 임포트 중 오류 발생: {e}")
    sys.exit(1)

try:
    print("OpenCV 임포트 중...")
    import cv2
    print(f"OpenCV 버전: {cv2.__version__}")
except ImportError:
    print("오류: OpenCV 패키지를 찾을 수 없습니다. 다음 명령어로 설치하세요: pip install opencv-python")
    sys.exit(1)
except Exception as e:
    print(f"OpenCV 임포트 중 오류 발생: {e}")
    sys.exit(1)

# 스크립트 위치 기반 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"스크립트 위치: {script_dir}")

# 경로 설정
# 절대 경로로 설정
BACKGROUND_DIR = "/home/dk926/workspace/Hand/original"
HANDWRITING_DIR = "/home/dk926/workspace/MIL/line_segments"
OUTPUT_DIR = os.path.join(script_dir, 'synth_output')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

# 경로 존재 확인
print(f"BACKGROUND_DIR 존재: {os.path.exists(BACKGROUND_DIR)}")
print(f"HANDWRITING_DIR 존재: {os.path.exists(HANDWRITING_DIR)}")

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
print(f"출력 디렉토리 생성됨: {OUTPUT_DIR}")
print(f"로그 디렉토리 생성됨: {LOG_DIR}")

# 이미지 생성 함수
def create_dummy_image(width=800, height=600, text="테스트 이미지"):
    """더미 이미지 생성"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (width//4, height//2), font, 1, (0, 0, 255), 2)
    return img

# 파일 이름 추출 함수
def extract_filename(path):
    """전체 경로에서 파일 이름만 추출합니다."""
    return os.path.basename(path)

# 데이터 로딩
def load_data():
    """배경 이미지와 손글씨 이미지 데이터를 로드합니다."""
    try:
        # 배경 이미지 로드
        background_files = [
            os.path.join(BACKGROUND_DIR, f) 
            for f in os.listdir(BACKGROUND_DIR) 
            if f.lower().endswith('.png')
        ]
        
        # 손글씨 이미지 로드
        handwriting_files = [
            os.path.join(HANDWRITING_DIR, f) 
            for f in os.listdir(HANDWRITING_DIR) 
            if f.lower().endswith('.png')
        ]
        
        print(f"배경 이미지 파일 수: {len(background_files)}")
        print(f"손글씨 이미지 파일 수: {len(handwriting_files)}")
        
        return background_files, handwriting_files
    except Exception as e:
        print(f"데이터 로드 중 오류: {e}")
        traceback.print_exc()
        return [], []

# 샘플 선택
def select_samples(background_files, handwriting_files, num_background_samples=5, num_handwriting_samples=5):
    """일부 샘플만 선택합니다."""
    try:
        selected_backgrounds = random.sample(
            background_files, 
            min(num_background_samples, len(background_files))
        )
        selected_handwritings = random.sample(
            handwriting_files, 
            min(num_handwriting_samples, len(handwriting_files))
        )
        return selected_backgrounds, selected_handwritings
    except Exception as e:
        print(f"샘플 선택 중 오류: {e}")
        # 오류 발생 시 첫 번째 파일 또는 빈 리스트 반환
        bg = background_files[:1] if background_files else []
        hw = handwriting_files[:1] if handwriting_files else []
        return bg, hw

# 이미지에 텍스트 추가 함수
def add_text_to_image(image, text_list, position=(10, 30), font_scale=0.7, color=(0, 0, 255), thickness=2):
    """이미지에 텍스트를 추가합니다."""
    try:
        result = image.copy()
        y_offset = position[1]
        
        for text in text_list:
            cv2.putText(
                result, text, (position[0], y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
            )
            y_offset += 30
        
        return result
    except Exception as e:
        print(f"텍스트 추가 중 오류: {e}")
        return image  # 오류 발생 시 원본 이미지 반환

# 단순 이미지 합성 함수 (안전한 버전)
def simple_overlay(background, overlay, position=(0, 0)):
    """단순히 한 이미지를 다른 이미지 위에 위치시킵니다."""
    try:
        # 배경 이미지 복사
        result = background.copy()
        
        # 오버레이 이미지 크기 확인
        h, w = overlay.shape[:2]
        x, y = position
        
        # 배경 크기 확인
        bg_h, bg_w = result.shape[:2]
        
        # 위치 조정 (배경 이미지를 벗어나지 않도록)
        if x < 0: x = 0
        if y < 0: y = 0
        if x + w > bg_w: w = bg_w - x
        if y + h > bg_h: h = bg_h - y
        
        # 오버레이 이미지 크기 조정 (필요한 경우)
        if w <= 0 or h <= 0:
            print("오버레이 영역이 유효하지 않습니다.")
            return result
            
        overlay_resized = cv2.resize(overlay, (w, h)) if (w, h) != overlay.shape[:2] else overlay
        
        # 알파 블렌딩 (단순 버전)
        roi = result[y:y+h, x:x+w]
        result[y:y+h, x:x+w] = cv2.addWeighted(roi, 0.5, overlay_resized, 0.5, 0)
        
        return result
    except Exception as e:
        print(f"이미지 합성 중 오류: {e}")
        traceback.print_exc()
        return background  # 오류 발생 시 원본 배경 반환

# 메인 실행 코드
def main():
    """주요 실행 함수"""
    try:
        # 데이터 로드
        background_files, handwriting_files = load_data()
        if not background_files or not handwriting_files:
            print("오류: 필요한 이미지 파일을 찾을 수 없습니다.")
            return
        
        # 샘플 선택
        selected_backgrounds, selected_handwritings = select_samples(
            background_files, handwriting_files, 
            num_background_samples=2, 
            num_handwriting_samples=3
        )
        
        # 타임스탬프
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 시나리오 1: 단순 이미지 합성
        print("\n--- 시나리오 1: 단순 이미지 합성 ---")
        if selected_backgrounds and selected_handwritings:
            bg_path = selected_backgrounds[0]
            hw_path = selected_handwritings[0]
            
            print(f"배경 이미지: {extract_filename(bg_path)}")
            print(f"손글씨 이미지: {extract_filename(hw_path)}")
            
            # 이미지 로드
            bg_img = cv2.imread(bg_path)
            hw_img = cv2.imread(hw_path)
            
            if bg_img is None or hw_img is None:
                print("오류: 이미지를 로드할 수 없습니다.")
                if bg_img is None: print(f"배경 로드 실패: {bg_path}")
                if hw_img is None: print(f"손글씨 로드 실패: {hw_path}")
            else:
                # 이미지 크기 확인
                bg_h, bg_w = bg_img.shape[:2]
                hw_h, hw_w = hw_img.shape[:2]
                print(f"배경 크기: {bg_w}x{bg_h}")
                print(f"손글씨 크기: {hw_w}x{hw_h}")
                
                # 간단한 이미지 합성 (하단 중앙에 배치)
                position = (bg_w // 2 - hw_w // 2, int(bg_h * 0.7))
                merged_img = simple_overlay(bg_img, hw_img, position)
                
                # 정보 텍스트 추가
                info_text = [
                    f"배경: {extract_filename(bg_path)}",
                    f"손글씨: {extract_filename(hw_path)}"
                ]
                merged_img = add_text_to_image(merged_img, info_text)
                
                # 결과 저장
                bg_name = os.path.splitext(extract_filename(bg_path))[0]
                hw_name = os.path.splitext(extract_filename(hw_path))[0]
                output_filename = f"simple_overlay_{bg_name}_with_{hw_name}_{timestamp}.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                cv2.imwrite(output_path, merged_img)
                print(f"결과 저장됨: {output_path}")
                
                # 로그 정보 저장
                log_info = {
                    "timestamp": timestamp,
                    "background": bg_path,
                    "handwriting": hw_path,
                    "output": output_path
                }
                
                log_path = os.path.join(LOG_DIR, f"simple_overlay_log_{timestamp}.json")
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(log_info, f, ensure_ascii=False, indent=2)
                print(f"로그 저장됨: {log_path}")
        
        # 시나리오 2: 여러 이미지 세로 배치
        print("\n--- 시나리오 2: 여러 이미지 세로 배치 ---")
        if len(selected_handwritings) >= 2:
            # 사용할 손글씨 이미지
            hw_images = []
            hw_names = []
            
            # 손글씨 이미지 로드
            for hw_path in selected_handwritings[:3]:  # 최대 3개
                hw_img = cv2.imread(hw_path)
                if hw_img is not None:
                    hw_images.append(hw_img)
                    hw_names.append(extract_filename(hw_path))
                else:
                    print(f"손글씨 로드 실패: {hw_path}")
            
            if hw_images:
                # 통일된 너비
                target_width = 800
                combined_height = 0
                resized_images = []
                
                # 모든 이미지 크기 조정
                for img in hw_images:
                    h, w = img.shape[:2]
                    new_h = int(h * (target_width / w))
                    resized = cv2.resize(img, (target_width, new_h))
                    resized_images.append(resized)
                    combined_height += new_h + 20  # 여백 추가
                
                # 결합된 이미지 생성
                combined_img = np.ones((combined_height, target_width, 3), dtype=np.uint8) * 255
                y_pos = 0
                
                # 각 이미지 배치
                for img in resized_images:
                    h = img.shape[0]
                    combined_img[y_pos:y_pos+h, :] = img
                    y_pos += h + 20  # 여백 추가
                
                # 정보 텍스트 추가
                info_text = ["결합된 이미지:"] + hw_names
                combined_img = add_text_to_image(combined_img, info_text)
                
                # 결과 저장
                output_filename = f"vertical_combined_{timestamp}.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                cv2.imwrite(output_path, combined_img)
                print(f"결과 저장됨: {output_path}")
                
                # 로그 정보 저장
                log_info = {
                    "timestamp": timestamp,
                    "handwritings": selected_handwritings[:3],
                    "output": output_path
                }
                
                log_path = os.path.join(LOG_DIR, f"vertical_combined_log_{timestamp}.json")
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(log_info, f, ensure_ascii=False, indent=2)
                print(f"로그 저장됨: {log_path}")
        
        print("\n=== 스크립트 완료 ===")
    except Exception as e:
        print(f"메인 함수 실행 중 오류 발생: {e}")
        traceback.print_exc()

# 스크립트로 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"전체 프로그램 실행 중 오류 발생: {e}")
        traceback.print_exc()