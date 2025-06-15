# 마스크 이미지 시각화

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 마스크 파일 경로를 지정하세요.
#    '_mask.png'로 끝나는 파일이어야 합니다!
mask_path = 'mil_dataset/multi_author/pair_w0001_w0003/masks/synth_w0001_w0003_pair_0_20250331000311118250_mask.png' # 예: './mil_dataset/single_author/writer_w0009/masks/synth_w0009_single_0_...._mask.png'

# 2. 마스크 파일을 원본 데이터 타입으로 로드합니다.
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

# 3. 파일 로드 성공 여부 확인
if mask is None:
    print(f"오류: 마스크 파일을 로드할 수 없습니다. 경로를 확인하세요: {mask_path}")
else:
    print(f"마스크 로드 성공: {mask_path}")
    print(f"원본 마스크 데이터 타입: {mask.dtype}")
    original_unique_values = np.unique(mask)
    print(f"원본 마스크 고유 픽셀 값: {original_unique_values}")

    # 4. 시각화를 위한 픽셀 값 증폭
    #    - 배경(0)은 그대로 0으로 유지됩니다.
    #    - 작성자 ID 1은 50, ID 2는 100 등으로 변환되어 회색 음영 차이가 커집니다.
    #    - scaling_factor 값을 조절하여 밝기 차이를 조절할 수 있습니다.
    scaling_factor = 50
    # astype(np.float32)를 사용하여 오버플로우 방지 후 uint8로 변환
    visual_mask = np.clip(mask.astype(np.float32) * scaling_factor, 0, 255).astype(np.uint8)

    scaled_unique_values = np.unique(visual_mask)
    print(f"스케일링된 마스크 고유 픽셀 값: {scaled_unique_values}")

    # 5. 증폭된 마스크 이미지를 화면에 표시합니다.
    plt.figure(figsize=(10, 10)) # 이미지 크기 조절 (선택 사항)
    plt.imshow(visual_mask, cmap='gray') # 'gray' 컬러맵 사용
    plt.title(f"Visualized Mask (Scaled by {scaling_factor}) - {mask_path.split('/')[-1]}")
    plt.colorbar(label='Scaled Pixel Value') # 컬러바 추가 (선택 사항)
    plt.show()