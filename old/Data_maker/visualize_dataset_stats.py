# 데이터 합성 후 데이터셋 통계 시각화

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
from matplotlib import font_manager, rc
import seaborn as sns
import pandas as pd
from collections import Counter, defaultdict

# 폰트 설정은 제거하고 영어로 진행
plt.rcParams['axes.unicode_minus'] = False
# 글꼴 굵기 기본 설정 추가
plt.rcParams['font.weight'] = 'bold'

def count_documents_per_author(mil_dataset_dir):
    """
    각 작성자별 참여 문서 수를 계산하는 함수
    """
    # 작성자별 카운터 초기화
    single_author_counts = defaultdict(int)
    multi_author_counts = defaultdict(int)
    
    # 단일 작성자 문서 카운트
    single_author_dirs = glob.glob(os.path.join(mil_dataset_dir, 'single_author', 'writer_*'))
    for dir_path in single_author_dirs:
        # 디렉토리 이름에서 작성자 ID 추출
        author_id = os.path.basename(dir_path).replace('writer_', '')
        # 해당 작성자의 이미지 수 계산
        img_count = len(glob.glob(os.path.join(dir_path, 'images', '*.png')))
        single_author_counts[author_id] += img_count
    
    # 다중 작성자 문서 카운트
    multi_author_dirs = glob.glob(os.path.join(mil_dataset_dir, 'multi_author', 'pair_*'))
    for dir_path in multi_author_dirs:
        # 디렉토리 이름에서 작성자 ID들 추출
        pair_name = os.path.basename(dir_path).replace('pair_', '')
        authors = pair_name.split('_')
        # 이미지 수 계산
        img_count = len(glob.glob(os.path.join(dir_path, 'images', '*.png')))
        # 각 작성자에게 카운트 추가
        for author in authors:
            multi_author_counts[author] += img_count
    
    return single_author_counts, multi_author_counts

def create_dataset_visualizations(mil_dataset_dir='mil_dataset'):
    # 데이터셋 통계 정보
    total_docs = 210
    single_author_docs = 20
    multi_author_docs = 190
    total_authors = 20
    train_docs = 147
    val_docs = 31
    test_docs = 32
    
    # 폴더 생성
    os.makedirs('visualization_results', exist_ok=True)
    
    # 1. 문서 유형 분포 (파이 차트)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    labels = ['Single Author Documents', 'Multiple Author Documents']
    sizes = [single_author_docs, multi_author_docs]
    percentages = [s/total_docs*100 for s in sizes]
    explode = (0.1, 0)  # 첫 번째 조각을 약간 분리
    
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
    ax1.axis('equal')  # 원형 파이 차트
    plt.title('Document Type Distribution', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualization_results/document_type_distribution.png', dpi=300)
    plt.close()
    
    # 2. 데이터 분할 분포 (파이 차트)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    labels = ['Training Set', 'Validation Set', 'Test Set']
    sizes = [train_docs, val_docs, test_docs]
    
    # 퍼센트와 숫자를 모두 표시하는 함수
    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{:.1f}%\n({:d})'.format(pct, val)
        return my_format
    
    ax2.pie(sizes, explode=(0.1, 0, 0), labels=labels, 
            autopct=autopct_format(sizes),
            shadow=True, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
    ax2.axis('equal')  # 원형 파이 차트
    plt.title('Dataset Split Distribution', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualization_results/dataset_split_distribution.png', dpi=300)
    plt.close()
    
    # 3. 단일/복수 작성자 문서 비교 (막대 그래프)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    categories = ['Single Author', 'Multiple Author', 'Total']
    values = [single_author_docs, multi_author_docs, total_docs]
    
    bar_colors = ['#3274A1', '#E1812C', '#3A923A']
    ax3.bar(categories, values, color=bar_colors)
    
    # 값 표시
    for i, v in enumerate(values):
        ax3.text(i, v + 3, str(v), ha='center', fontsize=14, fontweight='bold')
    
    plt.title('Document Count by Type', fontsize=18, fontweight='bold')
    plt.ylabel('Number of Documents', fontsize=14, fontweight='bold')
    plt.ylim(0, total_docs + 20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualization_results/document_type_comparison.png', dpi=300)
    plt.close()
    
    # 4. 작성자별 문서 분포 시각화 (수평 막대 그래프)
    # 실제 데이터에서 작성자별 문서 수 계산
    single_counts, multi_counts = count_documents_per_author(mil_dataset_dir)
    
    # 모든 작성자 ID 목록 생성
    all_authors = sorted(set(list(single_counts.keys()) + list(multi_counts.keys())))
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'Writer_ID': all_authors,
        'Single_Author_Docs': [single_counts.get(author, 0) for author in all_authors],
        'Multi_Author_Docs': [multi_counts.get(author, 0) for author in all_authors]
    })
    
    # ID 기준으로 정렬
    df = df.sort_values(by='Writer_ID')
    
    # 그래프 그리기
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    
    # 막대 그래프 생성
    x = np.arange(len(df))
    width = 0.35
    
    ax4.barh(x - width/2, df['Single_Author_Docs'], width, label='Single Author Documents', color='#3274A1')
    ax4.barh(x + width/2, df['Multi_Author_Docs'], width, label='Multiple Author Documents', color='#E1812C')
    
    ax4.set_yticks(x)
    ax4.set_yticklabels(df['Writer_ID'], fontsize=12, fontweight='bold')
    ax4.legend(fontsize=14, frameon=True)
    
    plt.title('Documents per Writer', fontsize=18, fontweight='bold')
    plt.xlabel('Number of Documents', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualization_results/author_document_distribution.png', dpi=300)
    plt.close()
    
    # 5. 히트맵: 작성자 쌍 빈도
    # 작성자 쌍 빈도 데이터 생성 (예: w0001_w0002 쌍이 몇 개의 문서에 등장하는지)
    pair_counts = defaultdict(int)
    multi_author_dirs = glob.glob(os.path.join(mil_dataset_dir, 'multi_author', 'pair_*'))
    
    for dir_path in multi_author_dirs:
        pair_name = os.path.basename(dir_path).replace('pair_', '')
        authors = pair_name.split('_')
        if len(authors) == 2:  # 작성자가 2명인 경우만 고려
            img_count = len(glob.glob(os.path.join(dir_path, 'images', '*.png')))
            pair_counts[(authors[0], authors[1])] += img_count
    
    # 작성자 목록
    unique_authors = sorted(set([a for pair in pair_counts.keys() for a in pair]))
    
    # 히트맵용 행렬 생성
    heatmap_data = np.zeros((len(unique_authors), len(unique_authors)))
    author_to_idx = {author: i for i, author in enumerate(unique_authors)}
    
    for (author1, author2), count in pair_counts.items():
        i, j = author_to_idx[author1], author_to_idx[author2]
        heatmap_data[i, j] = count
        heatmap_data[j, i] = count  # 대칭으로 설정
    
    # 히트맵 그리기 (너무 크면 작성자 수를 제한할 수 있음)
    if len(unique_authors) <= 20:  # 작성자 수가 20명 이하일 때
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data.astype(int), annot=True, fmt='.0f', cmap='YlGnBu', 
                    xticklabels=unique_authors, yticklabels=unique_authors,
                    annot_kws={"fontsize":12, "fontweight":"bold"})
        plt.title('Writer Pair Frequency Heatmap', fontsize=18, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold', rotation=45)
        plt.yticks(fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualization_results/author_pair_heatmap.png', dpi=300)
        plt.close()
    
    print("Visualization completed! Results saved in 'visualization_results' folder.")

def create_comparative_visualizations():
    """
    원본 데이터, 라인 세그먼트, 최종 데이터셋 간의 비교 시각화를 생성하는 함수
    """
    # 데이터셋 통계 정보
    original_images = 12825
    line_segments = 426
    final_dataset = 210
    
    # 작성자 수 통계
    original_authors = 475
    line_segment_authors = 426
    final_authors = 20
    
    # 학습/검증/테스트 분할 (원본)
    original_train = 10800
    original_val = 945
    original_test = 1080
    
    # 학습/검증/테스트 분할 (최종)
    final_train = 147
    final_val = 31
    final_test = 32
    
    # 폴더 생성
    os.makedirs('visualization_results', exist_ok=True)
    
    # 1. 데이터셋 크기 비교 (막대 그래프)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    categories = ['Original Dataset', 'Line Segments', 'Final Dataset']
    values = [original_images, line_segments, final_dataset]
    
    bar_colors = ['#3274A1', '#E1812C', '#3A923A']
    bars = ax1.bar(categories, values, color=bar_colors)
    
    # 값 표시 (수치가 크기 때문에 로그 스케일 사용)
    ax1.set_yscale('log')
    
    # 각 막대 위에 정확한 값 표시
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height):,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Dataset Size Comparison', fontsize=18, fontweight='bold')
    plt.ylabel('Number of Images (log scale)', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualization_results/dataset_size_comparison.png', dpi=300)
    plt.close()
    
    # 2. 작성자 수 비교 (막대 그래프)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    categories = ['Original Dataset', 'Line Segments', 'Final Dataset']
    values = [original_authors, line_segment_authors, final_authors]
    
    bar_colors = ['#3274A1', '#E1812C', '#3A923A']
    bars = ax2.bar(categories, values, color=bar_colors)
    
    # 각 막대 위에 정확한 값 표시
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Number of Writers Comparison', fontsize=18, fontweight='bold')
    plt.ylabel('Number of Writers', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualization_results/writer_count_comparison.png', dpi=300)
    plt.close()
    
    # 3. 데이터 분할 비교 (원본 vs 최종) - 그룹화된 막대 그래프
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    x = np.arange(3)  # 3개 카테고리 (훈련, 검증, 테스트)
    width = 0.35
    
    # 원본 데이터와 최종 데이터셋의 백분율 계산
    orig_total = original_train + original_val + original_test
    orig_train_pct = original_train / orig_total * 100
    orig_val_pct = original_val / orig_total * 100
    orig_test_pct = original_test / orig_total * 100
    
    final_total = final_train + final_val + final_test
    final_train_pct = final_train / final_total * 100
    final_val_pct = final_val / final_total * 100
    final_test_pct = final_test / final_total * 100
    
    # 막대 그래프 생성
    orig_bars = ax3.bar(x - width/2, [orig_train_pct, orig_val_pct, orig_test_pct], 
                        width, label='Original Dataset', color='#3274A1')
    final_bars = ax3.bar(x + width/2, [final_train_pct, final_val_pct, final_test_pct], 
                        width, label='Final Dataset', color='#E1812C')
    
    # 각 막대 위에 정확한 값 표시
    for i, bars in enumerate([orig_bars, final_bars]):
        label = "Original" if i == 0 else "Final"
        values = [original_train, original_val, original_test] if i == 0 else [final_train, final_val, final_test]
        percentages = [orig_train_pct, orig_val_pct, orig_test_pct] if i == 0 else [final_train_pct, final_val_pct, final_test_pct]
        
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{values[j]:,}\n({percentages[j]:.1f}%)', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Training', 'Validation', 'Test'], fontsize=14, fontweight='bold')
    ax3.legend(fontsize=14, frameon=True)
    
    plt.title('Dataset Split Comparison (Original vs Final)', fontsize=18, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualization_results/dataset_split_comparison.png', dpi=300)
    plt.close()
    
    # 4. 데이터셋 생성 과정 (개략적인 흐름도) - 막대 그래프와 화살표
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    
    stages = ['Original\nDataset', 'Line\nSegments', 'Final\nDataset']
    counts = [original_images, line_segments, final_dataset]
    
    bars = ax4.bar(stages, counts, color=['#3274A1', '#E1812C', '#3A923A'], width=0.6)
    
    # 각 막대 위에 정확한 값과 감소율 표시
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i == 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height + 200,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            prev_height = bars[i-1].get_height()
            reduction = (1 - height/prev_height) * 100
            ax4.text(bar.get_x() + bar.get_width()/2., height + 200,
                    f'{int(height):,}\n(-{reduction:.1f}%)', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 화살표 추가
    arrow_props = dict(arrowstyle='->', linewidth=2, color='gray')
    ax4.annotate('', xy=(0.9, counts[1]/2), xytext=(0.1, counts[1]/2), arrowprops=arrow_props)
    ax4.annotate('', xy=(1.9, counts[2]/2), xytext=(1.1, counts[2]/2), arrowprops=arrow_props)
    
    # 화살표 레이블 추가
    ax4.text(0.5, counts[1]/2 + 500, 'Line Segmentation\n& Selection', ha='center', va='center', fontsize=12, fontweight='bold')
    ax4.text(1.5, counts[2]/2 + 500, 'Dataset Creation\n(Single & Multi Author)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.title('Dataset Creation Process', fontsize=18, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualization_results/dataset_creation_process.png', dpi=300)
    plt.close()
    
    print("Comparative visualization completed! Results saved in 'visualization_results' folder.")


if __name__ == "__main__":
    create_dataset_visualizations('mil_dataset')
    create_comparative_visualizations() 