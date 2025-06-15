# Data2.py test 버전


# 보완 및 개선
# 인스턴스 수준 어노테이션 강화: 현재 마스크는 작성자별 클래스 구분만 제공한다. 만약 한 작성자가 한 문서 내 여러 라인을 작성했다면, 이 모든 라인이 동일한 ID로 묶인다. 경우에 따라 라인 단위 인스턴스 식별이 필요할 수 있으므로, 마스크나 메타데이터에 각 라인 객체를 구분하는 식별자를 추가하는 것을 고려할 수 있다. 예를 들어 COCO Panoptic segmentation처럼 각 연결된 글자 영역을 별도의 인스턴스로 할당하고, 그 인스턴스에 작성자 ID 속성을 부여하는 구조를 생각할 수 있다. 이렇게 하면 한 문서 내 동일 작성자의 여러 필기 조각도 각각 개별 객체로 취급되어, MIL 모델에서 인스턴스별 특징 맵핑이나, 객체 탐지/분할 모델 학습에도 활용할 수 있다.
# 단일 작성자 문서 데이터 추가: 현재 구현은 두 명의 필체를 합성하는데 초점을 맞추고 있지만, 연구의 대조군으로서 한 명의 필체로만 이루어진 문서들도 함께 생성하여 활용하는 것을 제안한다. 이를 통해 모델 학습 시 single-author vs multi-author를 모두 경험하게 할 수 있고, 위조 탐지 모델의 경우 **음성 샘플(진짜 문서)**을 제공하여 분류 경계를 더 명확히 학습시킬 수 있다. 구현 측면에서도, 두 작성자를 고르는 대신 하나의 작성자 라인들만 이어 붙이는 옵션을 추가하면 쉽게 실현 가능하다.


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


import os
import cv2
import numpy as np
import random
import pandas as pd
import json
import datetime
import traceback
# from skimage.exposure import match_histograms # 필요시 주석 해제

# --- Configuration ---
BASE_INPUT_DIR = "/home/dk926/workspace"
HANDWRITING_DIR = os.path.join(BASE_INPUT_DIR, "MIL/line_segments")
OUTPUT_DIR = './synth_output_mil_mixed/' # 출력 디렉토리 변경 (단일/복수 혼합)
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs/')
METADATA_PATH = './writer_metadata.csv' # 필요시 경로 수정

# Generation Parameters - Multi-Author (Pairs)
NUM_DOCS_PER_PAIR = 5       # 작성자 쌍당 생성할 문서 수
MIN_LINES_PER_AUTHOR_PAIR = 1    # 쌍 문서에서 저자별 최소 라인 수
MAX_LINES_PER_AUTHOR_PAIR = 3    # 쌍 문서에서 저자별 최대 라인 수

# Generation Parameters - Single-Author
GENERATE_SINGLE_AUTHOR_DOCS = True # 단일 작성자 문서 생성 여부
NUM_SINGLE_AUTHOR_DOCS_PER_WRITER = 3 # 작성자당 생성할 단일 문서 수
MIN_LINES_SINGLE_AUTHOR = 1        # 단일 문서 최소 라인 수
MAX_LINES_SINGLE_AUTHOR = 5        # 단일 문서 최대 라인 수

# Common Parameters
DOC_WIDTH = 1200
LINE_SPACING = 30
BACKGROUND_ID = 0

# --- Utility Functions ---

def extract_filename(path):
    return os.path.basename(path)

def extract_writer_id(filename):
    try:
        return filename.split('_')[0]
    except IndexError:
        print(f"Warning: Could not extract writer ID from filename: {filename}")
        return None

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def create_log_file(result_info, filename, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    try:
        # Update log path within the info itself before saving
        result_info["output_log_file"] = log_path
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(result_info, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        return log_path
    except Exception as e:
        print(f"Error writing log file {log_path}: {e}")
        # Attempt to remove potentially corrupted log file
        if os.path.exists(log_path):
            try:
                os.remove(log_path)
            except OSError:
                pass
        return None

# --- Core Synthesis Functions ---

def load_handwriting_data(handwriting_dir):
    handwriting_files = [os.path.join(handwriting_dir, f) for f in os.listdir(handwriting_dir) if f.endswith('.png')]
    print(f"Found {len(handwriting_files)} total handwriting line files.")

    writers = {}
    for f_path in handwriting_files:
        filename = extract_filename(f_path)
        writer_id = extract_writer_id(filename)
        if writer_id:
            if writer_id not in writers: writers[writer_id] = []
            writers[writer_id].append(f_path)

    print(f"Grouped lines for {len(writers)} unique writers.")
    # Filter based on the *minimum* lines needed for either single or pair generation
    min_lines_needed = min(MIN_LINES_SINGLE_AUTHOR if GENERATE_SINGLE_AUTHOR_DOCS else 999, MIN_LINES_PER_AUTHOR_PAIR)
    original_writer_count = len(writers)
    writers = {wid: paths for wid, paths in writers.items() if len(paths) >= min_lines_needed}
    print(f"Keeping {len(writers)} writers out of {original_writer_count} with at least {min_lines_needed} lines.")

    return writers

def create_multi_doc_with_mask(lines_list, doc_width, line_spacing):
    """
    Creates a synthetic document image and segmentation mask from a list of lines.
    Now includes a unique ID for each line instance within the document metadata.
    """
    if not lines_list:
        print("Warning: create_multi_doc_with_mask called with empty lines_list.")
        return None, None, None

    final_img = np.full((0, doc_width, 3), 255, dtype=np.uint8)
    segmentation_mask = np.full((0, doc_width), BACKGROUND_ID, dtype=np.int32)
    instances = []
    writer_id_map = {}
    next_int_id = 1
    current_y = 0
    loaded_data = []
    present_str_ids = set()

    for line_path in lines_list:
        src_line = cv2.imread(line_path)
        if src_line is None: continue # Skip unreadable files silently here

        filename = extract_filename(line_path)
        author_str_id = extract_writer_id(filename)
        if not author_str_id: continue # Skip files without valid ID

        if author_str_id not in writer_id_map:
            writer_id_map[author_str_id] = next_int_id
            next_int_id += 1
        author_int_id = writer_id_map[author_str_id]
        present_str_ids.add(author_str_id)

        loaded_data.append({
            "path": line_path, "img": src_line, "author_str_id": author_str_id,
            "author_int_id": author_int_id, "original_h": src_line.shape[0],
            "original_w": src_line.shape[1]
        })

    if not loaded_data:
        print("Error: No valid images could be loaded for this document.")
        return None, None, None

    # --- Process and Stack Images ---
    for i, data in enumerate(loaded_data): # 'i' will be our instance_doc_id
        img = data["img"]
        author_int_id = data["author_int_id"]

        scale = doc_width / img.shape[1]
        new_height = max(1, int(img.shape[0] * scale)) # Ensure height is at least 1
        resized = cv2.resize(img, (doc_width, new_height), interpolation=cv2.INTER_AREA)

        gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        current_segment_mask = np.full((new_height, doc_width), BACKGROUND_ID, dtype=np.int32)
        current_segment_mask[binary_mask == 255] = author_int_id

        if final_img.shape[0] > 0:
            space_img = np.full((line_spacing, doc_width, 3), 255, dtype=np.uint8)
            space_mask = np.full((line_spacing, doc_width), BACKGROUND_ID, dtype=np.int32)
            final_img = np.concatenate((final_img, space_img), axis=0)
            segmentation_mask = np.concatenate((segmentation_mask, space_mask), axis=0)
            current_y += line_spacing

        bbox_y = current_y
        bbox = [0, bbox_y, doc_width, new_height]

        final_img = np.concatenate((final_img, resized), axis=0)
        segmentation_mask = np.concatenate((segmentation_mask, current_segment_mask), axis=0)

        # **** ENHANCEMENT: Added instance_doc_id ****
        instances.append({
            "instance_doc_id": i, # Unique ID for this line *within this document*
            "author_id_str": data["author_str_id"],
            "author_id_int": author_int_id,
            "source_file": data["path"],
            "bbox": bbox,
            "original_height": data["original_h"], "original_width": data["original_w"],
            "resized_height": new_height, "resized_width": doc_width,
            "scaling_factor": scale
        })
        current_y += new_height

    is_multi_author_flag = len(present_str_ids) > 1
    result_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "generation_method": "create_multi_doc_with_mask",
        "parameters": {"doc_width": doc_width, "line_spacing": line_spacing, "background_id": BACKGROUND_ID},
        "source_lines_used": [d["path"] for d in loaded_data],
        "bag_labels": {
            "authors_present_str": sorted(list(present_str_ids)),
            "is_multi_author": is_multi_author_flag,
            "num_authors": len(present_str_ids) # Explicitly add number of authors
        },
        "instance_labels": instances,
        "writer_id_map": writer_id_map,
        "output_image_file": None, "output_mask_file": None, "output_log_file": None,
        "author_metadata": None
    }

    return final_img, segmentation_mask, result_info

# --- Metadata Handling ---
def load_metadata(metadata_path):
    metadata_df = None
    try:
        metadata_df = pd.read_csv(metadata_path)
        if 'writer_id' in metadata_df.columns:
            # Check for duplicates before setting index
            if metadata_df['writer_id'].duplicated().any():
                 print(f"Warning: Duplicate writer_ids found in {metadata_path}. Using the first occurrence.")
                 metadata_df = metadata_df.drop_duplicates(subset=['writer_id'], keep='first')
            metadata_df.set_index('writer_id', inplace=True)
            print(f"Successfully loaded metadata for {len(metadata_df)} unique writers from {metadata_path}")
        else:
             print(f"Warning: 'writer_id' column not found in {metadata_path}. Metadata will not be linked.")
             metadata_df = None
    except FileNotFoundError:
        print(f"Info: Metadata file not found at {metadata_path}. Proceeding without metadata.")
    except Exception as e:
        print(f"Warning: Error loading or processing metadata file {metadata_path}: {e}")
        metadata_df = None
    return metadata_df

def add_metadata_to_info(info_dict, metadata_df):
    if metadata_df is None or 'bag_labels' not in info_dict or 'authors_present_str' not in info_dict['bag_labels']:
        return # No metadata or no author info to link

    author_ids = info_dict['bag_labels']['authors_present_str']
    doc_metadata = {}
    for author_id in author_ids:
        try:
            if author_id in metadata_df.index:
                # Convert Series to dict, handle potential NaNs safely
                author_meta = metadata_df.loc[author_id].where(pd.notnull(metadata_df.loc[author_id]), None).to_dict()
                doc_metadata[author_id] = author_meta
            else:
                doc_metadata[author_id] = {} # Author not found in metadata
        except Exception as meta_e:
            print(f"  Warning: Error accessing metadata for {author_id}: {meta_e}")
            doc_metadata[author_id] = {"error": str(meta_e)}
    info_dict["author_metadata"] = doc_metadata


# --- Generation Orchestration ---

def generate_documents(writers_data, output_dir, log_dir, metadata_df=None):
    """
    Generates both single-author and two-author synthetic documents.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    generation_count = 0
    writer_ids = list(writers_data.keys())

    # --- 1. Generate Two-Author Documents ---
    if len(writer_ids) >= 2:
        author_pairs = []
        for i in range(len(writer_ids)):
            for j in range(i + 1, len(writer_ids)):
                author_pairs.append((writer_ids[i], writer_ids[j]))

        print(f"\n--- Starting Two-Author Document Generation ({len(author_pairs)} pairs) ---")
        for pair_idx, (w1_id, w2_id) in enumerate(author_pairs):
            # Basic check if enough lines exist (more robust check inside loop)
            if len(writers_data.get(w1_id, [])) < MIN_LINES_PER_AUTHOR_PAIR or \
               len(writers_data.get(w2_id, [])) < MIN_LINES_PER_AUTHOR_PAIR:
                continue # Skip pair if statically known to be insufficient

            print(f"Processing Pair {pair_idx+1}/{len(author_pairs)}: ({w1_id}, {w2_id})")
            docs_for_this_pair = 0
            for i in range(NUM_DOCS_PER_PAIR):
                try:
                    w1_lines = writers_data[w1_id]
                    w2_lines = writers_data[w2_id]
                    if len(w1_lines) < MIN_LINES_PER_AUTHOR_PAIR or len(w2_lines) < MIN_LINES_PER_AUTHOR_PAIR:
                         # This check prevents sampling errors if MAX > available
                         print(f"  Skipping doc {i+1} for pair {w1_id}-{w2_id}, insufficient lines.")
                         break # Stop generating for this pair if lines run out

                    num_w1 = random.randint(MIN_LINES_PER_AUTHOR_PAIR, min(MAX_LINES_PER_AUTHOR_PAIR, len(w1_lines)))
                    num_w2 = random.randint(MIN_LINES_PER_AUTHOR_PAIR, min(MAX_LINES_PER_AUTHOR_PAIR, len(w2_lines)))
                    selected_lines_w1 = random.sample(w1_lines, num_w1)
                    selected_lines_w2 = random.sample(w2_lines, num_w2)
                    lines_for_doc = selected_lines_w1 + selected_lines_w2
                    random.shuffle(lines_for_doc)

                    multi_img, mask, info = create_multi_doc_with_mask(lines_for_doc, DOC_WIDTH, LINE_SPACING)

                    if multi_img is None: continue

                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                    base_filename = f"synth_{w1_id}_{w2_id}_pair_{i}_{timestamp}"
                    img_filename = f"{base_filename}_img.png"
                    mask_filename = f"{base_filename}_mask.png"
                    log_filename = f"{base_filename}_log.json"

                    img_path = os.path.join(output_dir, img_filename)
                    mask_path = os.path.join(output_dir, mask_filename)

                    cv2.imwrite(img_path, multi_img)
                    if np.max(mask) < 256: cv2.imwrite(mask_path, mask.astype(np.uint8))
                    else: cv2.imwrite(mask_path, mask.astype(np.uint16))

                    info["output_image_file"] = img_path
                    info["output_mask_file"] = mask_path
                    add_metadata_to_info(info, metadata_df) # Add metadata before saving log

                    if create_log_file(info, log_filename, log_dir):
                        generation_count += 1
                        docs_for_this_pair += 1
                    else: # Log failed, try to clean up image/mask
                         print(f"  Log saving failed for {base_filename}. Removing associated image/mask.")
                         if os.path.exists(img_path): os.remove(img_path)
                         if os.path.exists(mask_path): os.remove(mask_path)

                except Exception as e:
                    print(f"  Error generating pair document {i+1} for ({w1_id}, {w2_id}): {e}")
                    traceback.print_exc()
            if docs_for_this_pair > 0:
                 print(f"  Generated {docs_for_this_pair} documents for pair ({w1_id}, {w2_id}).")

    else:
        print("Warning: Less than 2 writers available, skipping two-author document generation.")


    # --- 2. Generate Single-Author Documents ---
    if GENERATE_SINGLE_AUTHOR_DOCS:
        print(f"\n--- Starting Single-Author Document Generation ---")
        if len(writer_ids) == 0:
             print("No writers available for single-author document generation.")
        else:
            for writer_idx, writer_id in enumerate(writer_ids):
                lines = writers_data.get(writer_id, [])
                if len(lines) < MIN_LINES_SINGLE_AUTHOR:
                    print(f"Skipping writer {writer_id} (index {writer_idx+1}/{len(writer_ids)}): Insufficient lines ({len(lines)} < {MIN_LINES_SINGLE_AUTHOR}).")
                    continue

                print(f"Processing Writer {writer_idx+1}/{len(writer_ids)}: {writer_id}")
                docs_for_this_writer = 0
                for i in range(NUM_SINGLE_AUTHOR_DOCS_PER_WRITER):
                    try:
                        num_lines = random.randint(MIN_LINES_SINGLE_AUTHOR, min(MAX_LINES_SINGLE_AUTHOR, len(lines)))
                        # Ensure we don't try to sample more lines than available
                        if num_lines > len(lines): num_lines = len(lines)
                        if num_lines < MIN_LINES_SINGLE_AUTHOR:
                             print(f"  Cannot sample {MIN_LINES_SINGLE_AUTHOR} lines from available {len(lines)} for writer {writer_id}. Stopping for this writer.")
                             break # Not enough lines even for min requirement

                        selected_lines = random.sample(lines, num_lines)
                        # No shuffle needed for single author, order is less critical

                        single_img, mask, info = create_multi_doc_with_mask(selected_lines, DOC_WIDTH, LINE_SPACING)

                        if single_img is None: continue

                        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                        base_filename = f"synth_{writer_id}_single_{i}_{timestamp}"
                        img_filename = f"{base_filename}_img.png"
                        mask_filename = f"{base_filename}_mask.png"
                        log_filename = f"{base_filename}_log.json"

                        img_path = os.path.join(output_dir, img_filename)
                        mask_path = os.path.join(output_dir, mask_filename)

                        cv2.imwrite(img_path, single_img)
                        if np.max(mask) < 256: cv2.imwrite(mask_path, mask.astype(np.uint8))
                        else: cv2.imwrite(mask_path, mask.astype(np.uint16))

                        info["output_image_file"] = img_path
                        info["output_mask_file"] = mask_path
                        add_metadata_to_info(info, metadata_df) # Add metadata

                        if create_log_file(info, log_filename, log_dir):
                            generation_count += 1
                            docs_for_this_writer +=1
                        else: # Log failed
                            print(f"  Log saving failed for {base_filename}. Removing associated image/mask.")
                            if os.path.exists(img_path): os.remove(img_path)
                            if os.path.exists(mask_path): os.remove(mask_path)

                    except Exception as e:
                        print(f"  Error generating single-author document {i+1} for {writer_id}: {e}")
                        traceback.print_exc()
                if docs_for_this_writer > 0:
                     print(f"  Generated {docs_for_this_writer} documents for writer {writer_id}.")
    else:
        print("\nSkipping single-author document generation as per configuration.")


    print(f"\n--- Generation Complete ---")
    print(f"Total documents generated (single + pairs): {generation_count}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting synthetic document generation for MIL...")

    # 1. Load Handwriting Data
    writers_data = load_handwriting_data(HANDWRITING_DIR)

    # 2. Load Metadata (Optional)
    metadata_df = load_metadata(METADATA_PATH)

    # 3. Generate Documents (Both Single and Pairs, if configured and possible)
    if writers_data: # Proceed only if some writers met the minimum line requirements
        generate_documents(
            writers_data,
            output_dir=OUTPUT_DIR,
            log_dir=LOG_DIR,
            metadata_df=metadata_df
        )
    else:
        print("No writers with sufficient data found. No documents generated.")

    print("Script finished.")