1) 헬퍼: 슬라이딩/혼합 유틸 (추가)

# --- Sliding window helper ---
def sliding_windows(seq, win=5, stride=1):
    # seq: list of tuples (embedding, writer_id, path, orig_idx)
    out = []
    metas = []
    for i in range(0, len(seq) - win + 1, stride):
        window = seq[i:i+win]
        out.append(np.stack([emb for emb, _, _, _ in window]))
        metas.append({
            'window_idx': i,
            'word_indices': [orig for _, _, _, orig in window],
            'word_paths':   [pth  for _, _, pth,  _   in window],
            'writer_ids':   [wid  for _, wid, _,  _   in window],  # 혼합이면 두 writer 섞여 들어감
        })
    return out, metas

# --- Calc: 혼합 윈도우 개수 ---
def count_mixed_windows(instance_metas):
    return sum(1 for m in instance_metas if len(set(m['writer_ids'])) > 1)

# --- Build word-level mixed sequence with boundary ---
def build_word_mix_sequence(A_pack, B_pack, min_mixed=3, max_tries=10):
    """
    A_pack/B_pack: list of tuples (embedding, writer_id, path, orig_idx), 길이 동일(예: 7)
    14개 시퀀스를 만들되 경계 b를 3~10에서 뽑아 A[:b] + B[:(14-b)]을 기본으로 하고,
    남은 토큰은 A/B에서 섞어 넣어 '혼합 윈도우 수'가 min_mixed 이상 되도록 보장.
    """
    assert len(A_pack) == len(B_pack)
    L = len(A_pack) + len(B_pack)
    for _ in range(max_tries):
        b = random.randint(3, L-4)  # 3~10 → 혼합 윈도우가 충분히 생기도록
        seq = A_pack[:b] + B_pack[:(L-b)]
        # 남은 토큰(있다면) 무작위 삽입
        restA = A_pack[b:] if b < len(A_pack) else []
        restB = B_pack[(L-b):] if (L-b) < len(B_pack) else []
        rest = restA + restB
        random.shuffle(rest)
        for r in rest:
            pos = random.randint(0, len(seq))
            seq.insert(pos, r)
            if len(seq) > L:
                seq = seq[:L]
                break
        # 임시 인스턴스 구성해 혼합 개수 검사
        win, metas = sliding_windows(seq, win=5, stride=1)
        if count_mixed_windows(metas) >= min_mixed and len(win) >= 10:
            return seq
    # 보장 실패 시 마지막 시퀀스 반환(최소한 경계로 일부 혼합은 생김)
    return seq


2) 파트너 샘플링: 작성자‑노출 균형화 & 페어 반복 상한 (추가)

from collections import defaultdict

class PairScheduler:
    def __init__(self, eligible_writers, max_pair_repeat=2):
        self.pos_particip = defaultdict(int)  # Positive 참여 횟수(= A든 B든)
        self.pair_count   = defaultdict(int)  # (min(a,b), max(a,b)) 페어 반복 수
        self.max_pair_repeat = max_pair_repeat
        self.eligible = set(eligible_writers)

    def choose_partner(self, A):
        candidates = [w for w in self.eligible if w != A and self.pair_count[(min(A,w), max(A,w))] < self.max_pair_repeat]
        if not candidates:
            candidates = [w for w in self.eligible if w != A]  # 백업
        # Positive 참여가 가장 적은 작성자 우선(동률이면 랜덤)
        candidates.sort(key=lambda w: (self.pos_particip[w], random.random()))
        B = candidates[0]
        # 카운트 갱신은 바깥에서 한 번에
        return B

    def update_counts(self, A, B):
        self.pos_particip[A] += 1
        self.pos_particip[B] += 1
        self.pair_count[(min(A,B), max(A,B))] += 1


3) generate_bags_for_split 교체본 (하이브리드 + 균형화 + 중첩완화)
기본값: pos_mode_probs = {'word_mix': 0.7, 'instance_mix': 0.3}

word_mix: A 7개 + B 7개 → 경계 기반 시퀀스 → stride=1 슬라이딩(10개 인스턴스). 혼합윈도 최소 K 보장

instance_mix: A 13개, B 13개 → stride=2로 각 5개 인스턴스(중첩 감소) → 총 10개 인스턴스 → 최종 셔플

def pack_words(emb_arr, paths, idxs, sel_idx, writer_id):
    return [(emb_arr[i], int(writer_id), (paths[i] if paths else f"unknown_{i}"), (idxs[i] if idxs else i)) for i in sel_idx]

def generate_bags_for_split_v2(
    writer_ids, embedding_dict, desired_neg_per_writer=20, 
    embed_dim=256, pos_mode_probs={'word_mix':0.7, 'instance_mix':0.3},
    min_mixed_windows=3, max_pair_repeat=2
):
    bags, labels, metadata = [], [], []
    # Positive에 참여 가능한 작성자(충분 토큰)
    eligible_wordmix = [w for w in writer_ids if len(embedding_dict[w]['embeddings']) >= 7]
    eligible_instmix = [w for w in writer_ids if len(embedding_dict[w]['embeddings']) >= 13]
    scheduler = PairScheduler(eligible_wordmix, max_pair_repeat=max_pair_repeat)

    for A in writer_ids:
        dataA = embedding_dict[A]; embA, pathsA, idxA = dataA['embeddings'], dataA['paths'], dataA['indices']
        # --- Negative bags for A ---
        max_neg = len(embA) // 14
        n_neg = min(max_neg, desired_neg_per_writer)
        for _ in range(n_neg):
            sel = random.sample(range(len(embA)), 14)
            packA = pack_words(embA, pathsA, idxA, sel, A)
            # stride=1, 10 windows
            instances, inst_meta = sliding_windows(packA, win=5, stride=1)
            bag_tensor = np.stack(instances)[:10]
            bags.append(bag_tensor)
            labels.append(0)
            metadata.append({'authors':[int(A)], 'bag_type':'negative', 'instances':inst_meta[:10]})

            # --- Positive bag paired with A ---
            # 모드 선택(확률)
            mode = 'word_mix' if random.random() < pos_mode_probs.get('word_mix', 1.0) else 'instance_mix'
            # 파트너 B 선택(균형화/중복 상한)
            # 모드별 eligible이 다르면 보정
            Bcand_pool = eligible_wordmix if mode=='word_mix' else eligible_instmix
            if A not in Bcand_pool:
                # A가 그 모드에 불충분하면 모드 강제로 변경
                mode = 'word_mix' if A in eligible_wordmix else 'instance_mix'
                Bcand_pool = eligible_wordmix if mode=='word_mix' else eligible_instmix

            # 스케줄러의 후보 집합을 해당 모드 eligible로 제한
            scheduler.eligible = set(Bcand_pool)
            B = scheduler.choose_partner(A)
            dataB = embedding_dict[B]; embB, pathsB, idxB = dataB['embeddings'], dataB['paths'], dataB['indices']

            if mode == 'word_mix':
                # A 7, B 7
                idxsA = random.sample(range(len(embA)), 7)
                idxsB = random.sample(range(len(embB)), 7)
                packA7 = pack_words(embA, pathsA, idxA, idxsA, A)
                packB7 = pack_words(embB, pathsB, idxB, idxsB, B)
                seq = build_word_mix_sequence(packA7, packB7, min_mixed=min_mixed_windows)
                instances, inst_meta = sliding_windows(seq, win=5, stride=1)   # 10개 생성
                bag_tensor = np.stack(instances)[:10]
                meta_inst = inst_meta[:10]

            else:  # instance_mix with low overlap
                # A 13, B 13 → stride=2로 각 5개 윈도우
                idxsA = random.sample(range(len(embA)), 13)
                idxsB = random.sample(range(len(embB)), 13)
                packA13 = pack_words(embA, pathsA, idxA, idxsA, A)
                packB13 = pack_words(embB, pathsB, idxB, idxsB, B)
                winA, metaA = sliding_windows(packA13, win=5, stride=2)  # 5개
                winB, metaB = sliding_windows(packB13, win=5, stride=2)  # 5개
                # 인스턴스 레벨 셔플
                mixed = list(zip(winA+winB, metaA+metaB))
                random.shuffle(mixed)
                bag_tensor = np.stack([w for w, _ in mixed])
                meta_inst  = [m for _, m in mixed]

            # 혼합 윈도우 최소 K 보장(only word_mix). instance_mix는 0이 정상
            if mode == 'word_mix' and count_mixed_windows(meta_inst) < min_mixed_windows:
                # 재시도 한 번 더
                seq = build_word_mix_sequence(packA7, packB7, min_mixed=min_mixed_windows)
                instances, inst_meta = sliding_windows(seq, win=5, stride=1)
                bag_tensor = np.stack(instances)[:10]
                meta_inst  = inst_meta[:10]

            bags.append(bag_tensor)
            labels.append(1)
            metadata.append({'authors':[int(A), int(B)], 'bag_type':'positive', 'instances':meta_inst})
            scheduler.update_counts(A, B)

    return bags, labels, metadata

바로 치환 포인트
기존 generate_bags_for_split 함수만 위 generate_bags_for_split_v2로 교체하고, 호출부에서 함수명만 바꿔주면 됩니다. 파일명/포맷은 동일하게 유지하세요(Stage 3 호환).

4) 품질 검증(업데이트): 혼합 윈도우 개수 통계 추가

def summarize_split_plus(name, bags, labels, metadata):
    total = len(bags)
    n_neg = labels.count(0)
    n_pos = labels.count(1)
    print(f"{name}: total={total}, neg={n_neg} ({n_neg/total*100:.1f}%), pos={n_pos} ({n_pos/total*100:.1f}%)")
    # Positive 혼합 윈도우 통계
    mix_counts = []
    for lab, meta in zip(labels, metadata):
        if lab == 1:
            mix_counts.append(count_mixed_windows(meta['instances']))
    if mix_counts:
        print(f"  Positive mixed windows per bag: mean={np.mean(mix_counts):.2f}, "
              f"median={np.median(mix_counts):.0f}, min={np.min(mix_counts)}, max={np.max(mix_counts)}")
        print(f"  % with >={3} mixed windows: {(np.mean(np.array(mix_counts) >= 3)*100):.1f}%")

결과가 min mixed < K로 많이 나오면 min_mixed_windows를 4로 올리거나, 경계 b 분포(예: 4~9)로 더 중앙에 몰리게 조정.


6) Stage 3에 건드리면 좋은 한 줄
Stage 3는 그대로 둬도 되지만, “분포적 신호”도 약간 보태려면 bag‑통계 피처를 분류기에 concat하세요(간단 튜닝):

# x: (B, 10, 256) -> h: (B, 10, H)
mu  = h.mean(dim=1)              # (B, H)
var = h.var(dim=1)               # (B, H)
dists = torch.cdist(h, h).mean(dim=(1,2), keepdim=False)  # (B,)
z = torch.cat([mu, var, dists.unsqueeze(-1)], dim=-1)     # (B, 2H+1)
logits = self.classifier2(z)     # 작은 MLP 1~2층
