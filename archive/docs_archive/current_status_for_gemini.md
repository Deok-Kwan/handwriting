# MIL 프로젝트 현황 정리 (Gemini PPT 작업용)

## 작업 요청 사항
Gemini 공식 LLM을 사용하여 랩미팅 발표 PPT를 제작하려고 합니다. 아래 정보를 참고하여 PPT를 만들어주세요.

## 프로젝트 개요
- **프로젝트명**: Multiple Instance Learning (MIL) 기반 다중 작성자 문서 탐지
- **목표**: 문서가 단일 작성자 또는 복수 작성자로 작성되었는지 AI로 판별
- **현재 단계**: Siamese Network 실험 중 임베딩 붕괴 문제 해결 중

## 연구 진행 경과

### 1. 이전 랩미팅 (Autoencoder 실험)
- **방법**: Autoencoder로 필기 특징 추출 (300차원 → 128차원)
- **결과**: 50.2% 정확도 (랜덤 수준)
- **결론**: 재구성 목적의 학습이 작성자 구분에 부적합

### 2. 이번 랩미팅 보고 내용 (Siamese Network 실험)

#### 실험 1: 기본 Siamese Network
- **방법**: Contrastive Learning으로 같은/다른 작성자 구분 학습
- **설정**: Margin=1.0, Learning rate=1e-3
- **결과**: 64.4% 정확도 (14.2%p 향상)
- **의미**: 올바른 방향이지만 추가 개선 필요

#### 실험 2: 개선된 Siamese Network
- **개선 시도**:
  - Margin 값 증가 (1.0 → 0.5)
  - Mixed Precision Training
  - 데이터셋 버그 수정
  - 최적 임계값 자동 탐색
- **결과**: 56.5% 정확도 (오히려 7.9%p 하락)
- **문제**: 임베딩 붕괴 현상 발생
  - 모든 샘플이 비슷한 임베딩으로 수렴
  - 거리가 모두 0에 가까운 값
  - NaN loss 발생

### 3. 향후 계획
- **단기 (1-2주)**:
  - Margin 값 재조정 (0.5 → 0.2)
  - Triplet Loss 도입
  - 학습률 안정화
  - 데이터 균형 맞추기
- **중장기**:
  - ArcFace 등 고급 metric learning 기법
  - 실제 법의학 문서 적용

## PPT 제작 가이드

### 필요한 슬라이드 (13장)
1. 제목
2. 지난 랩미팅 요약 (Autoencoder 실패)
3. Siamese Network 소개
4. 실험 설계
5. 첫 번째 Siamese 실험 결과 (64.4%)
6. 개선 전략
7. 개선된 Siamese 실험 결과 (56.5%)
8. 임베딩 붕괴 분석
9. 해결 방안
10. 전체 연구 로드맵
11. 핵심 통찰 및 교훈
12. 결론
13. Q&A

### 중요 포인트
- **스토리라인**: 문제 → 시도 → 부분 성공 → 개선 시도 → 실패 → 원인 분석 → 해결 방안
- **톤**: 실패를 숨기지 않고 학습 과정의 일부로 설명
- **시간**: 총 20분 발표

### 시각 자료가 필요한 부분
- Siamese Network 구조도
- 거리 분포 히스토그램 (실험 1 vs 실험 2)
- 임베딩 붕괴 t-SNE 시각화
- Triplet Loss 개념도
- 전체 연구 로드맵 타임라인

### 핵심 메시지
1. Autoencoder보다 Siamese Network가 적합한 접근법임을 확인
2. 임베딩 붕괴라는 기술적 문제에 직면했지만 원인을 파악함
3. 명확한 해결 방안을 가지고 있으며 체계적으로 진행 중
4. 실패도 중요한 학습 과정의 일부

## 참고 파일
- 전체 가이드: `/workspace/MIL/docs/lab_meeting_ppt_gemini_guide.md`
- 실험 결과: `/workspace/MIL/output/results/` 폴더
- 시각화 자료: `/workspace/MIL/output/figures/` 폴더

이 정보를 바탕으로 Gemini에서 PPT를 제작해주시면 됩니다.