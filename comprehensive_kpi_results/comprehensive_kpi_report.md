# 포괄적 KPI 분석 보고서

**분석 일시**: 2025-09-07 17:16:01

## 📊 측정된 KPI 카테고리

### 1. 클러스터링 품질 지표
- **Silhouette Score**: 클러스터 내 응집도와 클러스터 간 분리도
- **Calinski-Harabasz Score**: 클러스터 간 분산 대 클러스터 내 분산 비율
- **Davies-Bouldin Score**: 클러스터 내 거리 대 클러스터 간 거리 (낮을수록 좋음)
- **Adjusted Rand Score**: 실제 라벨과의 일치도
- **Normalized Mutual Information**: 정보 이론 기반 클러스터링 품질

### 2. 처리 효율성 지표
- **Total Processing Time**: 전체 처리 시간
- **Flag Generation Time**: 플래그 생성 시간
- **Clustering Time**: 클러스터링 수행 시간
- **Processing Speed**: 초당 처리 문서 수
- **Throughput**: 분당 처리량

### 3. 메모리 및 리소스 지표
- **Memory Usage**: 메모리 사용량 (MB)
- **Peak Memory**: 최대 메모리 사용량
- **Memory per Document**: 문서당 메모리 사용량
- **CPU Usage**: CPU 사용률

### 4. 확장성 지표
- **Time Complexity per Doc**: 문서당 처리 시간
- **Estimated Time for 1K/10K Docs**: 대용량 처리 예상 시간
- **Feature Matrix Size**: 특성 행렬 크기
- **Scalability Readiness**: 확장성 준비도

### 5. 안정성 지표
- **Flag Variance/Stability**: 플래그 값의 안정성
- **Feature Stability**: 특성별 안정성
- **Cluster Stability**: 클러스터 분포의 안정성
- **Outlier Ratio**: 이상치 비율

### 6. 실용성 지표
- **Implementation Complexity**: 구현 복잡도
- **Weight Complexity**: 가중치 복잡도
- **Flag Density/Sparsity**: 플래그 밀도
- **Effective Dimension Ratio**: 유효 차원 비율

### 7. 비즈니스 지표
- **Cost per Document**: 문서당 처리 비용
- **Productivity Score**: 생산성 점수
- **Time Saving Ratio**: 시간 절약 비율
- **Automation Benefit Score**: 자동화 효과 점수

## 🎯 주요 결과 요약

**최고 클러스터링 품질**: best_2flags (Silhouette: 0.7157)

**가장 빠른 처리**: stable_2flags (12.0201초)

**가장 메모리 효율적**: baseline_tfidf (5.12MB)

## 📋 상세 KPI 데이터

### 주요 성능 지표 비교

