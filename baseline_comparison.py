"""
베이스라인 알고리즘과의 성능 비교 실험

Dynamic Context Flag-Based Hierarchical Algorithm과 
기존 클러스터링 알고리즘들의 성능을 객관적으로 비교
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
import logging

from main import DynamicContextFlag, HierarchicalDocumentClustering, ContextLinkingAlgorithm
from data_loader import DatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineComparison:
    """베이스라인 알고리즘 비교 클래스"""
    
    def __init__(self):
        self.data_loader = DatasetLoader()
        self.results = {}
        
    def run_baseline_algorithms(self, documents, labels, dataset_name):
        """베이스라인 알고리즘들 실행"""
        logger.info(f"{dataset_name} 베이스라인 알고리즘 비교 시작")
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        baseline_results = {}
        
        # 1. K-Means 클러스터링
        logger.info("  K-Means 클러스터링...")
        start_time = time.time()
        try:
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(tfidf_matrix.toarray())
            
            baseline_results['kmeans'] = {
                'labels': kmeans_labels,
                'processing_time': time.time() - start_time,
                'silhouette_score': silhouette_score(tfidf_matrix.toarray(), kmeans_labels),
                'calinski_harabasz_score': calinski_harabasz_score(tfidf_matrix.toarray(), kmeans_labels),
                'davies_bouldin_score': davies_bouldin_score(tfidf_matrix.toarray(), kmeans_labels),
                'n_clusters': len(np.unique(kmeans_labels))
            }
            logger.info(f"    K-Means 완료: 실루엣 점수 = {baseline_results['kmeans']['silhouette_score']:.3f}")
        except Exception as e:
            logger.error(f"    K-Means 오류: {e}")
            baseline_results['kmeans'] = {'error': str(e)}
        
        # 2. Agglomerative Clustering
        logger.info("  Agglomerative 클러스터링...")
        start_time = time.time()
        try:
            agg_clustering = AgglomerativeClustering(n_clusters=5, linkage='ward')
            agg_labels = agg_clustering.fit_predict(tfidf_matrix.toarray())
            
            baseline_results['agglomerative'] = {
                'labels': agg_labels,
                'processing_time': time.time() - start_time,
                'silhouette_score': silhouette_score(tfidf_matrix.toarray(), agg_labels),
                'calinski_harabasz_score': calinski_harabasz_score(tfidf_matrix.toarray(), agg_labels),
                'davies_bouldin_score': davies_bouldin_score(tfidf_matrix.toarray(), agg_labels),
                'n_clusters': len(np.unique(agg_labels))
            }
            logger.info(f"    Agglomerative 완료: 실루엣 점수 = {baseline_results['agglomerative']['silhouette_score']:.3f}")
        except Exception as e:
            logger.error(f"    Agglomerative 오류: {e}")
            baseline_results['agglomerative'] = {'error': str(e)}
        
        # 3. DBSCAN
        logger.info("  DBSCAN 클러스터링...")
        start_time = time.time()
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
            dbscan_labels = dbscan.fit_predict(tfidf_matrix.toarray())
            
            if len(np.unique(dbscan_labels)) > 1:
                baseline_results['dbscan'] = {
                    'labels': dbscan_labels,
                    'processing_time': time.time() - start_time,
                    'silhouette_score': silhouette_score(tfidf_matrix.toarray(), dbscan_labels),
                    'calinski_harabasz_score': calinski_harabasz_score(tfidf_matrix.toarray(), dbscan_labels),
                    'davies_bouldin_score': davies_bouldin_score(tfidf_matrix.toarray(), dbscan_labels),
                    'n_clusters': len(np.unique(dbscan_labels[dbscan_labels != -1]))
                }
                logger.info(f"    DBSCAN 완료: 실루엣 점수 = {baseline_results['dbscan']['silhouette_score']:.3f}")
            else:
                baseline_results['dbscan'] = {'error': 'All points in single cluster or noise'}
                logger.warning("    DBSCAN: 모든 점이 단일 클러스터 또는 노이즈")
        except Exception as e:
            logger.error(f"    DBSCAN 오류: {e}")
            baseline_results['dbscan'] = {'error': str(e)}
        
        return baseline_results, tfidf_matrix
    
    def run_proposed_algorithm(self, documents, labels, dataset_name):
        """제안된 알고리즘 실행"""
        logger.info(f"  제안된 알고리즘 (Dynamic Context Flag)...")
        
        start_time = time.time()
        try:
            # Dynamic Context Flag 알고리즘
            context_flag_gen = DynamicContextFlag(flag_dimensions=10)
            context_flags = context_flag_gen.generate_context_flags(documents, labels)
            
            # K-Means로 클러스터링 (공정한 비교를 위해)
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            proposed_labels = kmeans.fit_predict(context_flags)
            
            proposed_results = {
                'labels': proposed_labels,
                'processing_time': time.time() - start_time,
                'silhouette_score': silhouette_score(context_flags, proposed_labels),
                'calinski_harabasz_score': calinski_harabasz_score(context_flags, proposed_labels),
                'davies_bouldin_score': davies_bouldin_score(context_flags, proposed_labels),
                'n_clusters': len(np.unique(proposed_labels)),
                'context_flags': context_flags
            }
            
            logger.info(f"    제안된 알고리즘 완료: 실루엣 점수 = {proposed_results['silhouette_score']:.3f}")
            return proposed_results
            
        except Exception as e:
            logger.error(f"    제안된 알고리즘 오류: {e}")
            return {'error': str(e)}
    
    def compare_all_algorithms(self):
        """모든 알고리즘 비교"""
        logger.info("전체 알고리즘 성능 비교 시작")
        
        datasets = [
            ('enron', 'Enron Email', 200),
            ('newsgroups', '20 Newsgroups', 200),
            ('reuters', 'Reuters-21578', 200)
        ]
        
        comparison_results = {}
        
        for dataset_name, display_name, limit in datasets:
            logger.info(f"\n{display_name} 데이터셋 비교...")
            
            try:
                # 데이터 로드
                if dataset_name == 'enron':
                    documents, labels = self.data_loader.load_enron_emails(limit)
                elif dataset_name == 'newsgroups':
                    documents, labels = self.data_loader.load_20newsgroups(limit)
                else:  # reuters
                    documents, labels = self.data_loader.load_reuters21578(limit)
                
                if not documents or len(documents) < 10:
                    logger.warning(f"{display_name}: 데이터 부족")
                    continue
                
                # 베이스라인 알고리즘들 실행
                baseline_results, tfidf_matrix = self.run_baseline_algorithms(documents, labels, dataset_name)
                
                # 제안된 알고리즘 실행
                proposed_results = self.run_proposed_algorithm(documents, labels, dataset_name)
                
                # 결과 저장
                comparison_results[dataset_name] = {
                    'display_name': display_name,
                    'document_count': len(documents),
                    'baseline_results': baseline_results,
                    'proposed_results': proposed_results,
                    'tfidf_shape': tfidf_matrix.shape
                }
                
            except Exception as e:
                logger.error(f"{display_name} 비교 오류: {e}")
                comparison_results[dataset_name] = {'error': str(e)}
        
        self.results = comparison_results
        return comparison_results
    
    def create_comparison_visualizations(self):
        """비교 시각화 생성"""
        logger.info("비교 시각화 생성 중...")
        
        if not self.results:
            logger.warning("비교 결과가 없습니다.")
            return
        
        # 메트릭별 비교 데이터 준비
        algorithms = ['kmeans', 'agglomerative', 'dbscan', 'proposed']
        algorithm_names = ['K-Means', 'Agglomerative', 'DBSCAN', 'Proposed (DCF)']
        datasets = []
        
        silhouette_data = []
        calinski_data = []
        davies_bouldin_data = []
        processing_time_data = []
        
        for dataset_name, result in self.results.items():
            if 'error' in result:
                continue
                
            datasets.append(result.get('display_name', dataset_name))
            
            # 각 알고리즘별 메트릭 수집
            sil_row = []
            cal_row = []
            db_row = []
            time_row = []
            
            # 베이스라인 알고리즘들
            for alg in ['kmeans', 'agglomerative', 'dbscan']:
                if alg in result['baseline_results'] and 'error' not in result['baseline_results'][alg]:
                    sil_row.append(result['baseline_results'][alg].get('silhouette_score', 0))
                    cal_row.append(result['baseline_results'][alg].get('calinski_harabasz_score', 0))
                    db_row.append(result['baseline_results'][alg].get('davies_bouldin_score', float('inf')))
                    time_row.append(result['baseline_results'][alg].get('processing_time', 0))
                else:
                    sil_row.append(0)
                    cal_row.append(0)
                    db_row.append(float('inf'))
                    time_row.append(0)
            
            # 제안된 알고리즘
            if 'proposed_results' in result and 'error' not in result['proposed_results']:
                sil_row.append(result['proposed_results'].get('silhouette_score', 0))
                cal_row.append(result['proposed_results'].get('calinski_harabasz_score', 0))
                db_row.append(result['proposed_results'].get('davies_bouldin_score', float('inf')))
                time_row.append(result['proposed_results'].get('processing_time', 0))
            else:
                sil_row.append(0)
                cal_row.append(0)
                db_row.append(float('inf'))
                time_row.append(0)
            
            silhouette_data.append(sil_row)
            calinski_data.append(cal_row)
            davies_bouldin_data.append(db_row)
            processing_time_data.append(time_row)
        
        if not datasets:
            logger.warning("시각화할 데이터가 없습니다.")
            return
        
        # 시각화 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. 실루엣 점수 비교
        silhouette_df = pd.DataFrame(silhouette_data, index=datasets, columns=algorithm_names)
        sns.heatmap(silhouette_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0,0])
        axes[0,0].set_title('Silhouette Score (Higher is Better)')
        axes[0,0].set_ylabel('Datasets')
        
        # 2. Calinski-Harabasz 점수 비교
        calinski_df = pd.DataFrame(calinski_data, index=datasets, columns=algorithm_names)
        sns.heatmap(calinski_df, annot=True, fmt='.1f', cmap='YlGnBu', ax=axes[0,1])
        axes[0,1].set_title('Calinski-Harabasz Score (Higher is Better)')
        
        # 3. Davies-Bouldin 점수 비교 (무한대 값 처리)
        davies_bouldin_clean = []
        for row in davies_bouldin_data:
            clean_row = [val if val != float('inf') else 10 for val in row]  # inf를 10으로 대체
            davies_bouldin_clean.append(clean_row)
        
        davies_df = pd.DataFrame(davies_bouldin_clean, index=datasets, columns=algorithm_names)
        sns.heatmap(davies_df, annot=True, fmt='.3f', cmap='YlOrRd_r', ax=axes[1,0])
        axes[1,0].set_title('Davies-Bouldin Score (Lower is Better)')
        axes[1,0].set_ylabel('Datasets')
        
        # 4. 처리 시간 비교
        time_df = pd.DataFrame(processing_time_data, index=datasets, columns=algorithm_names)
        sns.heatmap(time_df, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1,1])
        axes[1,1].set_title('Processing Time (Seconds)')
        
        plt.tight_layout()
        plt.savefig('baseline_comparison_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 막대 그래프로 평균 성능 비교
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Average Performance Comparison Across Datasets', fontsize=16, fontweight='bold')
        
        # 평균 계산
        avg_silhouette = np.mean(silhouette_data, axis=0)
        avg_calinski = np.mean(calinski_data, axis=0)
        avg_davies = np.mean(davies_bouldin_clean, axis=0)
        avg_time = np.mean(processing_time_data, axis=0)
        
        # 막대 그래프
        x_pos = np.arange(len(algorithm_names))
        
        bars1 = axes[0,0].bar(x_pos, avg_silhouette, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        axes[0,0].set_title('Average Silhouette Score')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(algorithm_names, rotation=45)
        axes[0,0].set_ylabel('Silhouette Score')
        
        # 값 표시
        for bar, val in zip(bars1, avg_silhouette):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{val:.3f}', ha='center', va='bottom')
        
        bars2 = axes[0,1].bar(x_pos, avg_calinski, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        axes[0,1].set_title('Average Calinski-Harabasz Score')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(algorithm_names, rotation=45)
        axes[0,1].set_ylabel('Calinski-Harabasz Score')
        
        bars3 = axes[1,0].bar(x_pos, avg_davies, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        axes[1,0].set_title('Average Davies-Bouldin Score')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(algorithm_names, rotation=45)
        axes[1,0].set_ylabel('Davies-Bouldin Score')
        
        bars4 = axes[1,1].bar(x_pos, avg_time, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        axes[1,1].set_title('Average Processing Time')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(algorithm_names, rotation=45)
        axes[1,1].set_ylabel('Processing Time (seconds)')
        
        plt.tight_layout()
        plt.savefig('baseline_comparison_bars.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("비교 시각화 완료: baseline_comparison_heatmap.png, baseline_comparison_bars.png")
        
        return {
            'avg_silhouette': avg_silhouette,
            'avg_calinski': avg_calinski,
            'avg_davies': avg_davies,
            'avg_time': avg_time,
            'algorithm_names': algorithm_names
        }
    
    def generate_comparison_report(self, avg_metrics):
        """비교 보고서 생성"""
        logger.info("비교 보고서 생성 중...")
        
        with open('baseline_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write("Dynamic Context Flag-Based Hierarchical Algorithm vs Baseline Algorithms\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("실험 개요:\n")
            f.write("- 제안된 Dynamic Context Flag 알고리즘과 기존 클러스터링 알고리즘들의 성능 비교\n")
            f.write("- 평가 메트릭: Silhouette Score, Calinski-Harabasz Score, Davies-Bouldin Score, Processing Time\n")
            f.write("- 데이터셋: Enron Email, 20 Newsgroups, Reuters-21578\n\n")
            
            if avg_metrics:
                f.write("평균 성능 비교 결과:\n")
                f.write("-" * 50 + "\n")
                
                algorithms = avg_metrics['algorithm_names']
                
                f.write("1. Silhouette Score (높을수록 좋음):\n")
                for i, (alg, score) in enumerate(zip(algorithms, avg_metrics['avg_silhouette'])):
                    f.write(f"   {alg}: {score:.4f}\n")
                
                best_sil_idx = np.argmax(avg_metrics['avg_silhouette'])
                f.write(f"   → 최고 성능: {algorithms[best_sil_idx]}\n\n")
                
                f.write("2. Calinski-Harabasz Score (높을수록 좋음):\n")
                for i, (alg, score) in enumerate(zip(algorithms, avg_metrics['avg_calinski'])):
                    f.write(f"   {alg}: {score:.2f}\n")
                
                best_cal_idx = np.argmax(avg_metrics['avg_calinski'])
                f.write(f"   → 최고 성능: {algorithms[best_cal_idx]}\n\n")
                
                f.write("3. Davies-Bouldin Score (낮을수록 좋음):\n")
                for i, (alg, score) in enumerate(zip(algorithms, avg_metrics['avg_davies'])):
                    f.write(f"   {alg}: {score:.4f}\n")
                
                best_db_idx = np.argmin(avg_metrics['avg_davies'])
                f.write(f"   → 최고 성능: {algorithms[best_db_idx]}\n\n")
                
                f.write("4. Processing Time (낮을수록 좋음):\n")
                for i, (alg, time_val) in enumerate(zip(algorithms, avg_metrics['avg_time'])):
                    f.write(f"   {alg}: {time_val:.4f}초\n")
                
                best_time_idx = np.argmin(avg_metrics['avg_time'])
                f.write(f"   → 최고 성능: {algorithms[best_time_idx]}\n\n")
                
                # 종합 평가
                f.write("종합 평가:\n")
                f.write("-" * 50 + "\n")
                
                # 제안된 알고리즘 (인덱스 3)의 성능 분석
                proposed_idx = 3
                if proposed_idx < len(algorithms):
                    proposed_sil = avg_metrics['avg_silhouette'][proposed_idx]
                    proposed_rank_sil = sorted(avg_metrics['avg_silhouette'], reverse=True).index(proposed_sil) + 1
                    
                    f.write(f"제안된 알고리즘 (Dynamic Context Flag):\n")
                    f.write(f"- Silhouette Score 순위: {proposed_rank_sil}/{len(algorithms)}\n")
                    f.write(f"- 실루엣 점수: {proposed_sil:.4f}\n")
                    
                    if proposed_rank_sil == 1:
                        f.write("- 결과: 최고 성능 달성! ✓\n")
                    elif proposed_rank_sil <= 2:
                        f.write("- 결과: 우수한 성능 ✓\n")
                    else:
                        f.write("- 결과: 개선 필요\n")
        
        logger.info("비교 보고서 생성 완료: baseline_comparison_report.txt")

def main():
    """메인 실행 함수"""
    print("베이스라인 알고리즘과의 성능 비교 실험")
    print("=" * 60)
    
    comparison = BaselineComparison()
    
    # 1. 모든 알고리즘 비교
    results = comparison.compare_all_algorithms()
    
    # 2. 시각화 생성
    avg_metrics = comparison.create_comparison_visualizations()
    
    # 3. 보고서 생성
    comparison.generate_comparison_report(avg_metrics)
    
    print("\n베이스라인 비교 실험 완료!")
    print("결과 파일:")
    print("- baseline_comparison_heatmap.png")
    print("- baseline_comparison_bars.png") 
    print("- baseline_comparison_report.txt")
    
    return results

if __name__ == "__main__":
    main()
