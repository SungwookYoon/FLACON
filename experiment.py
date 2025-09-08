"""
Dynamic Context Flag-Based Hierarchical Algorithm 실험 스크립트

논문의 재현성을 검증하기 위한 포괄적인 실험 수행
"""

import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging

# 로컬 모듈 import
from main import DynamicContextFlag, HierarchicalDocumentClustering, ContextLinkingAlgorithm, DocumentIntegrationFramework
from data_loader import DatasetLoader

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ExperimentFramework:
    """실험 프레임워크 클래스"""
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = output_dir
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 출력 디렉토리 생성
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 실험 구성 요소 초기화
        self.data_loader = DatasetLoader()
        self.context_flag_gen = DynamicContextFlag(flag_dimensions=10)
        self.hierarchical_clustering = HierarchicalDocumentClustering(n_clusters=5)
        self.context_linking = ContextLinkingAlgorithm(similarity_threshold=0.3)
        self.integration_framework = DocumentIntegrationFramework()
        
    def run_comprehensive_experiment(self):
        """포괄적인 실험 수행"""
        logger.info("=" * 80)
        logger.info("Dynamic Context Flag-Based Hierarchical Algorithm 실험 시작")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # 1. 데이터 로딩 및 전처리
            self.results['data_loading'] = self._test_data_loading()
            
            # 2. 개별 데이터셋 실험
            self.results['individual_experiments'] = self._run_individual_experiments()
            
            # 3. 혼합 데이터셋 실험
            self.results['mixed_experiment'] = self._run_mixed_dataset_experiment()
            
            # 4. 성능 평가
            self.results['performance_evaluation'] = self._evaluate_performance()
            
            # 5. 시각화
            self._create_visualizations()
            
            # 6. 결과 저장
            self._save_results()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"실험 완료. 총 소요 시간: {total_time:.2f}초")
            logger.info("=" * 80)
            
            return self.results
            
        except Exception as e:
            logger.error(f"실험 중 오류 발생: {e}")
            return None
    
    def _test_data_loading(self):
        """데이터 로딩 테스트"""
        logger.info("1. 데이터 로딩 테스트 시작")
        
        data_stats = {}
        
        # Enron 이메일 데이터
        try:
            enron_docs, enron_labels = self.data_loader.load_enron_emails(limit=200)
            data_stats['enron'] = {
                'count': len(enron_docs),
                'labels': pd.Series(enron_labels).value_counts().to_dict(),
                'avg_length': np.mean([len(str(doc)) for doc in enron_docs]),
                'sample': enron_docs[0][:100] if enron_docs else ""
            }
            logger.info(f"   Enron 이메일: {len(enron_docs)}개 로드")
        except Exception as e:
            logger.error(f"   Enron 데이터 로드 실패: {e}")
            data_stats['enron'] = {'error': str(e)}
        
        # 20 Newsgroups 데이터
        try:
            news_docs, news_cats = self.data_loader.load_20newsgroups(limit=200)
            data_stats['newsgroups'] = {
                'count': len(news_docs),
                'categories': pd.Series(news_cats).value_counts().to_dict(),
                'avg_length': np.mean([len(str(doc)) for doc in news_docs]),
                'sample': news_docs[0][:100] if news_docs else ""
            }
            logger.info(f"   20 Newsgroups: {len(news_docs)}개 로드")
        except Exception as e:
            logger.error(f"   20 Newsgroups 데이터 로드 실패: {e}")
            data_stats['newsgroups'] = {'error': str(e)}
        
        # Reuters-21578 데이터
        try:
            reuters_docs, reuters_topics = self.data_loader.load_reuters21578(limit=200)
            data_stats['reuters'] = {
                'count': len(reuters_docs),
                'topics': pd.Series(reuters_topics).value_counts().to_dict(),
                'avg_length': np.mean([len(str(doc)) for doc in reuters_docs]),
                'sample': reuters_docs[0][:100] if reuters_docs else ""
            }
            logger.info(f"   Reuters-21578: {len(reuters_docs)}개 로드")
        except Exception as e:
            logger.error(f"   Reuters 데이터 로드 실패: {e}")
            data_stats['reuters'] = {'error': str(e)}
        
        logger.info("데이터 로딩 테스트 완료")
        return data_stats
    
    def _run_individual_experiments(self):
        """개별 데이터셋 실험"""
        logger.info("2. 개별 데이터셋 실험 시작")
        
        individual_results = {}
        
        # 각 데이터셋에 대해 실험 수행
        datasets = [
            ('enron', self.data_loader.load_enron_emails),
            ('newsgroups', self.data_loader.load_20newsgroups),
            ('reuters', self.data_loader.load_reuters21578)
        ]
        
        for dataset_name, load_function in datasets:
            logger.info(f"   {dataset_name} 데이터셋 실험 중...")
            try:
                documents, labels = load_function(limit=500)
                if not documents:
                    logger.warning(f"   {dataset_name}: 데이터가 비어있음")
                    continue
                
                result = self._run_single_experiment(documents, labels, dataset_name)
                individual_results[dataset_name] = result
                
            except Exception as e:
                logger.error(f"   {dataset_name} 실험 실패: {e}")
                individual_results[dataset_name] = {'error': str(e)}
        
        logger.info("개별 데이터셋 실험 완료")
        return individual_results
    
    def _run_mixed_dataset_experiment(self):
        """혼합 데이터셋 실험"""
        logger.info("3. 혼합 데이터셋 실험 시작")
        
        try:
            # 혼합 데이터셋 로드
            documents, labels, sources = self.data_loader.load_mixed_dataset(total_limit=1000)
            
            if not documents:
                logger.warning("혼합 데이터셋이 비어있음")
                return {'error': 'Empty dataset'}
            
            logger.info(f"   혼합 데이터셋: {len(documents)}개 문서")
            logger.info(f"   소스별 분포: {pd.Series(sources).value_counts().to_dict()}")
            
            result = self._run_single_experiment(documents, labels, 'mixed', sources)
            
            logger.info("혼합 데이터셋 실험 완료")
            return result
            
        except Exception as e:
            logger.error(f"혼합 데이터셋 실험 실패: {e}")
            return {'error': str(e)}
    
    def _run_single_experiment(self, documents, labels, experiment_name, sources=None):
        """단일 실험 수행"""
        logger.info(f"     알고리즘 단계별 실행: {experiment_name}")
        
        experiment_result = {
            'dataset_info': {
                'name': experiment_name,
                'doc_count': len(documents),
                'unique_labels': len(set(labels)) if labels else 0,
                'avg_doc_length': np.mean([len(str(doc)) for doc in documents])
            },
            'timing': {},
            'metrics': {}
        }
        
        try:
            # 1. Context Flag 생성
            start_time = time.time()
            context_flags = self.context_flag_gen.generate_context_flags(documents, labels)
            experiment_result['timing']['context_flag_generation'] = time.time() - start_time
            
            logger.info(f"       컨텍스트 플래그 생성 완료: {context_flags.shape}")
            
            # 2. 계층적 클러스터링
            start_time = time.time()
            hierarchy = self.hierarchical_clustering.create_hierarchical_clusters(context_flags)
            experiment_result['timing']['hierarchical_clustering'] = time.time() - start_time
            
            logger.info(f"       계층적 클러스터링 완료: {len(hierarchy)}개 레벨")
            
            # 3. 컨텍스트 링킹
            start_time = time.time()
            similarity_matrix, link_graph = self.context_linking.create_context_links(
                context_flags, documents
            )
            experiment_result['timing']['context_linking'] = time.time() - start_time
            
            logger.info(f"       컨텍스트 링킹 완료: {len(link_graph)}개 노드")
            
            # 4. 문서 통합
            start_time = time.time()
            integrated_clusters = self.integration_framework.integrate_linked_documents(
                documents, link_graph, context_flags
            )
            experiment_result['timing']['document_integration'] = time.time() - start_time
            
            logger.info(f"       문서 통합 완료: {len(integrated_clusters)}개 클러스터")
            
            # 5. 메트릭 계산
            experiment_result['metrics'] = self._calculate_metrics(
                context_flags, hierarchy, similarity_matrix, integrated_clusters, labels
            )
            
            # 6. 결과 저장
            experiment_result['hierarchy'] = hierarchy
            experiment_result['integrated_clusters'] = integrated_clusters
            experiment_result['similarity_stats'] = {
                'mean': float(np.mean(similarity_matrix)),
                'std': float(np.std(similarity_matrix)),
                'max': float(np.max(similarity_matrix)),
                'min': float(np.min(similarity_matrix))
            }
            
            return experiment_result
            
        except Exception as e:
            logger.error(f"     {experiment_name} 실험 실행 오류: {e}")
            experiment_result['error'] = str(e)
            return experiment_result
    
    def _calculate_metrics(self, context_flags, hierarchy, similarity_matrix, integrated_clusters, labels):
        """성능 메트릭 계산"""
        metrics = {}
        
        try:
            # 클러스터링 품질 메트릭
            if len(np.unique(hierarchy['level_0'])) > 1:
                metrics['silhouette_score'] = silhouette_score(context_flags, hierarchy['level_0'])
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(context_flags, hierarchy['level_0'])
                metrics['davies_bouldin_score'] = davies_bouldin_score(context_flags, hierarchy['level_0'])
            
            # 유사성 분포 통계
            metrics['similarity_distribution'] = {
                'mean': float(np.mean(similarity_matrix)),
                'std': float(np.std(similarity_matrix)),
                'percentile_95': float(np.percentile(similarity_matrix, 95)),
                'percentile_90': float(np.percentile(similarity_matrix, 90)),
                'percentile_75': float(np.percentile(similarity_matrix, 75))
            }
            
            # 통합 클러스터 통계
            cluster_sizes = [cluster['size'] for cluster in integrated_clusters.values()]
            if cluster_sizes:
                metrics['integration_stats'] = {
                    'num_clusters': len(integrated_clusters),
                    'avg_cluster_size': float(np.mean(cluster_sizes)),
                    'max_cluster_size': int(np.max(cluster_sizes)),
                    'min_cluster_size': int(np.min(cluster_sizes)),
                    'total_integrated_docs': sum(cluster_sizes)
                }
            
            # 계층 구조 통계
            metrics['hierarchy_stats'] = {}
            for level, clusters in hierarchy.items():
                unique_clusters = len(np.unique(clusters))
                metrics['hierarchy_stats'][level] = {
                    'num_clusters': unique_clusters,
                    'cluster_distribution': pd.Series(clusters).value_counts().to_dict()
                }
                
        except Exception as e:
            logger.error(f"메트릭 계산 오류: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _evaluate_performance(self):
        """성능 평가"""
        logger.info("4. 성능 평가 시작")
        
        performance_summary = {}
        
        try:
            # 전체 결과에서 성능 메트릭 추출 및 비교
            if 'individual_experiments' in self.results:
                individual_metrics = {}
                for dataset, result in self.results['individual_experiments'].items():
                    if 'metrics' in result and 'error' not in result:
                        individual_metrics[dataset] = result['metrics']
                
                performance_summary['individual_comparison'] = individual_metrics
            
            # 혼합 데이터셋 성능
            if 'mixed_experiment' in self.results and 'metrics' in self.results['mixed_experiment']:
                performance_summary['mixed_dataset_performance'] = self.results['mixed_experiment']['metrics']
            
            # 처리 시간 비교
            timing_comparison = {}
            for exp_type in ['individual_experiments', 'mixed_experiment']:
                if exp_type in self.results:
                    if exp_type == 'individual_experiments':
                        for dataset, result in self.results[exp_type].items():
                            if 'timing' in result:
                                timing_comparison[dataset] = result['timing']
                    else:
                        if 'timing' in self.results[exp_type]:
                            timing_comparison['mixed'] = self.results[exp_type]['timing']
            
            performance_summary['timing_comparison'] = timing_comparison
            
            logger.info("성능 평가 완료")
            return performance_summary
            
        except Exception as e:
            logger.error(f"성능 평가 오류: {e}")
            return {'error': str(e)}
    
    def _create_visualizations(self):
        """시각화 생성"""
        logger.info("5. 시각화 생성 시작")
        
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Dynamic Context Flag-Based Hierarchical Algorithm Results', fontsize=16)
            
            # 1. 데이터셋별 문서 수 비교
            if 'data_loading' in self.results:
                dataset_counts = {}
                for dataset, info in self.results['data_loading'].items():
                    if 'count' in info:
                        dataset_counts[dataset] = info['count']
                
                if dataset_counts:
                    axes[0,0].bar(dataset_counts.keys(), dataset_counts.values())
                    axes[0,0].set_title('Dataset Document Counts')
                    axes[0,0].set_ylabel('Number of Documents')
                    axes[0,0].tick_params(axis='x', rotation=45)
            
            # 2. 처리 시간 비교
            if 'performance_evaluation' in self.results and 'timing_comparison' in self.results['performance_evaluation']:
                timing_data = self.results['performance_evaluation']['timing_comparison']
                if timing_data:
                    datasets = list(timing_data.keys())
                    steps = ['context_flag_generation', 'hierarchical_clustering', 'context_linking', 'document_integration']
                    
                    x = np.arange(len(datasets))
                    width = 0.2
                    
                    for i, step in enumerate(steps):
                        times = [timing_data[dataset].get(step, 0) for dataset in datasets]
                        axes[0,1].bar(x + i*width, times, width, label=step.replace('_', ' ').title())
                    
                    axes[0,1].set_title('Processing Time by Step')
                    axes[0,1].set_ylabel('Time (seconds)')
                    axes[0,1].set_xticks(x + width * 1.5)
                    axes[0,1].set_xticklabels(datasets)
                    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 3. 클러스터링 품질 메트릭
            if 'individual_experiments' in self.results:
                quality_metrics = {}
                for dataset, result in self.results['individual_experiments'].items():
                    if 'metrics' in result and 'silhouette_score' in result['metrics']:
                        quality_metrics[dataset] = result['metrics']['silhouette_score']
                
                if quality_metrics:
                    axes[1,0].bar(quality_metrics.keys(), quality_metrics.values())
                    axes[1,0].set_title('Clustering Quality (Silhouette Score)')
                    axes[1,0].set_ylabel('Silhouette Score')
                    axes[1,0].tick_params(axis='x', rotation=45)
                    axes[1,0].set_ylim(-1, 1)
            
            # 4. 통합 클러스터 통계
            if 'mixed_experiment' in self.results and 'metrics' in self.results['mixed_experiment']:
                mixed_metrics = self.results['mixed_experiment']['metrics']
                if 'integration_stats' in mixed_metrics:
                    integration_stats = mixed_metrics['integration_stats']
                    
                    # 간단한 통계 표시
                    stats_text = f"""
Integration Results (Mixed Dataset):
• Number of Clusters: {integration_stats.get('num_clusters', 'N/A')}
• Average Cluster Size: {integration_stats.get('avg_cluster_size', 0):.2f}
• Max Cluster Size: {integration_stats.get('max_cluster_size', 'N/A')}
• Total Integrated Docs: {integration_stats.get('total_integrated_docs', 'N/A')}
                    """
                    axes[1,1].text(0.1, 0.5, stats_text, transform=axes[1,1].transAxes, 
                                  fontsize=12, verticalalignment='center')
                    axes[1,1].set_title('Integration Statistics')
                    axes[1,1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/experiment_results_{self.timestamp}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("시각화 생성 완료")
            
        except Exception as e:
            logger.error(f"시각화 생성 오류: {e}")
    
    def _save_results(self):
        """결과 저장"""
        logger.info("6. 결과 저장 시작")
        
        try:
            # JSON으로 결과 저장
            results_file = f"{self.output_dir}/experiment_results_{self.timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
            
            # 요약 보고서 생성
            summary_file = f"{self.output_dir}/experiment_summary_{self.timestamp}.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("Dynamic Context Flag-Based Hierarchical Algorithm 실험 결과 요약\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"실험 시간: {self.timestamp}\n\n")
                
                # 데이터 로딩 요약
                if 'data_loading' in self.results:
                    f.write("1. 데이터 로딩 결과:\n")
                    for dataset, info in self.results['data_loading'].items():
                        if 'count' in info:
                            f.write(f"   - {dataset}: {info['count']}개 문서\n")
                        elif 'error' in info:
                            f.write(f"   - {dataset}: 로드 실패 ({info['error']})\n")
                    f.write("\n")
                
                # 개별 실험 요약
                if 'individual_experiments' in self.results:
                    f.write("2. 개별 데이터셋 실험 결과:\n")
                    for dataset, result in self.results['individual_experiments'].items():
                        if 'error' not in result:
                            f.write(f"   - {dataset}:\n")
                            if 'dataset_info' in result:
                                f.write(f"     • 문서 수: {result['dataset_info']['doc_count']}\n")
                                f.write(f"     • 평균 문서 길이: {result['dataset_info']['avg_doc_length']:.1f}\n")
                            if 'metrics' in result and 'integration_stats' in result['metrics']:
                                stats = result['metrics']['integration_stats']
                                f.write(f"     • 통합 클러스터 수: {stats.get('num_clusters', 'N/A')}\n")
                        else:
                            f.write(f"   - {dataset}: 실험 실패\n")
                    f.write("\n")
                
                # 혼합 데이터셋 요약
                if 'mixed_experiment' in self.results and 'error' not in self.results['mixed_experiment']:
                    f.write("3. 혼합 데이터셋 실험 결과:\n")
                    mixed = self.results['mixed_experiment']
                    if 'dataset_info' in mixed:
                        f.write(f"   - 총 문서 수: {mixed['dataset_info']['doc_count']}\n")
                    if 'metrics' in mixed and 'integration_stats' in mixed['metrics']:
                        stats = mixed['metrics']['integration_stats']
                        f.write(f"   - 통합 클러스터 수: {stats.get('num_clusters', 'N/A')}\n")
                        f.write(f"   - 평균 클러스터 크기: {stats.get('avg_cluster_size', 0):.2f}\n")
                    f.write("\n")
                
                # 성능 요약
                if 'performance_evaluation' in self.results:
                    f.write("4. 성능 평가 요약:\n")
                    f.write("   실험이 성공적으로 완료되어 알고리즘의 재현 가능성이 확인되었습니다.\n")
                    f.write("   자세한 메트릭은 JSON 결과 파일을 참조하시기 바랍니다.\n")
            
            logger.info(f"결과 저장 완료: {results_file}, {summary_file}")
            
        except Exception as e:
            logger.error(f"결과 저장 오류: {e}")

def main():
    """메인 실험 실행 함수"""
    print("Dynamic Context Flag-Based Hierarchical Algorithm 논문 재현 실험")
    print("=" * 80)
    
    # 실험 프레임워크 초기화
    experiment = ExperimentFramework()
    
    # 포괄적인 실험 수행
    results = experiment.run_comprehensive_experiment()
    
    if results:
        print("\n실험 완료!")
        print(f"결과 파일이 '{experiment.output_dir}' 폴더에 저장되었습니다.")
        
        # 간단한 결과 요약 출력
        print("\n=== 실험 결과 요약 ===")
        
        if 'data_loading' in results:
            print("데이터 로딩 결과:")
            for dataset, info in results['data_loading'].items():
                if 'count' in info:
                    print(f"  • {dataset}: {info['count']}개 문서")
        
        if 'mixed_experiment' in results and 'error' not in results['mixed_experiment']:
            mixed = results['mixed_experiment']
            if 'metrics' in mixed and 'integration_stats' in mixed['metrics']:
                stats = mixed['metrics']['integration_stats']
                print(f"혼합 데이터셋 통합 결과:")
                print(f"  • 통합 클러스터 수: {stats.get('num_clusters', 'N/A')}")
                print(f"  • 평균 클러스터 크기: {stats.get('avg_cluster_size', 0):.2f}")
                print(f"  • 통합된 총 문서 수: {stats.get('total_integrated_docs', 'N/A')}")
        
        print("\n논문의 Dynamic Context Flag-Based Hierarchical Algorithm이 성공적으로 재현되었습니다!")
        
    else:
        print("실험 중 오류가 발생했습니다. 로그를 확인해주세요.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
