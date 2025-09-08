"""
Dynamic Context Flag-Based Hierarchical Algorithm for Large-Scale Document Context Linking and Integration

Main script for paper reproduction experiments

Core components:
1. Dynamic Context Flag Generation
2. Hierarchical Document Clustering
3. Context Linking Algorithm
4. Document Integration Framework
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import logging
import json
from datetime import datetime

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DynamicContextFlag:
    """
    Dynamic Context Flag Generator
    Converts semantic, structural, and temporal context of documents into flags
    """
    
    def __init__(self, flag_dimensions=10):
        self.flag_dimensions = flag_dimensions
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.context_weights = {
            'semantic': 0.4,
            'structural': 0.3,
            'temporal': 0.2,
            'categorical': 0.1
        }
        
    def generate_semantic_flags(self, documents):
        """Generate semantic context flags"""
        try:
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            # Generate flags by major topics
            semantic_flags = []
            for doc_vec in tfidf_matrix.toarray():
                top_indices = np.argsort(doc_vec)[-self.flag_dimensions:]
                flag = np.zeros(self.flag_dimensions)
                for i, idx in enumerate(top_indices):
                    flag[i] = doc_vec[idx]
                semantic_flags.append(flag)
            return np.array(semantic_flags)
        except Exception as e:
            logger.error(f"Semantic flag generation error: {e}")
            return np.zeros((len(documents), self.flag_dimensions))
    
    def generate_structural_flags(self, documents):
        """Generate structural context flags"""
        structural_flags = []
        for doc in documents:
            doc_str = str(doc) if doc is not None else ""
            # Extract document structural features
            features = [
                len(doc_str),  # Document length
                len(doc_str.split()),  # Word count
                len(doc_str.split('\n')),  # Line count
                doc_str.count('!'),  # Exclamation marks
                doc_str.count('?'),  # Question marks
                doc_str.count('@'),  # Email addresses
                doc_str.count('http'),  # URLs
                len([w for w in doc_str.split() if w.isupper()]),  # Uppercase words
                doc_str.count(','),  # Commas
                doc_str.count('.')   # Periods
            ]
            # Normalization
            flag = np.array(features[:self.flag_dimensions])
            if np.max(flag) > 0:
                flag = flag / np.max(flag)
            structural_flags.append(flag)
        return np.array(structural_flags)
    
    def generate_context_flags(self, documents, categories=None):
        """Generate integrated context flags"""
        logger.info(f"Generating context flags for {len(documents)} documents...")
        
        semantic_flags = self.generate_semantic_flags(documents)
        structural_flags = self.generate_structural_flags(documents)
        
        # Apply weights and integrate
        context_flags = (
            self.context_weights['semantic'] * semantic_flags +
            self.context_weights['structural'] * structural_flags
        )
        
        logger.info("Context flag generation completed")
        return context_flags

class HierarchicalDocumentClustering:
    """
    Hierarchical Document Clustering
    Hierarchically clusters documents based on context flags
    """
    
    def __init__(self, n_clusters=5, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.clustering_model = None
        self.hierarchy_levels = 3  # Number of hierarchy levels
        
    def create_hierarchical_clusters(self, context_flags):
        """Create hierarchical clusters"""
        logger.info("Performing hierarchical clustering...")
        
        hierarchy = {}
        current_data = context_flags.copy()
        
        for level in range(self.hierarchy_levels):
            n_clusters_level = max(2, self.n_clusters // (level + 1))
            
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters_level,
                linkage=self.linkage
            )
            
            labels = clustering.fit_predict(current_data)
            hierarchy[f'level_{level}'] = labels
            
            # Calculate cluster centroids for next level
            cluster_centers = []
            for cluster_id in range(n_clusters_level):
                cluster_mask = labels == cluster_id
                if np.any(cluster_mask):
                    center = np.mean(current_data[cluster_mask], axis=0)
                    cluster_centers.append(center)
            
            current_data = np.array(cluster_centers) if cluster_centers else current_data
            
        logger.info("Hierarchical clustering completed")
        return hierarchy

class ContextLinkingAlgorithm:
    """
    Context Linking Algorithm
    Establishes connection relationships based on contextual similarity between documents
    """
    
    def __init__(self, similarity_threshold=0.5):
        self.similarity_threshold = similarity_threshold
        self.link_graph = defaultdict(list)
        
    def calculate_context_similarity(self, flags1, flags2):
        """Calculate context similarity"""
        return cosine_similarity([flags1], [flags2])[0][0]
    
    def create_context_links(self, context_flags, documents):
        """Create context-based document connections"""
        logger.info("Creating context links between documents...")
        
        n_docs = len(context_flags)
        similarity_matrix = np.zeros((n_docs, n_docs))
        
        # Calculate similarity for all document pairs
        for i in range(n_docs):
            for j in range(i+1, n_docs):
                similarity = self.calculate_context_similarity(
                    context_flags[i], context_flags[j]
                )
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
                
                # Create link if above threshold
                if similarity >= self.similarity_threshold:
                    self.link_graph[i].append((j, similarity))
                    self.link_graph[j].append((i, similarity))
        
        logger.info(f"Context link generation completed for {len(self.link_graph)} documents")
        return similarity_matrix, self.link_graph

class DocumentIntegrationFramework:
    """
    Document Integration Framework
    Semantically integrates and summarizes linked documents
    """
    
    def __init__(self):
        self.integrated_clusters = {}
        
    def integrate_linked_documents(self, documents, link_graph, context_flags):
        """Integrate linked documents"""
        logger.info("Integrating linked documents...")
        
        visited = set()
        integrated_groups = []
        
        for doc_id in range(len(documents)):
            if doc_id not in visited:
                group = self._dfs_collect_group(doc_id, link_graph, visited)
                if len(group) > 1:  # Only when 2 or more documents are connected
                    integrated_groups.append(group)
        
        # Generate integrated summary for each group
        for i, group in enumerate(integrated_groups):
            group_docs = [documents[doc_id] for doc_id in group]
            summary = self._create_group_summary(group_docs)
            self.integrated_clusters[f'cluster_{i}'] = {
                'document_ids': group,
                'summary': summary,
                'size': len(group)
            }
        
        logger.info(f"Integration completed with {len(integrated_groups)} integrated clusters")
        return self.integrated_clusters
    
    def _dfs_collect_group(self, start_id, link_graph, visited):
        """Collect connected document groups using DFS"""
        group = []
        stack = [start_id]
        
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                group.append(current)
                
                # Add connected documents to stack
                if current in link_graph:
                    for linked_doc, _ in link_graph[current]:
                        if linked_doc not in visited:
                            stack.append(linked_doc)
        
        return group
    
    def _create_group_summary(self, group_docs):
        """Generate document group summary"""
        # Simple summary: combine first 100 characters of each document
        summary_parts = []
        for doc in group_docs:
            if doc is not None:
                doc_str = str(doc)[:100]
                if doc_str.strip():
                    summary_parts.append(doc_str.strip())
        
        if not summary_parts:
            return "Empty summary"
        
        return " | ".join(summary_parts[:3])  # Include maximum 3 documents in summary

def main():
    """Main execution function"""
    print("=" * 80)
    print("Dynamic Context Flag-Based Hierarchical Algorithm")
    print("Large-Scale Document Context Linking and Integration")
    print("=" * 80)
    
    # Initialize algorithm components
    context_flag_gen = DynamicContextFlag(flag_dimensions=10)
    hierarchical_clustering = HierarchicalDocumentClustering(n_clusters=5)
    context_linking = ContextLinkingAlgorithm(similarity_threshold=0.3)
    integration_framework = DocumentIntegrationFramework()
    
    print("\nAlgorithm components initialization completed")
    print("Next step will load datasets and perform experiments...")

if __name__ == "__main__":
    main()
