"""
Dataset Loading and Preprocessing Module

Loads and preprocesses various datasets from rawdata folder
for use in Dynamic Context Flag-Based Hierarchical Algorithm experiments
"""

import os
import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import email
from email.parser import Parser
import tarfile
import zipfile

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Dataset loading and preprocessing class"""
    
    def __init__(self, rawdata_path: str = "rawdata"):
        self.rawdata_path = Path(rawdata_path)
        self.datasets = {}
        
    def load_enron_emails(self, limit: int = 1000) -> Tuple[List[str], List[str]]:
        """
        Load Enron email dataset
        
        Args:
            limit: Maximum number of emails to load
            
        Returns:
            Tuple[List[str], List[str]]: (email content list, label list)
        """
        logger.info("Loading Enron email dataset...")
        
        try:
            # 검증된 인텐트 데이터셋 로드
            intent_pos_path = self.rawdata_path / "enron_intent_dataset_verified-master" / "enron_intent_dataset_verified-master" / "intent_pos"
            intent_neg_path = self.rawdata_path / "enron_intent_dataset_verified-master" / "enron_intent_dataset_verified-master" / "intent_neg"
            
            documents = []
            labels = []
            
            # Load positive intent emails
            if intent_pos_path.exists():
                with open(intent_pos_path, 'r', encoding='utf-8', errors='ignore') as f:
                    pos_emails = f.readlines()[:limit//2]
                    documents.extend([email.strip() for email in pos_emails if email.strip()])
                    labels.extend(['positive'] * len([email for email in pos_emails if email.strip()]))
            
            # Load negative intent emails
            if intent_neg_path.exists():
                with open(intent_neg_path, 'r', encoding='utf-8', errors='ignore') as f:
                    neg_emails = f.readlines()[:limit//2]
                    documents.extend([email.strip() for email in neg_emails if email.strip()])
                    labels.extend(['negative'] * len([email for email in neg_emails if email.strip()]))
            
            # Alternative method if files not found: generate sample data
            if not documents:
                logger.warning("Enron intent data not found. Generating sample data...")
                documents = self._generate_sample_emails(limit)
                labels = ['unknown'] * len(documents)
            
            logger.info(f"Enron email loading completed: {len(documents)} documents")
            return documents, labels
            
        except Exception as e:
            logger.error(f"Enron data loading error: {e}")
            # Return sample data on error
            documents = self._generate_sample_emails(limit)
            labels = ['unknown'] * len(documents)
            return documents, labels
    
    def load_20newsgroups(self, limit: int = 1000) -> Tuple[List[str], List[str]]:
        """
        Load 20 Newsgroups dataset
        
        Args:
            limit: Maximum number of documents to load
            
        Returns:
            Tuple[List[str], List[str]]: (document content list, category list)
        """
        logger.info("Loading 20 Newsgroups dataset...")
        
        documents = []
        categories = []
        
        try:
            # Check text files in archive (1) folder
            archive_path = self.rawdata_path / "archive (1)"
            if archive_path.exists():
                txt_files = list(archive_path.glob("*.txt"))
                
                for txt_file in txt_files:
                    if txt_file.name != "list.csv":
                        category = txt_file.stem
                        try:
                            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                # Split into individual documents (based on Newsgroup: header)
                                docs = re.split(r'\nNewsgroup:', content)
                                
                                for doc in docs[:limit//len(txt_files)]:
                                    if len(doc.strip()) > 100:  # Minimum length filter
                                        documents.append(doc.strip())
                                        categories.append(category)
                                        
                                        if len(documents) >= limit:
                                            break
                        except Exception as e:
                            logger.warning(f"{txt_file} 읽기 오류: {e}")
                            continue
                    
                    if len(documents) >= limit:
                        break
            
            # Alternative method: use 20news-18828 folder
            if not documents:
                newsgroups_path = self.rawdata_path / "20news-18828" / "20news-18828"
                if newsgroups_path.exists():
                    category_dirs = [d for d in newsgroups_path.iterdir() if d.is_dir()]
                    
                    for category_dir in category_dirs:
                        category = category_dir.name
                        doc_files = list(category_dir.glob("*"))[:limit//len(category_dirs)]
                        
                        for doc_file in doc_files:
                            try:
                                with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                    if len(content.strip()) > 50:
                                        documents.append(content.strip())
                                        categories.append(category)
                            except Exception as e:
                                continue
                        
                        if len(documents) >= limit:
                            break
            
            if not documents:
                logger.warning("20 Newsgroups data not found. Generating sample data...")
                documents = self._generate_sample_newsgroups(limit)
                categories = ['sample'] * len(documents)
            
            logger.info(f"20 Newsgroups loading completed: {len(documents)} documents")
            return documents, categories
            
        except Exception as e:
            logger.error(f"20 Newsgroups data loading error: {e}")
            documents = self._generate_sample_newsgroups(limit)
            categories = ['sample'] * len(documents)
            return documents, categories
    
    def load_reuters21578(self, limit: int = 1000) -> Tuple[List[str], List[str]]:
        """
        Load Reuters-21578 dataset
        
        Args:
            limit: Maximum number of documents to load
            
        Returns:
            Tuple[List[str], List[str]]: (document content list, topic list)
        """
        logger.info("Loading Reuters-21578 dataset...")
        
        documents = []
        topics = []
        
        try:
            # Try preprocessed CSV data first
            csv_path = self.rawdata_path / "archive (2)"
            if csv_path.exists():
                csv_files = list(csv_path.glob("*train.csv"))
                
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file, nrows=limit//len(csv_files))
                        if 'text' in df.columns and 'topics' in df.columns:
                            documents.extend(df['text'].dropna().tolist())
                            topics.extend(df['topics'].dropna().tolist())
                        elif len(df.columns) >= 2:
                            # Use first and second columns if column names differ
                            documents.extend(df.iloc[:, 0].dropna().astype(str).tolist())
                            topics.extend(df.iloc[:, 1].dropna().astype(str).tolist())
                    except Exception as e:
                        logger.warning(f"{csv_file} 읽기 오류: {e}")
                        continue
                
                if len(documents) >= limit:
                    documents = documents[:limit]
                    topics = topics[:limit]
            
            # Direct parsing from SGML files (simple text extraction due to complex structure)
            if not documents:
                sgml_path = self.rawdata_path / "reuters21578"
                if sgml_path.exists():
                    sgml_files = list(sgml_path.glob("reut2-*.sgm"))[:5]  # 처음 5개 파일만
                    
                    for sgml_file in sgml_files:
                        try:
                            with open(sgml_file, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                            # Extract body text using simple regex
                            doc_matches = re.findall(r'<BODY>(.*?)</BODY>', content, re.DOTALL)
                            topic_matches = re.findall(r'<TOPICS><D>(.*?)</D></TOPICS>', content, re.DOTALL)
                            
                            for i, doc_text in enumerate(doc_matches):
                                if len(doc_text.strip()) > 50:
                                    documents.append(doc_text.strip())
                                    topic = topic_matches[i] if i < len(topic_matches) else 'unknown'
                                    topics.append(topic)
                                    
                                if len(documents) >= limit:
                                    break
                        except Exception as e:
                            logger.warning(f"{sgml_file} 파싱 오류: {e}")
                            continue
                        
                        if len(documents) >= limit:
                            break
            
            if not documents:
                logger.warning("Reuters-21578 data not found. Generating sample data...")
                documents = self._generate_sample_reuters(limit)
                topics = ['sample'] * len(documents)
            
            logger.info(f"Reuters-21578 loading completed: {len(documents)} documents")
            return documents, topics
            
        except Exception as e:
            logger.error(f"Reuters-21578 data loading error: {e}")
            documents = self._generate_sample_reuters(limit)
            topics = ['sample'] * len(documents)
            return documents, topics
    
    def load_mixed_dataset(self, total_limit: int = 3000) -> Tuple[List[str], List[str], List[str]]:
        """
        Load all datasets in mixed format
        
        Args:
            total_limit: Total number of documents to load
            
        Returns:
            Tuple[List[str], List[str], List[str]]: (document list, label list, dataset source list)
        """
        logger.info("Loading mixed dataset...")
        
        all_documents = []
        all_labels = []
        all_sources = []
        
        limit_per_dataset = total_limit // 3
        
        # Enron emails
        try:
            enron_docs, enron_labels = self.load_enron_emails(limit_per_dataset)
            all_documents.extend(enron_docs)
            all_labels.extend(enron_labels)
            all_sources.extend(['enron'] * len(enron_docs))
        except Exception as e:
            logger.error(f"Enron data loading failed: {e}")
        
        # 20 Newsgroups
        try:
            news_docs, news_categories = self.load_20newsgroups(limit_per_dataset)
            all_documents.extend(news_docs)
            all_labels.extend(news_categories)
            all_sources.extend(['20newsgroups'] * len(news_docs))
        except Exception as e:
            logger.error(f"20 Newsgroups data loading failed: {e}")
        
        # Reuters-21578
        try:
            reuters_docs, reuters_topics = self.load_reuters21578(limit_per_dataset)
            all_documents.extend(reuters_docs)
            all_labels.extend(reuters_topics)
            all_sources.extend(['reuters'] * len(reuters_docs))
        except Exception as e:
            logger.error(f"Reuters data loading failed: {e}")
        
        logger.info(f"Mixed dataset loading completed: {len(all_documents)} total documents")
        logger.info(f"Distribution by data source: {pd.Series(all_sources).value_counts().to_dict()}")
        
        return all_documents, all_labels, all_sources
    
    def _generate_sample_emails(self, count: int) -> List[str]:
        """Generate sample email data"""
        sample_emails = [
            "Please review the quarterly report and send your feedback by Friday.",
            "Meeting scheduled for tomorrow at 10 AM in conference room A.",
            "Can you provide an update on the project status?",
            "The deadline for the proposal submission has been extended to next week.",
            "Please coordinate with the legal team for contract review.",
            "Budget allocation for Q3 needs immediate attention.",
            "System maintenance is scheduled for this weekend.",
            "New policy changes will be effective from next month.",
            "Please attend the training session on data security.",
            "Performance review meeting is scheduled for next Tuesday."
        ]
        
        # Generate by repeating as needed
        result = []
        for i in range(count):
            base_email = sample_emails[i % len(sample_emails)]
            result.append(f"Sample Email {i+1}: {base_email}")
        
        return result
    
    def _generate_sample_newsgroups(self, count: int) -> List[str]:
        """Generate sample newsgroup data"""
        sample_topics = [
            "Computer graphics and 3D rendering techniques are advancing rapidly in modern applications.",
            "Windows operating system updates bring new features and security improvements.",
            "Hardware compatibility issues between different PC components can cause system instability.",
            "Automotive industry trends show increasing focus on electric vehicle development.",
            "Motorcycle safety equipment has significantly improved over the past decade.",
            "Baseball statistics analysis reveals interesting patterns in player performance.",
            "Space exploration missions continue to discover new celestial phenomena.",
            "Medical research breakthrough offers hope for treating rare diseases.",
            "Cryptography algorithms play crucial role in modern data security systems.",
            "Political discussions often involve complex policy considerations and public opinion."
        ]
        
        result = []
        for i in range(count):
            base_topic = sample_topics[i % len(sample_topics)]
            result.append(f"Newsgroup Post {i+1}: {base_topic}")
        
        return result
    
    def _generate_sample_reuters(self, count: int) -> List[str]:
        """Generate sample Reuters news data"""
        sample_news = [
            "Financial markets showed mixed performance today as investors awaited earnings reports.",
            "Oil prices fluctuated amid concerns about global supply chain disruptions.",
            "Technology stocks gained ground following positive quarterly results announcements.",
            "Currency exchange rates remained stable despite economic uncertainty in emerging markets.",
            "Corporate merger activity increased significantly in the telecommunications sector.",
            "Interest rate decisions by central banks continue to influence investment strategies.",
            "Commodity prices rose due to supply constraints and increased industrial demand.",
            "International trade agreements face new challenges from changing political landscapes.",
            "Banking sector reforms aim to strengthen financial stability and consumer protection.",
            "Energy sector companies invest heavily in renewable technology development projects."
        ]
        
        result = []
        for i in range(count):
            base_news = sample_news[i % len(sample_news)]
            result.append(f"Reuters Article {i+1}: {base_news}")
        
        return result

def test_data_loading():
    """Data loading test function"""
    loader = DatasetLoader()
    
    print("=" * 60)
    print("Dataset Loading Test")
    print("=" * 60)
    
    # Individual dataset tests
    print("\n1. Enron Email Dataset Test")
    enron_docs, enron_labels = loader.load_enron_emails(limit=100)
    print(f"   Documents loaded: {len(enron_docs)}")
    print(f"   Label distribution: {pd.Series(enron_labels).value_counts().to_dict()}")
    print(f"   Sample document: {enron_docs[0][:100]}...")
    
    print("\n2. 20 Newsgroups Dataset Test")
    news_docs, news_cats = loader.load_20newsgroups(limit=100)
    print(f"   Documents loaded: {len(news_docs)}")
    print(f"   Category distribution: {pd.Series(news_cats).value_counts().to_dict()}")
    print(f"   Sample document: {news_docs[0][:100]}...")
    
    print("\n3. Reuters-21578 Dataset Test")
    reuters_docs, reuters_topics = loader.load_reuters21578(limit=100)
    print(f"   Documents loaded: {len(reuters_docs)}")
    print(f"   Topic distribution: {pd.Series(reuters_topics).value_counts().to_dict()}")
    print(f"   Sample document: {reuters_docs[0][:100]}...")
    
    print("\n4. Mixed Dataset Test")
    mixed_docs, mixed_labels, mixed_sources = loader.load_mixed_dataset(total_limit=300)
    print(f"   Total documents loaded: {len(mixed_docs)}")
    print(f"   Source distribution: {pd.Series(mixed_sources).value_counts().to_dict()}")
    
    return mixed_docs, mixed_labels, mixed_sources

if __name__ == "__main__":
    test_data_loading()
