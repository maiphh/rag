import pandas as pd
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import json

class RetrievalEvaluator:
    def __init__(self, rag_instance):
        self.rag = rag_instance
    
    def evaluate(self):
        queries = self.create_test_queries()
        return self.evaluate_retrieval(queries)
    
    def evaluate_retrieval(self, test_queries: List[Dict]) -> Dict:
        """
        Evaluate retrieval performance using test queries.
        test_queries format: [{"query": "...", "relevant_docs": ["doc1", "doc2"]}]
        """
        results = {
            "precision_at_k": [],
            "recall_at_k": [],
            "mrr": [],
            "ndcg": []
        }
        
        for test_case in test_queries:
            query = test_case["query"]
            relevant_docs = set(test_case["relevant_docs"])
            
            # Get retrieved documents
            retriever = self.rag.get_retriever()
            retrieved_docs = retriever.invoke(query)
            retrieved_sources = [doc.metadata.get("source", "") for doc in retrieved_docs]
            
            # Calculate metrics
            precision_k = self._precision_at_k(retrieved_sources, relevant_docs)
            recall_k = self._recall_at_k(retrieved_sources, relevant_docs)
            mrr = self._mean_reciprocal_rank(retrieved_sources, relevant_docs)
            ndcg = self._ndcg_at_k(retrieved_sources, relevant_docs)
            
            results["precision_at_k"].append(precision_k)
            results["recall_at_k"].append(recall_k)
            results["mrr"].append(mrr)
            results["ndcg"].append(ndcg)
        
        # Calculate averages
        return {
            "avg_precision_at_k": np.mean(results["precision_at_k"]),
            "avg_recall_at_k": np.mean(results["recall_at_k"]),
            "avg_mrr": np.mean(results["mrr"]),
            "avg_ndcg": np.mean(results["ndcg"]),
            "details": results
        }
    
    def benchmark_different_settings(self, test_queries: List[Dict]) -> pd.DataFrame:
        """Benchmark different retrieval settings"""
        results = []
        
        # Store original settings
        original_threshold = self.rag.get_threshold()
        original_rerank = self.rag.is_rerank
        original_top_n = self.rag.get_top_n()
        original_retrieve_num = self.rag.get_retrieve_num()
        
        try:
            # Test different thresholds
            thresholds = [0.0, 0.1, 0.3, 0.5, 0.7]
            for threshold in thresholds:
                self.rag.set_threshold(threshold)
                metrics = self.evaluate_retrieval(test_queries)
                results.append({
                    'setting_type': 'threshold',
                    'threshold': threshold,
                    'rerank': self.rag.is_rerank,
                    'top_n': self.rag.get_top_n(),
                    'retrieve_num': self.rag.get_retrieve_num(),
                    'precision': metrics['avg_precision_at_k'],
                    'recall': metrics['avg_recall_at_k'],
                    'mrr': metrics['avg_mrr'],
                    'ndcg': metrics['avg_ndcg']
                })
            
            # Reset threshold and test reranking
            self.rag.set_threshold(original_threshold)
            for rerank in [True, False]:
                self.rag.is_rerank = rerank
                metrics = self.evaluate_retrieval(test_queries)
                results.append({
                    'setting_type': 'rerank',
                    'threshold': original_threshold,
                    'rerank': rerank,
                    'top_n': self.rag.get_top_n(),
                    'retrieve_num': self.rag.get_retrieve_num(),
                    'precision': metrics['avg_precision_at_k'],
                    'recall': metrics['avg_recall_at_k'],
                    'mrr': metrics['avg_mrr'],
                    'ndcg': metrics['avg_ndcg']
                })
            
            # Reset rerank and test top_n values
            self.rag.is_rerank = original_rerank
            if original_rerank:  # Only test top_n if reranking is enabled
                top_n_values = [3, 5, 8, 10, 15]
                for top_n in top_n_values:
                    self.rag.set_top_n(top_n)
                    metrics = self.evaluate_retrieval(test_queries)
                    results.append({
                        'setting_type': 'top_n',
                        'threshold': original_threshold,
                        'rerank': self.rag.is_rerank,
                        'top_n': top_n,
                        'retrieve_num': self.rag.get_retrieve_num(),
                        'precision': metrics['avg_precision_at_k'],
                        'recall': metrics['avg_recall_at_k'],
                        'mrr': metrics['avg_mrr'],
                        'ndcg': metrics['avg_ndcg']
                    })
            
            # Reset top_n and test retrieve_num values
            self.rag.set_top_n(original_top_n)
            retrieve_nums = [10, 20, 30, 50]
            for retrieve_num in retrieve_nums:
                self.rag.set_retrieve_num(retrieve_num)
                metrics = self.evaluate_retrieval(test_queries)
                results.append({
                    'setting_type': 'retrieve_num',
                    'threshold': original_threshold,
                    'rerank': self.rag.is_rerank,
                    'top_n': self.rag.get_top_n(),
                    'retrieve_num': retrieve_num,
                    'precision': metrics['avg_precision_at_k'],
                    'recall': metrics['avg_recall_at_k'],
                    'mrr': metrics['avg_mrr'],
                    'ndcg': metrics['avg_ndcg']
                })
        
        finally:
            # Restore original settings
            self.rag.set_threshold(original_threshold)
            self.rag.is_rerank = original_rerank
            self.rag.set_top_n(original_top_n)
            self.rag.set_retrieve_num(original_retrieve_num)
        
        return pd.DataFrame(results)
    
    def _precision_at_k(self, retrieved: List[str], relevant: set) -> float:
        if not retrieved:
            return 0.0
        return len([doc for doc in retrieved if doc in relevant]) / len(retrieved)
    
    def _recall_at_k(self, retrieved: List[str], relevant: set) -> float:
        if not relevant:
            return 0.0
        return len([doc for doc in retrieved if doc in relevant]) / len(relevant)
    
    def _mean_reciprocal_rank(self, retrieved: List[str], relevant: set) -> float:
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def _ndcg_at_k(self, retrieved: List[str], relevant: set, k=None) -> float:
        if k is None:
            k = len(retrieved)
        
        # Binary relevance (1 if relevant, 0 if not)
        relevance_scores = [1 if doc in relevant else 0 for doc in retrieved[:k]]
        
        if sum(relevance_scores) == 0:
            return 0.0
        
        # DCG calculation
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
        
        # IDCG (ideal DCG)
        ideal_relevance = sorted([1] * len(relevant) + [0] * (k - len(relevant)), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0
    

    def create_test_queries(self):
        """Create test queries with known relevant documents"""
        test_queries = [
            # AI/ML Related Queries
            {
                "query": "What is machine learning?",
                "relevant_docs": ["Machine_learning_application_in_growth_and_health_prediction_of_broiler_chickens"],
                "domain": "all"
            },
            {
                "query": "How can machine learning be used for chicken health prediction?",
                "relevant_docs": ["Machine_learning_application_in_growth_and_health_prediction_of_broiler_chickens"],
                "domain": "all"
            },
            {
                "query": "Japan AI policy and white paper",
                "relevant_docs": ["The_Japan_s_AI_White_Paper_English_Translaiton__1684318555"],
                "domain": "all"
            },
            {
                "query": "What are Japan's artificial intelligence strategies?",
                "relevant_docs": ["The_Japan_s_AI_White_Paper_English_Translaiton__1684318555"],
                "domain": "all"
            },
            
            # Economic/Vietnam Related Queries
            {
                "query": "COVID-19 economic impact on Vietnam",
                "relevant_docs": ["2020_02_25_COVID-19's_Economic_Impact_on_Vietnam", "2020_04_06_COVID-19's_Economic_Impact_on_Vietnam_Intensifies"],
                "domain": "econ"
            },
            {
                "query": "Vietnam coronavirus economic effects 2020",
                "relevant_docs": ["2020_02_25_COVID-19's_Economic_Impact_on_Vietnam", "2020_04_06_COVID-19's_Economic_Impact_on_Vietnam_Intensifies"],
                "domain": "econ"
            },
            {
                "query": "VND depreciation analysis",
                "relevant_docs": ["2020_03_23_Comments_on_the_VND_Depreciation"],
                "domain": "econ"
            },
            {
                "query": "Vietnamese dong currency devaluation",
                "relevant_docs": ["2020_03_23_Comments_on_the_VND_Depreciation"],
                "domain": "econ"
            },
            {
                "query": "Vietnam flattened COVID curve",
                "relevant_docs": ["2020_04_17_Vietnam_has_\"Flattened_the_COVID_Curve\""],
                "domain": "econ"
            },
            {
                "query": "Vietnam exit strategy COVID-19",
                "relevant_docs": ["2020_04_21_Vietnam's_Exit_Strategy__Balancing_Medical_and_Economic_Health"],
                "domain": "econ"
            },
            {
                "query": "Foreign direct investment Vietnam FDI",
                "relevant_docs": ["2020_07_03_The_Next_Wave_of_FDI_to_Vietnam_is_Coming_–_Part_2", "2020_07_06_The_Next_Wave_of_FDI_to_Vietnam_is_Coming_–_Part_3", "2020_07_22_The_Next_Wave_of_FDI_to_Vietnam_is_Coming"],
                "domain": "econ"
            },
            {
                "query": "Attracting foreign institutional investors Vietnam",
                "relevant_docs": ["2020_05_28_Attracting_FII_to_Vietnam"],
                "domain": "econ"
            },
            {
                "query": "US tariffs Vietnam trade",
                "relevant_docs": ["2020_10_08_There's_Little_Risk_the_US_Will_Impose_Draconian_Tariffs"],
                "domain": "econ"
            },
            {
                "query": "US election impact Vietnam stock market VN-Index",
                "relevant_docs": ["2020_11_10_The_US_Election,_and_What's_Ahead_for_the_VN-Index_and_the_VN_Dong"],
                "domain": "econ"
            },
            {
                "query": "Chinese yuan appreciation Vietnam economy",
                "relevant_docs": ["2020_11_13_Chinese_Yuan_Appreciation_is_Good_for_Vietnam"],
                "domain": "econ"
            },
            {
                "query": "Vietnam stock market rally 2021",
                "relevant_docs": ["2020_12_08_Vietnam's_\"Covert_Stock_Market_Rally\"_is_Set_to_Continue_in_2021"],
                "domain": "econ"
            },
            {
                "query": "VN-Index ASEAN stock market performance",
                "relevant_docs": ["2020_12_23_The_VN-Index_is_Outperforming_ASEAN_Stock_Markets,_and_Headed_Higher_in_2021"],
                "domain": "econ"
            },
            
            # Technical/Equipment Queries  
            {
                "query": "Fuji scanner function guide",
                "relevant_docs": ["Hướng_dẫn_dùng_chức_năng_Scan_trên_máy_Fuji"],
                "domain": "all"
            },
            {
                "query": "How to use Fuji machine scanning feature",
                "relevant_docs": ["Hướng_dẫn_dùng_chức_năng_Scan_trên_máy_Fuji"],
                "domain": "all"
            },
            
            # Knowledge Management
            {
                "query": "knowledge centralization best practices",
                "relevant_docs": ["Knowledge_Centralization__up_to_21.04.2023"],
                "domain": "all"
            },
            
            # Cross-domain queries (should retrieve from multiple domains)
            {
                "query": "economic recovery strategies during pandemic",
                "relevant_docs": ["2020_04_21_Vietnam's_Exit_Strategy__Balancing_Medical_and_Economic_Health", "2020_04_17_Vietnam_has_\"Flattened_the_COVID_Curve\""],
                "domain": "all"
            },
            {
                "query": "technology and economic development",
                "relevant_docs": ["The_Japan_s_AI_White_Paper_English_Translaiton__1684318555", "Machine_learning_application_in_growth_and_health_prediction_of_broiler_chickens"],
                "domain": "all"
            },
            
            # Negative test cases (queries that shouldn't match any documents)
            {
                "query": "blockchain cryptocurrency regulations",
                "relevant_docs": [],
                "domain": "all"
            },
            {
                "query": "climate change environmental policy",
                "relevant_docs": [],
                "domain": "all"
            },
            
            # Domain-specific precision tests
            {
                "query": "investment opportunities",
                "relevant_docs": ["2020_05_28_Attracting_FII_to_Vietnam", "2020_07_22_The_Next_Wave_of_FDI_to_Vietnam_is_Coming"],
                "domain": "econ"
            },
            {
                "query": "artificial intelligence applications",
                "relevant_docs": ["The_Japan_s_AI_White_Paper_English_Translaiton__1684318555", "Machine_learning_application_in_growth_and_health_prediction_of_broiler_chickens"],
                "domain": "all"
            }
        ]
        
        # Save to file
        with open("data/test_queries.json", "w") as f:
            json.dump(test_queries, f, indent=2)
        
        return test_queries