"""
BM25 Retriever
==============
Classic lexical retrieval using BM25 (Best Matching 25) algorithm.
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import time
import re


class BM25Retriever:
    """
    BM25-based lexical retriever.
    
    Uses term frequency with saturation and document length normalization
    for classic keyword-based retrieval.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter (default 1.5)
            b: Document length normalization parameter (default 0.75)
        """
        self.k1 = k1
        self.b = b
        
        self.corpus = None
        self.doc_ids = []
        self.doc_lengths = {}
        self.avg_doc_length = 0.0
        self.doc_freqs = defaultdict(int)  # term -> number of docs containing term
        self.inverted_index = defaultdict(dict)  # term -> {doc_id: term_freq}
        self.total_docs = 0
        
        self._is_indexed = False
    
    @property
    def is_indexed(self) -> bool:
        return self._is_indexed
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase, split on non-alphanumeric.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Lowercase and split on non-alphanumeric characters
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, int]:
        """
        Compute term frequencies for a list of tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dict of {term: frequency}
        """
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        return dict(tf)
    
    def build_index(
        self, 
        corpus: Dict[str, Dict], 
        progress_callback=None
    ) -> Tuple[float, Optional[str]]:
        """
        Build BM25 index from corpus.
        
        Args:
            corpus: Dict of {doc_id: {"title": ..., "text": ...}}
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            Tuple of (time_seconds, error_message)
        """
        start_time = time.time()
        
        try:
            self.corpus = corpus
            self.doc_ids = list(corpus.keys())
            self.total_docs = len(corpus)
            
            total_length = 0
            
            for i, (doc_id, doc) in enumerate(corpus.items()):
                title = doc.get("title", "")
                text = doc.get("text", "")
                full_text = f"{title} {text}".strip()
                
                # Tokenize
                tokens = self._tokenize(full_text)
                self.doc_lengths[doc_id] = len(tokens)
                total_length += len(tokens)
                
                # Compute term frequencies
                tf = self._compute_tf(tokens)
                
                # Update inverted index and document frequencies
                for term, freq in tf.items():
                    if term not in self.inverted_index or doc_id not in self.inverted_index[term]:
                        self.doc_freqs[term] += 1
                    self.inverted_index[term][doc_id] = freq
                
                if progress_callback:
                    progress_callback(i + 1, self.total_docs)
            
            # Compute average document length
            self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0
            
            self._is_indexed = True
            elapsed = time.time() - start_time
            return elapsed, None
            
        except Exception as e:
            return 0.0, str(e)
    
    def _idf(self, term: str) -> float:
        """
        Compute IDF (Inverse Document Frequency) for a term.
        
        Uses the standard BM25 IDF formula:
        IDF = log((N - df + 0.5) / (df + 0.5))
        
        Args:
            term: Term to compute IDF for
            
        Returns:
            IDF score
        """
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        
        # BM25 IDF formula
        idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
        return max(0, idf)  # Ensure non-negative
    
    def _score_document(self, query_terms: List[str], doc_id: str) -> float:
        """
        Compute BM25 score for a document given query terms.
        
        Args:
            query_terms: List of query tokens
            doc_id: Document ID
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_length = self.doc_lengths.get(doc_id, 0)
        
        if doc_length == 0:
            return 0.0
        
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            
            tf = self.inverted_index[term].get(doc_id, 0)
            if tf == 0:
                continue
            
            idf = self._idf(term)
            
            # BM25 scoring formula
            # score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            
            score += idf * numerator / denominator
        
        return score
    
    def retrieve(
        self, 
        query: str, 
        topk: int = 10
    ) -> Tuple[List[Tuple[str, float]], float]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query string
            topk: Number of documents to retrieve
            
        Returns:
            Tuple of (list of (doc_id, score), retrieval_time)
        """
        if not self._is_indexed:
            return [], 0.0
        
        start_time = time.time()
        
        # Tokenize query
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return [], time.time() - start_time
        
        # Find candidate documents (docs containing at least one query term)
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term].keys())
        
        # Score all candidates
        scores = []
        for doc_id in candidate_docs:
            score = self._score_document(query_terms, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        
        # Sort by score and get top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = scores[:topk]
        
        elapsed = time.time() - start_time
        return results, elapsed
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID from indexed corpus."""
        if self.corpus:
            return self.corpus.get(doc_id)
        return None
    
    def get_query_terms(self, query: str) -> List[str]:
        """Get tokenized query terms."""
        return self._tokenize(query)
