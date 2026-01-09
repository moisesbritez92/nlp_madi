"""
SPLADE Retriever
================
Sparse semantic retrieval using SPLADE (Sparse Lexical and Expansion Model).
Based on masked language modeling for context-aware term expansion.
"""

import torch
from collections import defaultdict
from math import sqrt
import heapq
from typing import Dict, List, Tuple, Optional
import time


class SpladeRetriever:
    """
    SPLADE-based sparse semantic retriever.
    
    Uses masked language modeling to create sparse vectors with
    context-aware term expansion for semantic matching.
    """
    
    MODEL_NAME = "naver/efficient-splade-VI-BT-large-doc"
    
    def __init__(self, topk_keep_doc: int = 120, topk_keep_query: int = 64):
        """
        Initialize SPLADE retriever.
        
        Args:
            topk_keep_doc: Number of top terms to keep per document
            topk_keep_query: Number of top terms to keep per query
        """
        self.topk_keep_doc = topk_keep_doc
        self.topk_keep_query = topk_keep_query
        
        self.tokenizer = None
        self.model = None
        self.device = None
        
        self.inverted_index = None
        self.doc_norms = None
        self.corpus = None
        
        self._is_loaded = False
        self._is_indexed = False
    
    def load_model(self) -> Tuple[bool, Optional[str]]:
        """
        Load SPLADE model and tokenizer.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForMaskedLM.from_pretrained(self.MODEL_NAME)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            self._is_loaded = True
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    @property
    def is_indexed(self) -> bool:
        return self._is_indexed
    
    @torch.no_grad()
    def _splade_pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply SPLADE pooling: log(1 + ReLU(logits)) with max pooling.
        
        Args:
            input_ids: Token IDs tensor
            attention_mask: Attention mask tensor
            
        Returns:
            Pooled sparse vector over vocabulary
        """
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        
        # Apply log(1 + ReLU(x)) activation
        act = torch.log1p(torch.relu(logits))
        
        # Mask padded positions
        mask = attention_mask.unsqueeze(-1)
        act = act * mask
        
        # Max pooling over sequence dimension
        pooled = act.max(dim=1).values
        
        return pooled
    
    def _doc_to_chunks(
        self, 
        title: str, 
        text: str, 
        max_len: int = 480, 
        stride: int = 416
    ) -> List[Dict]:
        """
        Split a long document into overlapping chunks.
        
        Args:
            title: Document title
            text: Document text
            max_len: Maximum chunk length in tokens
            stride: Step size between chunks
            
        Returns:
            List of chunk dictionaries with input_ids and attention_mask
        """
        s = (title.strip() + " " + text.strip()).strip()
        enc = self.tokenizer(s, return_tensors="pt", truncation=False, add_special_tokens=True)
        ids = enc["input_ids"][0]
        
        total_len = len(ids)
        if total_len == 0:
            return []
        
        # Calculate chunk start positions
        starts = []
        start = 0
        while start < total_len:
            starts.append(start)
            if start + max_len >= total_len:
                break
            start += stride
        
        # Create chunks
        chunks = []
        for start in starts:
            end = min(start + max_len, total_len)
            chunk_ids = ids[start:end]
            chunk_att = torch.ones_like(chunk_ids)
            chunks.append({
                "input_ids": chunk_ids,
                "attention_mask": chunk_att
            })
        
        return chunks
    
    @torch.no_grad()
    def _encode_document(self, title: str, text: str) -> Dict[int, float]:
        """
        Encode a document into a sparse SPLADE vector.
        
        Args:
            title: Document title
            text: Document text
            
        Returns:
            Sparse vector as dict {term_id: weight}
        """
        chunks = self._doc_to_chunks(title, text)
        if not chunks:
            return {}
        
        # Batch encode all chunks
        batch_ids = torch.stack([c["input_ids"] for c in chunks]).to(self.device)
        batch_att = torch.stack([c["attention_mask"] for c in chunks]).to(self.device)
        
        # Get SPLADE vectors for each chunk
        per_chunk = self._splade_pool(batch_ids, batch_att)  # (num_chunks, vocab_size)
        
        # Max pool across chunks
        doc_vec = per_chunk.max(dim=0).values  # (vocab_size,)
        
        # Keep top-k terms
        vals, idx = torch.topk(doc_vec, k=min(self.topk_keep_doc, doc_vec.numel()))
        vals, idx = vals[vals > 0], idx[vals > 0]
        
        return {int(i): float(v) for i, v in zip(idx, vals)}
    
    @torch.no_grad()
    def _encode_query(self, query: str) -> Dict[int, float]:
        """
        Encode a query into a sparse SPLADE vector.
        
        Args:
            query: Query string
            
        Returns:
            Sparse vector as dict {term_id: weight}
        """
        enc = self.tokenizer(
            query, 
            return_tensors="pt", 
            truncation=True, 
            max_length=64
        ).to(self.device)
        
        qv = self._splade_pool(enc["input_ids"], enc["attention_mask"])[0]
        
        vals, idx = torch.topk(qv, k=min(self.topk_keep_query, qv.numel()))
        vals, idx = vals[vals > 0], idx[vals > 0]
        
        return {int(i): float(v) for i, v in zip(idx, vals)}
    
    def build_index(
        self, 
        corpus: Dict[str, Dict], 
        progress_callback=None
    ) -> Tuple[float, Optional[str]]:
        """
        Build inverted index from corpus.
        
        Args:
            corpus: Dict of {doc_id: {"title": ..., "text": ...}}
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            Tuple of (time_seconds, error_message)
        """
        if not self._is_loaded:
            return 0.0, "Model not loaded. Call load_model() first."
        
        start_time = time.time()
        
        try:
            self.corpus = corpus
            self.inverted_index = defaultdict(list)
            self.doc_norms = {}
            
            total = len(corpus)
            
            for i, (doc_id, doc) in enumerate(corpus.items()):
                title = doc.get("title", "")
                text = doc.get("text", "")
                
                # Encode document
                doc_vec = self._encode_document(title, text)
                
                # Calculate norm
                norm = sqrt(sum(v*v for v in doc_vec.values())) or 1.0
                self.doc_norms[doc_id] = norm
                
                # Add to inverted index
                for term_id, weight in doc_vec.items():
                    self.inverted_index[term_id].append((doc_id, weight))
                
                if progress_callback:
                    progress_callback(i + 1, total)
            
            self._is_indexed = True
            elapsed = time.time() - start_time
            return elapsed, None
            
        except Exception as e:
            return 0.0, str(e)
    
    def retrieve(
        self, 
        query: str, 
        topk: int = 10, 
        normalize: bool = True
    ) -> Tuple[List[Tuple[str, float]], float]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query string
            topk: Number of documents to retrieve
            normalize: Whether to apply cosine normalization
            
        Returns:
            Tuple of (list of (doc_id, score), retrieval_time)
        """
        if not self._is_indexed:
            return [], 0.0
        
        start_time = time.time()
        
        # Encode query
        q_vec = self._encode_query(query)
        q_norm = sqrt(sum(v*v for v in q_vec.values())) or 1.0
        
        # Accumulate scores
        scores = defaultdict(float)
        
        for term_id, q_weight in q_vec.items():
            postings = self.inverted_index.get(term_id, [])
            for doc_id, d_weight in postings:
                scores[doc_id] += q_weight * d_weight
        
        # Normalize by cosine similarity
        if normalize and scores:
            for doc_id in scores:
                scores[doc_id] /= (q_norm * self.doc_norms.get(doc_id, 1.0))
        
        # Get top-k
        results = heapq.nlargest(topk, scores.items(), key=lambda x: x[1])
        
        elapsed = time.time() - start_time
        return results, elapsed
    
    def decode_sparse_vector(self, vec: Dict[int, float], topn: int = 20) -> List[Tuple[str, float]]:
        """
        Decode sparse vector term IDs to tokens.
        
        Args:
            vec: Sparse vector {term_id: weight}
            topn: Number of top terms to return
            
        Returns:
            List of (token, weight) tuples
        """
        if not self.tokenizer:
            return []
        
        pairs = sorted(vec.items(), key=lambda x: -x[1])[:topn]
        return [
            (self.tokenizer.convert_ids_to_tokens([tid])[0], w) 
            for tid, w in pairs
        ]
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID from indexed corpus."""
        if self.corpus:
            return self.corpus.get(doc_id)
        return None
