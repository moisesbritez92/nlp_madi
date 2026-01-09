"""
Data Loader
===========
Functions for loading and normalizing document corpora.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union


# ============================================
# Demo Corpus (for quick testing)
# ============================================
DEMO_CORPUS = {
    "doc1": "Machine learning models learn patterns from data and features",
    "doc2": "Artificial intelligence includes reasoning planning and perception in machines",
    "doc3": "Data science combines statistics programming and domain knowledge for insights",
    "doc4": "Deep learning uses neural networks with many layers for vision",
    "doc5": "TF IDF weighs frequent terms and downweights common corpus words",
    "doc6": "BM25 improves tf idf with saturation and length normalization",
    "doc7": "Cosine similarity compares vector directions for document ranking tasks",
    "doc8": "Information retrieval finds relevant documents for a user query quickly",
    "doc9": "Retrieval augmented generation combines search with a language model answer",
    "doc10": "SPLADE produces sparse expansions using masked language modeling logits",
    "doc11": "Indexing builds an inverted index mapping terms to document postings",
    "doc12": "Tokenization splits text into tokens sometimes using subword pieces",
    "doc13": "Stopwords like the and is can be removed to reduce noise",
    "doc14": "Stemming reduces words to roots like compute computing computed",
    "doc15": "Lemmatization maps words to dictionary forms using part of speech",
    "doc16": "Precision measures fraction of retrieved documents that are relevant",
    "doc17": "Recall measures fraction of relevant documents that are retrieved",
    "doc18": "Mean average precision summarizes ranking quality across many queries",
    "doc19": "Vector databases store embeddings for fast nearest neighbor search",
    "doc20": "Dense retrieval uses neural embeddings rather than exact term matching",
    "doc21": "Sparse retrieval relies on term weights and inverted index structures",
    "doc22": "Hybrid search combines sparse and dense signals for better recall",
    "doc23": "Query expansion adds related terms to improve recall in search",
    "doc24": "Synonyms can help retrieval but may introduce ambiguity in results",
    "doc25": "Ranking functions score documents based on term matches and weights",
    "doc26": "Chunking splits long documents into windows to fit model limits",
    "doc27": "Overlap stride keeps context continuity between adjacent text chunks",
    "doc28": "Normalization divides by vector norms to compute cosine style scores",
    "doc29": "Evaluation needs labeled relevance judgments from human assessors",
    "doc30": "A chatbot answers questions by retrieving evidence and generating text",
    "doc31": "Clinical notes require careful privacy handling and access control",
    "doc32": "Pipelines log steps outputs and metadata for reproducible analysis",
    "doc33": "Docker containers package apps with dependencies for portable deployment",
    "doc34": "Shiny apps serve interactive R dashboards through a web browser",
    "doc35": "GPU acceleration speeds up neural inference for large transformer models",
    "doc36": "Caching model files avoids repeated downloads and improves startup time",
    "doc37": "Attention mechanisms weigh token interactions in transformer encoders",
    "doc38": "Masked language modeling predicts missing tokens to learn representations",
    "doc39": "Distillation trains smaller models to mimic larger teacher outputs",
    "doc40": "Out of vocabulary terms may be split into multiple subword tokens",
    "doc41": "Relevance feedback lets users refine results and correct mismatches",
    "doc42": "Inverted index stores postings lists for each term in vocabulary",
    "doc43": "Document length affects BM25 scoring through normalization parameters",
    "doc44": "Term frequency saturation prevents long documents dominating rankings",
    "doc45": "RAG responses should cite sources and avoid hallucinating unsupported facts",
    "doc46": "Paella and tortilla are popular Spanish foods often served at lunch",
    "doc47": "Pizza and pasta are common Italian dishes with many regional variants",
    "doc48": "Running and cycling improve cardio fitness and overall health benefits",
    "doc49": "San Sebastian is known for pintxos beaches and culinary culture",
    "doc50": "Statistics includes hypothesis testing confidence intervals and p values",
}


def normalize_corpus(corpus: Union[Dict, list]) -> Dict[str, Dict]:
    """
    Normalize corpus to standard format: {doc_id: {"title": ..., "text": ...}}
    
    Handles various input formats:
    - Dict {doc_id: str} -> {doc_id: {"title": "", "text": str}}
    - Dict {doc_id: {"text": ...}} -> adds empty title if missing
    - List of strings -> auto-generates doc IDs
    - List of dicts -> extracts text/content fields
    
    Args:
        corpus: Input corpus in various formats
        
    Returns:
        Normalized corpus dict
    """
    normalized = {}
    
    if isinstance(corpus, list):
        # List of documents
        for i, doc in enumerate(corpus):
            doc_id = f"doc{i+1}"
            
            if isinstance(doc, str):
                normalized[doc_id] = {"title": "", "text": doc}
            elif isinstance(doc, dict):
                text = doc.get("text") or doc.get("content") or doc.get("body") or ""
                title = doc.get("title") or doc.get("name") or ""
                doc_id = doc.get("id") or doc.get("doc_id") or doc_id
                normalized[doc_id] = {"title": str(title), "text": str(text)}
            else:
                normalized[doc_id] = {"title": "", "text": str(doc)}
    
    elif isinstance(corpus, dict):
        # Dict of documents
        for doc_id, doc in corpus.items():
            if isinstance(doc, str):
                normalized[doc_id] = {"title": "", "text": doc}
            elif isinstance(doc, dict):
                text = doc.get("text") or doc.get("content") or doc.get("body") or ""
                title = doc.get("title") or doc.get("name") or ""
                normalized[doc_id] = {"title": str(title), "text": str(text)}
            else:
                normalized[doc_id] = {"title": "", "text": str(doc)}
    
    return normalized


def load_corpus(file_path: Union[str, Path]) -> tuple:
    """
    Load corpus from a file.
    
    Supported formats:
    - .json: Dict or list of documents
    - .txt: Single document (entire file content)
    - .jsonl: JSON Lines format (one JSON object per line)
    
    Args:
        file_path: Path to the corpus file
        
    Returns:
        Tuple of (normalized_corpus, error_message)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None, f"File not found: {file_path}"
    
    try:
        suffix = file_path.suffix.lower()
        
        if suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return normalize_corpus(data), None
        
        elif suffix == ".jsonl":
            documents = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        documents.append(json.loads(line))
            return normalize_corpus(documents), None
        
        elif suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"doc1": {"title": file_path.name, "text": content}}, None
        
        else:
            return None, f"Unsupported file format: {suffix}"
    
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {e}"
    except Exception as e:
        return None, f"Error loading file: {e}"


def get_demo_corpus() -> Dict[str, Dict]:
    """
    Get the demo corpus in normalized format.
    
    Returns:
        Normalized demo corpus
    """
    return normalize_corpus(DEMO_CORPUS)


def corpus_stats(corpus: Dict[str, Dict]) -> Dict:
    """
    Calculate basic statistics about a corpus.
    
    Args:
        corpus: Normalized corpus
        
    Returns:
        Dict with statistics
    """
    if not corpus:
        return {"num_docs": 0, "total_chars": 0, "avg_chars": 0}
    
    num_docs = len(corpus)
    total_chars = sum(
        len(doc.get("title", "")) + len(doc.get("text", ""))
        for doc in corpus.values()
    )
    avg_chars = total_chars / num_docs if num_docs > 0 else 0
    
    return {
        "num_docs": num_docs,
        "total_chars": total_chars,
        "avg_chars": round(avg_chars, 1)
    }
