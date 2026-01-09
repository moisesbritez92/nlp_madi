"""
RAG (Retrieval-Augmented Generation) Module
============================================
Módulos para búsqueda semántica con SPLADE, BM25 y generación con LLM.
"""

from .splade_retriever import SpladeRetriever
from .bm25_retriever import BM25Retriever
from .llm_client import LLMClient
from .data_loader import load_corpus, normalize_corpus, DEMO_CORPUS
from .prompts import SYSTEM_PROMPT, format_documents_for_prompt

__all__ = [
    'SpladeRetriever',
    'BM25Retriever', 
    'LLMClient',
    'load_corpus',
    'normalize_corpus',
    'DEMO_CORPUS',
    'SYSTEM_PROMPT',
    'format_documents_for_prompt'
]
