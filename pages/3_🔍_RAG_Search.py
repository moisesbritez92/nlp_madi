"""
🔍 Interactive RAG Search Tool
==============================
Retrieval-Augmented Generation search application.
Supports SPLADE (semantic sparse) and BM25 (lexical) retrieval modes.
"""

import streamlit as st
import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.splade_retriever import SpladeRetriever
from rag.bm25_retriever import BM25Retriever
from rag.llm_client import LLMClient, get_default_llm_config
from rag.data_loader import get_demo_corpus, load_corpus, corpus_stats, normalize_corpus
from rag.prompts import build_rag_messages, format_prompt_for_display, get_fallback_response

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="🔍 RAG Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# Session State Initialization
# ============================================
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'corpus': None,
        'corpus_source': None,
        'splade_retriever': None,
        'bm25_retriever': None,
        'splade_indexed': False,
        'bm25_indexed': False,
        'last_results': None,
        'last_response': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================
# Cached Model Loading
# ============================================
@st.cache_resource(show_spinner=False)
def load_splade_model():
    """Load SPLADE model (cached)."""
    retriever = SpladeRetriever()
    success, error = retriever.load_model()
    return retriever, success, error


# ============================================
# Sidebar Configuration
# ============================================
def render_sidebar():
    """Render sidebar configuration options."""
    st.sidebar.title("⚙️ Configuration")
    
    # ----------------------------------------
    # Corpus Selection
    # ----------------------------------------
    st.sidebar.header("📚 Corpus")
    
    corpus_option = st.sidebar.radio(
        "Select corpus source:",
        ["Demo Corpus (50 docs)", "Upload JSON file"],
        key="corpus_option"
    )
    
    corpus = None
    corpus_source = None
    
    if corpus_option == "Demo Corpus (50 docs)":
        corpus = get_demo_corpus()
        corpus_source = "demo"
        st.sidebar.success(f"✅ Demo corpus loaded ({len(corpus)} docs)")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload corpus file:",
            type=["json", "jsonl", "txt"],
            help="JSON dict or list of documents"
        )
        
        if uploaded_file:
            try:
                import json
                content = uploaded_file.read().decode("utf-8")
                
                if uploaded_file.name.endswith(".json"):
                    data = json.loads(content)
                    corpus = normalize_corpus(data)
                elif uploaded_file.name.endswith(".jsonl"):
                    docs = [json.loads(line) for line in content.strip().split("\n") if line]
                    corpus = normalize_corpus(docs)
                else:  # .txt
                    corpus = {"doc1": {"title": uploaded_file.name, "text": content}}
                
                corpus_source = uploaded_file.name
                st.sidebar.success(f"✅ Loaded {len(corpus)} documents")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {e}")
    
    # Show corpus stats
    if corpus:
        stats = corpus_stats(corpus)
        st.sidebar.caption(f"📊 {stats['num_docs']} docs | {stats['avg_chars']:.0f} avg chars")
    
    st.sidebar.divider()
    
    # ----------------------------------------
    # Retrieval Settings
    # ----------------------------------------
    st.sidebar.header("🔎 Retrieval")
    
    retrieval_mode = st.sidebar.selectbox(
        "Retrieval method:",
        ["SPLADE (Semantic Sparse)", "BM25 (Lexical)"],
        help="SPLADE: Context-aware term expansion\nBM25: Classic keyword matching"
    )
    
    topk = st.sidebar.slider(
        "Top-K documents:",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of documents to retrieve"
    )
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        topk_keep_doc = st.number_input(
            "SPLADE terms per doc:",
            min_value=50,
            max_value=200,
            value=120,
            help="Number of sparse terms to keep per document"
        )
        topk_keep_query = st.number_input(
            "SPLADE terms per query:",
            min_value=32,
            max_value=128,
            value=64,
            help="Number of sparse terms to keep per query"
        )
    
    st.sidebar.divider()
    
    # ----------------------------------------
    # LLM Configuration
    # ----------------------------------------
    st.sidebar.header("🤖 LLM Settings")
    
    llm_mode = st.sidebar.selectbox(
        "LLM backend:",
        ["Local/HTTP (Ollama)", "OpenAI API"],
        help="Choose LLM inference backend"
    )
    
    if llm_mode == "Local/HTTP (Ollama)":
        llm_url = st.sidebar.text_input(
            "API URL:",
            value="http://localhost:11434/api/chat",
            help="Ollama or compatible API endpoint"
        )
        llm_model = st.sidebar.text_input(
            "Model name:",
            value="gemma3",
            help="Model to use (e.g., gemma3, llama2, mistral)"
        )
        llm_config = {
            "mode": "local",
            "url": llm_url,
            "model": llm_model,
            "timeout": 120
        }
    else:
        openai_key = st.sidebar.text_input(
            "OpenAI API Key:",
            type="password",
            help="Your OpenAI API key"
        )
        llm_model = st.sidebar.selectbox(
            "Model:",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            help="OpenAI model to use"
        )
        llm_config = {
            "mode": "openai",
            "model": llm_model,
            "openai_api_key": openai_key,
            "timeout": 60
        }
    
    # Test connection button
    if st.sidebar.button("🔗 Test LLM Connection"):
        with st.sidebar:
            with st.spinner("Testing..."):
                client = LLMClient(**llm_config)
                success, msg = client.test_connection()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
    
    st.sidebar.divider()
    
    # ----------------------------------------
    # Display Options
    # ----------------------------------------
    st.sidebar.header("👁️ Display")
    
    show_prompt = st.sidebar.checkbox(
        "Show full prompt",
        value=False,
        help="Display the complete prompt sent to LLM"
    )
    
    show_scores = st.sidebar.checkbox(
        "Show retrieval scores",
        value=True,
        help="Display relevance scores for retrieved docs"
    )
    
    return {
        "corpus": corpus,
        "corpus_source": corpus_source,
        "retrieval_mode": "splade" if "SPLADE" in retrieval_mode else "bm25",
        "topk": topk,
        "topk_keep_doc": topk_keep_doc,
        "topk_keep_query": topk_keep_query,
        "llm_config": llm_config,
        "show_prompt": show_prompt,
        "show_scores": show_scores,
    }


# ============================================
# Main Search Interface
# ============================================
def render_search_interface(config):
    """Render the main search interface."""
    
    st.title("🔍 Interactive RAG Search Tool")
    
    st.markdown("""
    **Retrieval-Augmented Generation** search tool combining:
    - **SPLADE**: Semantic sparse retrieval with context-aware term expansion
    - **BM25**: Classic lexical retrieval with TF-IDF saturation
    - **LLM**: Generate answers grounded in retrieved documents
    """)
    
    # Check corpus
    if not config["corpus"]:
        st.warning("⚠️ Please select a corpus in the sidebar to begin.")
        return
    
    st.divider()
    
    # ----------------------------------------
    # Query Input
    # ----------------------------------------
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "🔎 Enter your query:",
            placeholder="e.g., What is the difference between TF IDF and BM25 scoring?",
            height=100,
            key="query_input"
        )
    
    with col2:
        st.markdown("**Retrieval Mode:**")
        mode_emoji = "🧠" if config["retrieval_mode"] == "splade" else "📝"
        mode_name = "SPLADE (Semantic)" if config["retrieval_mode"] == "splade" else "BM25 (Lexical)"
        st.info(f"{mode_emoji} {mode_name}")
        
        st.markdown(f"**Top-K:** {config['topk']} docs")
    
    # Search button
    search_clicked = st.button("🚀 Search & Generate", type="primary", use_container_width=True)
    
    if not search_clicked:
        # Show example queries
        with st.expander("💡 Example Queries"):
            st.markdown("""
            **Information Retrieval:**
            - What is the difference between TF IDF and BM25 scoring?
            - How does SPLADE create sparse query expansion terms?
            - What is retrieval augmented generation and why is it useful?
            
            **Semantic Understanding:**
            - Any document talking about Spanish products?
            - How do chunking and overlap stride help long document retrieval?
            
            **No Evidence Test:**
            - What is quantum computing? (not in corpus)
            """)
        return
    
    # Validate query
    if not query.strip():
        st.warning("⚠️ Please enter a query.")
        return
    
    # ----------------------------------------
    # Execute Search Pipeline
    # ----------------------------------------
    st.divider()
    
    # Metrics placeholders
    col_m1, col_m2, col_m3 = st.columns(3)
    metric_index = col_m1.empty()
    metric_retrieval = col_m2.empty()
    metric_llm = col_m3.empty()
    
    results_container = st.container()
    
    with st.spinner("Processing..."):
        corpus = config["corpus"]
        retrieval_mode = config["retrieval_mode"]
        topk = config["topk"]
        
        # ========================================
        # Step 1: Build/Load Index
        # ========================================
        index_time = 0.0
        retriever = None
        
        if retrieval_mode == "splade":
            # Check if we need to rebuild index
            if (st.session_state.splade_retriever is None or 
                st.session_state.corpus_source != config["corpus_source"]):
                
                with st.status("🔧 Loading SPLADE model...", expanded=True) as status:
                    st.write("Loading tokenizer and MLM model...")
                    retriever, success, error = load_splade_model()
                    
                    if not success:
                        st.error(f"❌ Failed to load SPLADE: {error}")
                        return
                    
                    st.write("✅ Model loaded")
                    
                    # Update retriever settings
                    retriever.topk_keep_doc = config["topk_keep_doc"]
                    retriever.topk_keep_query = config["topk_keep_query"]
                    
                    st.write(f"Building index for {len(corpus)} documents...")
                    
                    progress_bar = st.progress(0)
                    
                    def update_progress(current, total):
                        progress_bar.progress(current / total)
                    
                    index_time, error = retriever.build_index(corpus, progress_callback=update_progress)
                    
                    if error:
                        st.error(f"❌ Indexing failed: {error}")
                        return
                    
                    st.write(f"✅ Index built in {index_time:.2f}s")
                    status.update(label="✅ SPLADE ready", state="complete")
                    
                    # Cache retriever
                    st.session_state.splade_retriever = retriever
                    st.session_state.corpus_source = config["corpus_source"]
            else:
                retriever = st.session_state.splade_retriever
                
        else:  # BM25
            if (st.session_state.bm25_retriever is None or 
                st.session_state.corpus_source != config["corpus_source"]):
                
                with st.status("🔧 Building BM25 index...", expanded=False) as status:
                    retriever = BM25Retriever()
                    index_time, error = retriever.build_index(corpus)
                    
                    if error:
                        st.error(f"❌ Indexing failed: {error}")
                        return
                    
                    status.update(label=f"✅ BM25 indexed ({index_time:.2f}s)", state="complete")
                    
                    st.session_state.bm25_retriever = retriever
                    st.session_state.corpus_source = config["corpus_source"]
            else:
                retriever = st.session_state.bm25_retriever
        
        metric_index.metric("📊 Index Time", f"{index_time:.2f}s" if index_time > 0 else "cached")
        
        # ========================================
        # Step 2: Retrieve Documents
        # ========================================
        results, retrieval_time = retriever.retrieve(query, topk=topk)
        
        metric_retrieval.metric("🔎 Retrieval Time", f"{retrieval_time:.3f}s")
        
        if not results:
            st.warning("⚠️ No documents retrieved for this query.")
            return
        
        # Prepare documents for LLM
        retrieved_docs = []
        for doc_id, score in results:
            doc = retriever.get_document(doc_id)
            if doc:
                retrieved_docs.append((doc_id, doc, score))
        
        # ========================================
        # Step 3: Generate LLM Response
        # ========================================
        messages = build_rag_messages(query, retrieved_docs)
        
        llm_client = LLMClient(**config["llm_config"])
        response, llm_time, llm_error = llm_client.call(messages)
        
        metric_llm.metric("🤖 LLM Time", f"{llm_time:.2f}s")
        
        # ========================================
        # Display Results
        # ========================================
        with results_container:
            # LLM Response
            st.subheader("💬 Generated Answer")
            
            if llm_error:
                st.error(f"LLM Error: {llm_error}")
                fallback = get_fallback_response(retrieved_docs, llm_error)
                st.markdown(fallback)
            else:
                st.markdown(response)
            
            st.divider()
            
            # Retrieved Documents
            st.subheader("📄 Retrieved Documents")
            
            for i, (doc_id, doc, score) in enumerate(retrieved_docs):
                title = doc.get("title", "")
                text = doc.get("text", "")
                
                score_str = f" (score: {score:.4f})" if config["show_scores"] else ""
                
                with st.expander(f"**[{doc_id}]**{score_str} {title[:50] if title else text[:50]}..."):
                    if title:
                        st.markdown(f"**Title:** {title}")
                    st.markdown(f"**Text:** {text}")
            
            # Show prompt (debug)
            if config["show_prompt"]:
                st.divider()
                st.subheader("🔧 Debug: Full Prompt")
                prompt_text = format_prompt_for_display(messages)
                st.code(prompt_text, language="markdown")


# ============================================
# Main Entry Point
# ============================================
def main():
    init_session_state()
    config = render_sidebar()
    render_search_interface(config)
    
    # Footer
    st.divider()
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        **Interactive RAG Search Tool**
        
        This application demonstrates Retrieval-Augmented Generation (RAG):
        
        1. **Retrieval**: Find relevant documents using:
           - **SPLADE**: Sparse semantic search with neural term expansion
           - **BM25**: Classic lexical matching with TF-IDF saturation
        
        2. **Augmentation**: Pass retrieved documents to an LLM as context
        
        3. **Generation**: LLM generates an answer grounded in the evidence
        
        **Key Features:**
        - Semantic vs lexical retrieval comparison
        - Configurable Top-K document selection
        - Source citations in LLM responses
        - Fallback handling when LLM fails
        
        **Based on:**
        - SPLADE: Sparse Lexical and Expansion Model (Formal et al., 2021)
        - BM25: Best Matching 25 (Robertson & Zaragoza, 2009)
        """)


if __name__ == "__main__":
    main()
