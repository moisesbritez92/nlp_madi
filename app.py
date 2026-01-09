"""
🧠 NLP MADI - Home
==================
Aplicación multipage de NLP con Streamlit.

Ejecutar desde Terminal de JupyterLab:
    streamlit run app.py
"""

import streamlit as st

# ============================================
# Configuración de página
# ============================================
st.set_page_config(
    page_title="🧠 NLP MADI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================
# Página principal
# ============================================
def main():
    st.title("🧠 NLP MADI")
    
    st.markdown("""
    ## Bienvenido a la aplicación de Procesamiento de Lenguaje Natural
    
    Esta aplicación contiene herramientas de NLP desarrolladas para el curso MADI.
    """)
    
    st.divider()
    
    # ----------------------------------------
    # Tarjetas de navegación
    # ----------------------------------------
    st.subheader("📚 Herramientas disponibles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🖼️ Image to Speech
        
        Convierte imágenes en audio hablado:
        - Genera descripciones con **BLIP**
        - Síntesis de voz con **Kokoro TTS**
        - Descarga el audio generado
        
        👈 Selecciona en el menú lateral
        """)
    
    with col2:
        st.markdown("""
        ### 📝 Análisis de Sentimientos
        
        Analiza comentarios y reviews:
        - Detecta sentimiento **positivo/negativo/neutral**
        - Clasifica en escala **1-5 estrellas**
        - Identifica **emociones** específicas
        
        👈 Selecciona en el menú lateral
        """)
    
    # Segunda fila de herramientas
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        ### 🔍 RAG Search
        
        Búsqueda semántica con IA:
        - **SPLADE**: Expansión semántica de términos
        - **BM25**: Búsqueda léxica clásica
        - Respuestas generadas por **LLM**
        - Citación de fuentes automática
        
        👈 Selecciona en el menú lateral
        """)
    
    st.divider()
    
    # ----------------------------------------
    # Información del proyecto
    # ----------------------------------------
    with st.expander("ℹ️ Información del proyecto"):
        st.markdown("""
        **Tecnologías utilizadas:**
        - 🐍 Python 3.10+
        - 🎈 Streamlit
        - 🤗 Transformers (Hugging Face)
        - 🔊 Kokoro TTS
        - 🐳 Docker
        
        **Ejecución:**
        ```bash
        # Con Docker
        docker-compose up -d
        
        # Acceder a:
        # - Streamlit: http://localhost:8501
        # - JupyterLab: http://localhost:18888
        ```
        
        **Estructura del proyecto:**
        ```
        nlp_madi/
        ├── app.py              # Esta página (Home)
        ├── pages/
        │   ├── 1_🖼️_Image_to_Speech.py
        │   ├── 2_📝_Analisis_Sentimientos.py
        │   └── 3_🔍_RAG_Search.py
        ├── rag/                # Módulos RAG
        │   ├── splade_retriever.py
        │   ├── bm25_retriever.py
        │   ├── llm_client.py
        │   ├── data_loader.py
        │   └── prompts.py
        ├── Dockerfile
        └── docker-compose.yml
        ```
        """)
    
    # ----------------------------------------
    # Footer
    # ----------------------------------------
    st.divider()
    st.caption("🎓 Desarrollado para MADI - NLP")


# ============================================
# Entry point
# ============================================
if __name__ == "__main__":
    main()
