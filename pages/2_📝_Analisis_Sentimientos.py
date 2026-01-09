"""
📝 Análisis de Sentimientos
===========================
Analiza el sentimiento de comentarios y reviews usando
modelos de Hugging Face Transformers.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict

# ============================================
# Configuración de página
# ============================================
st.set_page_config(
    page_title="📝 Análisis de Sentimientos",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================
# Carga de modelos con caché
# ============================================
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    """
    Carga el modelo de análisis de sentimientos.
    Usa un modelo multilingüe que funciona bien con reviews (1-5 estrellas).
    """
    try:
        from transformers import pipeline
        
        # Modelo para reviews (1-5 estrellas) - soporta múltiples idiomas
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            truncation=True,
            max_length=512
        )
        return sentiment_pipeline, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_emotion_model():
    """
    Carga modelo para detectar emociones específicas.
    """
    try:
        from transformers import pipeline
        
        emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            truncation=True,
            max_length=512
        )
        return emotion_pipeline, None
    except Exception as e:
        return None, str(e)


# ============================================
# Funciones de análisis
# ============================================
def analyze_sentiment(text: str, pipeline) -> Dict:
    """
    Analiza el sentimiento de un texto.
    
    Returns:
        Dict con label (1-5 estrellas) y score
    """
    result = pipeline(text)[0]
    
    # Convertir label "X stars" a número
    stars = int(result['label'].split()[0])
    
    # Mapear a categoría
    if stars <= 2:
        category = "Negativo"
        emoji = "😞"
        color = "red"
    elif stars == 3:
        category = "Neutral"
        emoji = "😐"
        color = "gray"
    else:
        category = "Positivo"
        emoji = "😊"
        color = "green"
    
    return {
        'stars': stars,
        'score': result['score'],
        'category': category,
        'emoji': emoji,
        'color': color
    }


def analyze_emotions(text: str, pipeline) -> List[Dict]:
    """
    Detecta emociones en un texto.
    
    Returns:
        Lista de emociones con scores
    """
    results = pipeline(text)[0]
    
    # Mapeo de emociones a emojis
    emotion_emojis = {
        'anger': '😠',
        'disgust': '🤢',
        'fear': '😨',
        'joy': '😄',
        'neutral': '😐',
        'sadness': '😢',
        'surprise': '😲'
    }
    
    emotions = []
    for r in results:
        emotions.append({
            'emotion': r['label'],
            'emoji': emotion_emojis.get(r['label'], '❓'),
            'score': r['score']
        })
    
    return sorted(emotions, key=lambda x: x['score'], reverse=True)


def batch_analyze(texts: List[str], sentiment_pipe, emotion_pipe=None) -> pd.DataFrame:
    """
    Analiza múltiples textos y devuelve un DataFrame.
    """
    results = []
    
    for i, text in enumerate(texts):
        if not text.strip():
            continue
            
        sentiment = analyze_sentiment(text, sentiment_pipe)
        
        row = {
            'Texto': text[:100] + '...' if len(text) > 100 else text,
            'Estrellas': '⭐' * sentiment['stars'],
            'Categoría': f"{sentiment['emoji']} {sentiment['category']}",
            'Confianza': f"{sentiment['score']:.1%}"
        }
        
        # Agregar emoción principal si está disponible
        if emotion_pipe:
            try:
                emotions = analyze_emotions(text, emotion_pipe)
                top_emotion = emotions[0]
                row['Emoción'] = f"{top_emotion['emoji']} {top_emotion['emotion']}"
            except:
                row['Emoción'] = "N/A"
        
        results.append(row)
    
    return pd.DataFrame(results)


# ============================================
# Ejemplos de reviews
# ============================================
EXAMPLE_REVIEWS = {
    "Reviews de productos (EN)": [
        "This product is amazing! Best purchase I've ever made.",
        "Terrible quality, broke after one day. Complete waste of money.",
        "It's okay, nothing special but does the job.",
        "Exceeded my expectations! Will definitely buy again.",
        "Not worth the price. Very disappointed."
    ],
    "Reseñas de restaurantes (ES)": [
        "¡Excelente comida y servicio! Volveré sin duda.",
        "Pésimo servicio, tardaron una hora en traer la comida fría.",
        "Comida normal, nada del otro mundo.",
        "El mejor restaurante de la ciudad, muy recomendado.",
        "No volvería nunca. Horrible experiencia."
    ],
    "Comentarios de apps (Mixto)": [
        "Love this app! Works perfectly.",
        "Crashes every time I open it. Useless.",
        "Decent app, could use more features.",
        "La mejor aplicación que he usado, funciona perfectamente.",
        "Muy mala, llena de bugs y publicidad."
    ]
}


# ============================================
# Interfaz de usuario
# ============================================
def main():
    st.title("📝 Análisis de Sentimientos")
    st.markdown("""
    **Analiza el sentimiento de comentarios y reviews**
    
    - 🔍 Detecta si un texto es positivo, negativo o neutral
    - ⭐ Clasifica en escala de 1-5 estrellas
    - 😊 Identifica emociones específicas
    - 📊 Analiza múltiples textos a la vez
    """)
    
    st.divider()
    
    # ----------------------------------------
    # Cargar modelos
    # ----------------------------------------
    with st.status("Cargando modelos...", expanded=False) as status:
        st.write("Cargando modelo de sentimientos...")
        sentiment_pipe, sentiment_error = load_sentiment_model()
        
        if sentiment_error:
            st.error(f"❌ Error: {sentiment_error}")
            status.update(label="Error en modelos", state="error")
            return
        else:
            st.write("✅ Modelo de sentimientos cargado")
        
        st.write("Cargando modelo de emociones...")
        emotion_pipe, emotion_error = load_emotion_model()
        
        if emotion_error:
            st.warning(f"⚠️ Emociones no disponibles: {emotion_error}")
        else:
            st.write("✅ Modelo de emociones cargado")
        
        status.update(label="Modelos listos ✅", state="complete")
    
    # ----------------------------------------
    # Tabs para diferentes modos
    # ----------------------------------------
    tab1, tab2, tab3 = st.tabs(["📝 Texto único", "📋 Múltiples textos", "📚 Ejemplos"])
    
    # ========================================
    # TAB 1: Análisis de texto único
    # ========================================
    with tab1:
        st.subheader("Analizar un comentario")
        
        text_input = st.text_area(
            "Escribe o pega un comentario/review:",
            placeholder="Ej: This product exceeded all my expectations! Amazing quality.",
            height=100,
            key="single_text"
        )
        
        if st.button("🔍 Analizar", type="primary", key="analyze_single"):
            if not text_input.strip():
                st.warning("⚠️ Por favor, escribe un texto para analizar.")
            else:
                with st.spinner("Analizando..."):
                    # Análisis de sentimiento
                    sentiment = analyze_sentiment(text_input, sentiment_pipe)
                    
                    # Mostrar resultado principal
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Sentimiento",
                            value=f"{sentiment['emoji']} {sentiment['category']}"
                        )
                    
                    with col2:
                        st.metric(
                            label="Puntuación",
                            value='⭐' * sentiment['stars']
                        )
                    
                    with col3:
                        st.metric(
                            label="Confianza",
                            value=f"{sentiment['score']:.1%}"
                        )
                    
                    # Análisis de emociones si está disponible
                    if emotion_pipe:
                        st.divider()
                        st.subheader("🎭 Emociones detectadas")
                        
                        try:
                            emotions = analyze_emotions(text_input, emotion_pipe)
                            
                            # Crear barras de progreso para cada emoción
                            for emotion in emotions[:5]:  # Top 5
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    st.write(f"{emotion['emoji']} {emotion['emotion'].capitalize()}")
                                with col2:
                                    st.progress(emotion['score'], text=f"{emotion['score']:.1%}")
                        except Exception as e:
                            st.warning(f"No se pudieron analizar emociones: {e}")
    
    # ========================================
    # TAB 2: Análisis batch
    # ========================================
    with tab2:
        st.subheader("Analizar múltiples comentarios")
        
        st.markdown("Escribe un comentario por línea:")
        
        batch_input = st.text_area(
            "Comentarios (uno por línea):",
            placeholder="Comentario 1\nComentario 2\nComentario 3",
            height=200,
            key="batch_text"
        )
        
        include_emotions = st.checkbox(
            "Incluir análisis de emociones",
            value=True,
            disabled=(emotion_pipe is None)
        )
        
        if st.button("📊 Analizar todos", type="primary", key="analyze_batch"):
            texts = [t.strip() for t in batch_input.split('\n') if t.strip()]
            
            if not texts:
                st.warning("⚠️ Por favor, escribe al menos un comentario.")
            else:
                with st.spinner(f"Analizando {len(texts)} comentarios..."):
                    df = batch_analyze(
                        texts, 
                        sentiment_pipe, 
                        emotion_pipe if include_emotions else None
                    )
                    
                    st.success(f"✅ {len(df)} comentarios analizados")
                    
                    # Mostrar tabla
                    st.dataframe(df, use_container_width=True)
                    
                    # Estadísticas
                    st.divider()
                    st.subheader("📈 Resumen")
                    
                    # Contar categorías
                    categories = df['Categoría'].value_counts()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    positivos = sum(1 for c in df['Categoría'] if 'Positivo' in c)
                    neutrales = sum(1 for c in df['Categoría'] if 'Neutral' in c)
                    negativos = sum(1 for c in df['Categoría'] if 'Negativo' in c)
                    
                    with col1:
                        st.metric("😊 Positivos", positivos)
                    with col2:
                        st.metric("😐 Neutrales", neutrales)
                    with col3:
                        st.metric("😞 Negativos", negativos)
                    
                    # Descargar CSV
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Descargar CSV",
                        data=csv,
                        file_name="analisis_sentimientos.csv",
                        mime="text/csv"
                    )
    
    # ========================================
    # TAB 3: Ejemplos
    # ========================================
    with tab3:
        st.subheader("Probar con ejemplos")
        
        example_set = st.selectbox(
            "Selecciona un conjunto de ejemplos:",
            options=list(EXAMPLE_REVIEWS.keys())
        )
        
        st.markdown("**Comentarios de ejemplo:**")
        for i, review in enumerate(EXAMPLE_REVIEWS[example_set], 1):
            st.text(f"{i}. {review}")
        
        if st.button("🔍 Analizar ejemplos", type="primary", key="analyze_examples"):
            with st.spinner("Analizando ejemplos..."):
                df = batch_analyze(
                    EXAMPLE_REVIEWS[example_set],
                    sentiment_pipe,
                    emotion_pipe
                )
                
                st.dataframe(df, use_container_width=True)
    
    # ----------------------------------------
    # Footer
    # ----------------------------------------
    st.divider()
    
    with st.expander("ℹ️ Información técnica"):
        st.markdown("""
        **Modelos utilizados:**
        - **Sentimiento**: `nlptown/bert-base-multilingual-uncased-sentiment`
          - Clasificación en escala 1-5 estrellas
          - Soporta múltiples idiomas (EN, ES, DE, FR, IT, NL)
        - **Emociones**: `j-hartmann/emotion-english-distilroberta-base`
          - Detecta: anger, disgust, fear, joy, neutral, sadness, surprise
          - Optimizado para inglés
        
        **Limitaciones:**
        - Máximo 512 tokens por texto
        - Emociones funcionan mejor en inglés
        - Primera carga puede tardar ~30 segundos
        """)


# ============================================
# Entry point
# ============================================
if __name__ == "__main__":
    main()
