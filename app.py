"""
🖼️ Image to Speech App
======================
Aplicación Streamlit que:
1. Permite subir una imagen
2. Genera una descripción (caption) usando BLIP
3. Convierte la descripción a audio usando Kokoro TTS
4. Reproduce el audio en la interfaz

Ejecutar desde Terminal de JupyterLab:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image
import io
import os
import tempfile
from pathlib import Path

# ============================================
# Configuración de página
# ============================================
st.set_page_config(
    page_title="🖼️ Image to Speech",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================
# Directorio temporal para audio
# ============================================
TMP_DIR = Path("./tmp")
TMP_DIR.mkdir(exist_ok=True)


# ============================================
# Carga de modelos con caché
# ============================================
@st.cache_resource(show_spinner=False)
def load_caption_model():
    """
    Carga el modelo BLIP para image captioning.
    Usa cache para evitar recargar en cada interacción.
    """
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        return processor, model, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource(show_spinner=False)
def load_tts_model():
    """
    Carga el pipeline Kokoro TTS.
    Usa cache para evitar recargar en cada interacción.
    """
    try:
        from kokoro import KPipeline
        # lang_code="a" para inglés americano
        tts = KPipeline(lang_code="a")
        return tts, None
    except ImportError as e:
        return None, f"Kokoro no instalado: {e}. Ejecuta: python streamlit_preinstalation.py"
    except Exception as e:
        return None, str(e)


# ============================================
# Funciones de procesamiento
# ============================================
def generate_caption(image: Image.Image, processor, model) -> str:
    """
    Genera una descripción textual de la imagen usando BLIP.
    
    Args:
        image: Imagen PIL
        processor: BLIP processor
        model: BLIP model
    
    Returns:
        String con la descripción generada
    """
    import torch
    
    # Preprocesar imagen
    inputs = processor(image, return_tensors="pt")
    
    # Generar caption (sin gradientes para ahorrar memoria)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=3,
            early_stopping=True
        )
    
    # Decodificar
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def generate_audio(text: str, tts_pipeline) -> bytes:
    """
    Genera audio a partir de texto usando Kokoro TTS.
    
    Args:
        text: Texto a convertir
        tts_pipeline: Pipeline de Kokoro
    
    Returns:
        Bytes del archivo de audio WAV
    """
    import soundfile as sf
    import numpy as np
    
    # Generar audio con Kokoro
    # Especificar voz (af_heart es una voz femenina americana)
    # Voces disponibles: af_heart, af_bella, af_sarah, am_adam, am_michael, etc.
    audio_chunks = []
    
    for _, _, audio_chunk in tts_pipeline(text, voice="af_heart"):
        if audio_chunk is not None:
            # Convertir a numpy si es tensor
            if hasattr(audio_chunk, 'numpy'):
                audio_chunk = audio_chunk.numpy()
            audio_chunks.append(audio_chunk)
    
    if not audio_chunks:
        raise ValueError("No se generó audio")
    
    # Concatenar chunks
    full_audio = np.concatenate(audio_chunks)
    
    # Guardar a bytes usando soundfile
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, full_audio, samplerate=24000, format='WAV')
    audio_buffer.seek(0)
    
    return audio_buffer.read()


def generate_audio_fallback(text: str) -> tuple:
    """
    Fallback usando gTTS si Kokoro no está disponible.
    Requiere conexión a Internet.
    
    Returns:
        Tuple (bytes_audio, error_message)
    """
    try:
        from gtts import gTTS
        
        audio_buffer = io.BytesIO()
        tts = gTTS(text=text, lang='en', slow=False)
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.read(), None
    except ImportError:
        return None, "gTTS no instalado (fallback no disponible)"
    except Exception as e:
        return None, f"Error en gTTS: {e}"


# ============================================
# Interfaz de usuario
# ============================================
def main():
    # Título principal
    st.title("🖼️ Image to Speech")
    st.markdown("""
    **Convierte imágenes en audio hablado**
    
    1. 📤 Sube una imagen (JPG, PNG)
    2. 🔍 La IA generará una descripción
    3. 🔊 Escucha la descripción en audio
    """)
    
    st.divider()
    
    # ----------------------------------------
    # Cargar modelos (con indicador de progreso)
    # ----------------------------------------
    with st.status("Cargando modelos...", expanded=False) as status:
        st.write("Cargando modelo de captioning (BLIP)...")
        processor, caption_model, caption_error = load_caption_model()
        
        if caption_error:
            st.error(f"❌ Error cargando BLIP: {caption_error}")
            status.update(label="Error en modelos", state="error")
        else:
            st.write("✅ BLIP cargado")
        
        st.write("Cargando modelo TTS (Kokoro)...")
        tts_pipeline, tts_error = load_tts_model()
        
        if tts_error:
            st.warning(f"⚠️ TTS: {tts_error}")
            st.write("Se usará fallback si es posible")
        else:
            st.write("✅ Kokoro TTS cargado")
        
        if not caption_error:
            status.update(label="Modelos listos ✅", state="complete")
    
    # ----------------------------------------
    # Uploader de imagen
    # ----------------------------------------
    st.subheader("📤 Sube tu imagen")
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png", "webp"],
        help="Formatos soportados: JPG, PNG, WEBP"
    )
    
    # ----------------------------------------
    # Procesar imagen
    # ----------------------------------------
    if uploaded_file is not None:
        # Cargar y mostrar imagen
        image = Image.open(uploaded_file).convert("RGB")
        
        # Mostrar preview
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Imagen subida", use_container_width=True)
        
        with col2:
            st.subheader("📝 Descripción")
            
            # Verificar que el modelo está disponible
            if caption_model is None:
                st.error("❌ Modelo de captioning no disponible")
                st.info("Ejecuta: `python streamlit_preinstalation.py`")
                return
            
            # Botón para generar descripción
            if st.button("🔍 Generar descripción", type="primary", use_container_width=True):
                
                with st.spinner("Analizando imagen..."):
                    try:
                        caption = generate_caption(image, processor, caption_model)
                        st.session_state['caption'] = caption
                        st.session_state['audio_generated'] = False
                    except Exception as e:
                        st.error(f"Error generando caption: {e}")
                        return
            
            # Mostrar caption si existe
            if 'caption' in st.session_state and st.session_state['caption']:
                caption = st.session_state['caption']
                
                st.success(f"**{caption}**")
                
                # Separador visual
                st.divider()
                
                # ----------------------------------------
                # Sección de audio
                # ----------------------------------------
                st.subheader("🔊 Audio")
                
                if st.button("🎙️ Generar audio", use_container_width=True):
                    
                    with st.spinner("Generando audio..."):
                        audio_bytes = None
                        error_msg = None
                        
                        # Intentar con Kokoro primero
                        if tts_pipeline is not None:
                            try:
                                audio_bytes = generate_audio(caption, tts_pipeline)
                            except Exception as e:
                                error_msg = f"Error con Kokoro: {e}"
                        
                        # Fallback a gTTS si Kokoro falló
                        if audio_bytes is None:
                            st.warning(f"⚠️ {error_msg or 'Kokoro no disponible'}. Intentando fallback...")
                            audio_bytes, fallback_error = generate_audio_fallback(caption)
                            
                            if fallback_error:
                                error_msg = fallback_error
                        
                        if audio_bytes:
                            st.session_state['audio_bytes'] = audio_bytes
                            st.session_state['audio_generated'] = True
                        else:
                            st.error(f"❌ No se pudo generar audio: {error_msg}")
                            st.info("💡 **Solución**: Ejecuta `python streamlit_preinstalation.py` dos veces")
                
                # Mostrar reproductor si hay audio
                if st.session_state.get('audio_generated') and st.session_state.get('audio_bytes'):
                    st.audio(st.session_state['audio_bytes'], format='audio/wav')
                    
                    # Botón de descarga
                    st.download_button(
                        label="💾 Descargar audio",
                        data=st.session_state['audio_bytes'],
                        file_name="description_audio.wav",
                        mime="audio/wav"
                    )
    
    # ----------------------------------------
    # Footer informativo
    # ----------------------------------------
    st.divider()
    
    with st.expander("ℹ️ Información técnica"):
        st.markdown("""
        **Modelos utilizados:**
        - **Image Captioning**: BLIP (Salesforce/blip-image-captioning-base)
        - **Text-to-Speech**: Kokoro TTS con espeak-ng
        
        **Ejecución:**
        - Todo corre en CPU dentro del contenedor Docker
        - Los modelos se cachean para mejor rendimiento
        
        **Troubleshooting:**
        - Si el TTS falla, ejecuta: `python streamlit_preinstalation.py` (dos veces si es necesario)
        - Primera carga puede tardar ~30 segundos
        """)


# ============================================
# Entry point
# ============================================
if __name__ == "__main__":
    # Inicializar session state
    if 'caption' not in st.session_state:
        st.session_state['caption'] = None
    if 'audio_bytes' not in st.session_state:
        st.session_state['audio_bytes'] = None
    if 'audio_generated' not in st.session_state:
        st.session_state['audio_generated'] = False
    
    main()
