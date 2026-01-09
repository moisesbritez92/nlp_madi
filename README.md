# 🧠 NLP MADI - Streamlit Application

> **Aplicación multipage de NLP con Streamlit, incluyendo RAG Search, Image-to-Speech y Análisis de Sentimientos**

---

## 📋 Tabla de Contenidos

- [Requisitos](#-requisitos)
- [Instalación Rápida](#-instalación-rápida)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Páginas Disponibles](#-páginas-disponibles)
- [Configuración del LLM](#-configuración-del-llm)
- [Docker](#-docker)
- [Validación / Testing](#-validación--testing)

---

## 📦 Requisitos

| Requisito | Descripción |
|-----------|-------------|
| **Python** | 3.10+ |
| **Docker** | Docker Desktop (opcional, para contenedores) |
| **Puerto Streamlit** | `8501` (acceso via `http://localhost:8501`) |
| **LLM Backend** | Ollama local o OpenAI API |

---

## 🚀 Instalación Rápida

### Opción 1: Virtual Environment (Local)

```bash
# Crear y activar entorno virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app.py
```

### Opción 2: Docker

```bash
# Construir y ejecutar
docker-compose up -d --build

# Acceder a:
# - Streamlit: http://localhost:8501
# - JupyterLab: http://localhost:18888
```

---

## 📁 Estructura del Proyecto

```
nlp_madi/
├── app.py                      # 🏠 Home page
├── pages/
│   ├── 1_🖼️_Image_to_Speech.py  # Imagen → Audio
│   ├── 2_📝_Analisis_Sentimientos.py  # Sentiment Analysis
│   └── 3_🔍_RAG_Search.py      # ⭐ RAG Search Tool
├── rag/                        # Módulos RAG
│   ├── __init__.py
│   ├── splade_retriever.py     # SPLADE semantic retrieval
│   ├── bm25_retriever.py       # BM25 lexical retrieval
│   ├── llm_client.py           # LLM API client
│   ├── data_loader.py          # Corpus loading utilities
│   └── prompts.py              # System prompts for RAG
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 📄 Páginas Disponibles

### 🖼️ Image to Speech
Convierte imágenes en audio hablado usando:
- **BLIP** para generar descripciones
- **Kokoro TTS** para síntesis de voz

### 📝 Análisis de Sentimientos
Analiza comentarios y reviews:
- Clasificación 1-5 estrellas (multilingüe)
- Detección de emociones (inglés)
- Análisis batch con exportación CSV

### 🔍 RAG Search (NEW!)
Búsqueda semántica con Retrieval-Augmented Generation:
- **SPLADE**: Sparse semantic retrieval con expansión de términos
- **BM25**: Classic lexical retrieval
- **LLM Integration**: Respuestas generadas con citación de fuentes

---

## 🤖 Configuración del LLM

### Ollama (Local - Recomendado)

1. Instalar [Ollama](https://ollama.ai/)
2. Descargar un modelo:
   ```bash
   ollama pull gemma3
   # o
   ollama pull llama2
   ```
3. En RAG Search, configurar:
   - URL: `http://localhost:11434/api/chat`
   - Model: `gemma3`

### Ollama Remoto

Si tienes Ollama en otro servidor:
- URL: `http://<server-ip>:11434/api/chat`

### OpenAI API

1. Obtener API key en [OpenAI Platform](https://platform.openai.com/)
2. En RAG Search sidebar:
   - Backend: "OpenAI API"
   - API Key: tu clave
   - Model: `gpt-3.5-turbo` o `gpt-4`

---

## 🐳 Docker

### Archivos incluidos

```
├── Dockerfile              # Imagen base con dependencias
├── docker-compose.yml      # Orquestación del contenedor
```

### Construir e iniciar el contenedor

```bash
# Construir la imagen y levantar el contenedor
docker-compose up -d --build
```

### Verificar que está corriendo

```bash
docker-compose ps
```

---

## ✅ Validación / Testing

### Checklist de RAG Search

| Test | Query | Resultado Esperado |
|------|-------|-------------------|
| **BM25 vs SPLADE** | "What is the difference between TF IDF and BM25 scoring?" | Debe citar [doc5], [doc6], [doc43] |
| **Semantic Expansion** | "Any document talking about Spanish products?" | SPLADE debe encontrar doc46 (paella/tortilla) aunque no diga "Spanish" |
| **No Evidence** | "What is quantum computing?" | Debe indicar que no hay información suficiente |
| **Citation** | Cualquier query | La respuesta debe incluir [docX] inline |

### Queries de Prueba Recomendadas

```
1. What is the difference between TF IDF and BM25 scoring?
2. How does SPLADE create sparse query expansion terms?
3. Why do we use cosine similarity when ranking documents?
4. What is retrieval augmented generation and why is it useful?
5. How do chunking and overlap stride help long document retrieval?
6. Any document talking about Spanish products?
7. What are stopwords stemming and lemmatization used for?
```

### Verificar Conexión LLM

1. Ir a RAG Search
2. En sidebar, click "🔗 Test LLM Connection"
3. Debe mostrar "Connected successfully"

---

**Salida esperada:**
```
NAME                    STATUS    PORTS
jupyter-streamlit-app   Up        0.0.0.0:18888->8888/tcp, 0.0.0.0:8501->8501/tcp
```

### Comandos útiles de Docker

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Detener el contenedor
docker-compose down

# Reiniciar el contenedor
docker-compose restart

# Reconstruir (después de cambios en Dockerfile)
docker-compose up -d --build

# Entrar al contenedor con bash
docker exec -it jupyter-streamlit-app bash
```

---

## 🔄 Flujo Completo Paso a Paso

### Paso 1: Construir y levantar el contenedor

```bash
cd /ruta/a/streamlit_app
docker-compose up -d --build
```

### Paso 2: Verificar que el contenedor está activo

Asegúrate de que el contenedor Docker está corriendo:

```bash
docker-compose ps
# o
docker ps
```

### Paso 3: Acceder a JupyterLab

Abre tu navegador y navega a:

```
http://localhost:18888
```

### Paso 4: Abrir una Terminal en JupyterLab

1. En JupyterLab, ve al menú **File** → **New** → **Terminal**
2. O usa el **Launcher** y haz clic en **Terminal**

> ⚠️ **IMPORTANTE**: Todos los comandos se ejecutan desde la **Terminal de JupyterLab**, NO desde celdas de notebook.

### Paso 5: Navegar al directorio de los scripts

```bash
cd "/app/Materiales docentes/2526"
```

### Paso 6: Ejecutar el script de preinstalación

```bash
python streamlit_preinstalation.py
```

### Paso 7: Manejar el error de Kokoro (si aparece)

Si ves el siguiente error:

```
ModuleNotFoundError: No module named 'kokoro'
```

**Solución**: Ejecuta el script **UNA SEGUNDA VEZ**:

```bash
python streamlit_preinstalation.py
```

### Paso 8: Ejecutar Streamlit

```bash
streamlit run test_app.py
```

### Paso 9: Abrir la aplicación

El comando mostrará varias URLs. **Debes abrir la SEGUNDA URL** o directamente:

```
http://localhost:8501
```

También puedes ver la salida del comando:

```
You can now view your Streamlit app in your browser.

  Network URL: http://172.17.0.2:8501      ← ❌ NO esta
  External URL: http://192.168.1.100:8501  ← ✅ ABRIR ESTA (segunda URL)
```

> 💡 **TIP**: Como el puerto 8501 está mapeado en Docker, puedes acceder directamente a `http://localhost:8501`

---

## ✅ Checklist de Verificación

Usa esta checklist para asegurarte de que todo está configurado correctamente:

| # | Verificación | Estado |
|---|--------------|--------|
| 1 | ☐ `docker-compose up -d --build` ejecutado | |
| 2 | ☐ Contenedor Docker activo (`docker-compose ps`) | |
| 3 | ☐ JupyterLab accesible en `http://localhost:18888` | |
| 4 | ☐ Terminal abierta dentro de JupyterLab | |
| 5 | ☐ Navegado a `/app/Materiales docentes/2526` | |
| 6 | ☐ Script `streamlit_preinstalation.py` ejecutado | |
| 7 | ☐ Si hubo error de `kokoro`, script ejecutado por segunda vez | |
| 8 | ☐ Mensaje `[OK] All packages installed and models loaded successfully.` visible | |
| 9 | ☐ Streamlit corriendo con `streamlit run test_app.py` | |
| 10 | ☐ App accesible en `http://localhost:8501` | |
| 11 | ☐ Aplicación "Word Counter" visible y funcional | |

---

## 💻 Comandos Exactos

### Instalación de dependencias

```bash
python streamlit_preinstalation.py
```

**Salida esperada (exitosa):**

```
[INFO] Installing transformers...
[INFO] Installing torch...
[INFO] Installing kokoro>=0.9.4...
[INFO] Installing soundfile...
[INFO] Installing torchcodec...
[INFO] Installing datasets...
[INFO] Installing librosa...
[INFO] Installing Pillow...
[INFO] Installing streamlit...
[INFO] Installing system package espeak-ng with apt-get...
[INFO] Importing libraries and loading models...

[OK] All packages installed and models loaded successfully.
```

### Ejecución de Streamlit

```bash
streamlit run test_app.py
```

**Salida esperada:**

```
You can now view your Streamlit app in your browser.

  Network URL: http://172.17.0.2:8501
  External URL: http://192.168.1.100:8501    ← ✅ ABRIR ESTA URL

```

> 📌 **RECUERDA**: Siempre abre la **SEGUNDA URL** (External URL)

---

### 📍 Dónde ejecutar comandos

| ✅ Correcto | ❌ Incorrecto |
|-------------|---------------|
| Terminal de JupyterLab | Celda de notebook |
| Terminal dentro del contenedor | Terminal de tu máquina host |

### 🔗 Sobre las URLs de Streamlit

Streamlit muestra **múltiples URLs** al ejecutarse:

- **Local URL**: Solo accesible desde dentro del contenedor
- **Network URL**: IP interna del contenedor Docker
- **External URL**: La que debes usar para acceder desde tu navegador

> 🎯 **Regla de oro**: Siempre usa la **segunda URL mostrada**

---

## 🔧 Common Issues (Troubleshooting)

### Error: `ModuleNotFoundError: No module named 'kokoro'`

**Causa**: Algunas dependencias no se instalaron correctamente en la primera ejecución.

**Solución**: Ejecutar el script de preinstalación una segunda vez:

```bash
python streamlit_preinstalation.py
```

---

### Error: Modelos tardan mucho en cargar

**Causa**: Primera ejecución descarga modelos (~1-2 GB).

**Solución**: 
- Espera la primera vez (~1-5 minutos según conexión)
- Los modelos se cachean en `/root/.cache/huggingface`
- Siguientes ejecuciones serán rápidas (~10-30 seg)

---

### Error: "No se generó audio" en Image to Speech

**Causa**: Kokoro TTS no se instaló correctamente o falta `espeak-ng`.

**Solución**:
```bash
# Reinstalar dependencias
python streamlit_preinstalation.py

# Si persiste, verificar espeak-ng
apt-get update && apt-get install -y espeak-ng
```

---

### Error: Memoria insuficiente / App se congela

**Causa**: BLIP + Kokoro requieren ~2-4 GB RAM.

**Solución**:
- Aumentar memoria de Docker Desktop (Settings → Resources → Memory)
- Reiniciar el contenedor: `docker-compose restart`

---

### Error: Streamlit no abre / no muestra nada

**Posibles causas y soluciones**:

1. **El puerto no está expuesto**
   ```bash
   # Verificar que el contenedor expone el puerto 8501
   docker ps
   ```
   Busca `8501` en la columna PORTS.

2. **Firewall bloqueando el puerto**
   - Verificar configuración de firewall local

3. **Streamlit corriendo pero URL incorrecta**
   - Asegúrate de abrir la **segunda URL**, no la primera

---

### Error: No aparecen URLs al ejecutar Streamlit

**Solución**: Espera unos segundos. Si sigue sin aparecer:

```bash
# Detener con Ctrl+C y reintentar con:
streamlit run test_app.py --server.headless true
```

---

### Error: Puerto incorrecto / No puedo acceder a la app

**Verificar el mapeo de puertos del contenedor**:

```bash
docker ps --format "table {{.Names}}\t{{.Ports}}"
```

Si el puerto 8501 no está mapeado, puede que necesites recrear el contenedor con el puerto expuesto:

```bash
# Ejemplo de cómo debería verse el mapeo
# 0.0.0.0:8501->8501/tcp
```

**Alternativa - especificar puerto manualmente**:

```bash
streamlit run test_app.py --server.port 8501
```

---

### Error: Streamlit queda bloqueado en terminal

**Esto es comportamiento normal**. Streamlit es un servidor web que debe seguir corriendo.

**Para detenerlo**: Presiona `Ctrl + C` en la terminal.

**Para ejecutar en segundo plano** (no recomendado para desarrollo):

```bash
nohup streamlit run test_app.py &
```

**Para ver los logs si está en segundo plano**:

```bash
tail -f nohup.out
```

---

### Error: La app carga pero se ve en blanco

**Solución**: Refrescar la página con `Ctrl + Shift + R` (hard refresh)

Si persiste:
1. Verificar la consola del navegador (F12) para errores
2. Verificar que no hay errores en la terminal de Streamlit

---

## 🎮 Run Demo App

La demo incluida es una aplicación simple de **contador de palabras**.

### Ejecutar la demo

```bash
# Ejecutar la app
streamlit run test_app.py
```

### Qué hace la demo

- **Nombre**: Simple Word Counter
- **Funcionalidad**: Cuenta las palabras de un texto ingresado
- **Interfaz**: 
  - Área de texto para ingresar contenido
  - Botón "Count words"
  - Muestra el resultado del conteo

### Probar la demo

1. Abre http://localhost:8501 en tu navegador
2. Escribe o pega texto en el área de texto
3. Haz clic en "Count words"
4. Verás el conteo de palabras

---

## 🖼️ Run Image to Speech App

Aplicación completa que convierte imágenes en audio hablado usando IA.

### Características

| Funcionalidad | Descripción |
|---------------|-------------|
| **Image Upload** | Sube imágenes JPG, PNG o WEBP |
| **Image Captioning** | Genera descripción usando BLIP (Hugging Face) |
| **Text-to-Speech** | Convierte texto a audio con Kokoro TTS |
| **Audio Player** | Reproduce y descarga el audio generado |

### Arquitectura

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Imagen    │ ──▶ │  BLIP Model  │ ──▶ │   Caption   │
│   (PIL)     │     │  (Caption)   │     │   (Text)    │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Audio     │ ◀── │ Kokoro TTS   │ ◀── │   Caption   │
│   (WAV)     │     │  (Speech)    │     │   (Text)    │
└─────────────┘     └──────────────┘     └─────────────┘
```

### Ejecutar la app

```bash
# 1. (Primera vez) Instalar dependencias y descargar modelos
python streamlit_preinstalation.py

# 2. Si aparece error de kokoro, ejecutar de nuevo
python streamlit_preinstalation.py

# 3. Ejecutar la app
streamlit run app.py
```

### Abrir en el navegador

```
http://localhost:8501
```

### Quick Test

1. **Sube una imagen**: Haz clic en "Browse files" y selecciona una imagen JPG/PNG
2. **Ver preview**: La imagen aparece en la columna izquierda
3. **Generar descripción**: Haz clic en "🔍 Generar descripción"
4. **Ver caption**: Aparece el texto descriptivo en verde
5. **Generar audio**: Haz clic en "🎙️ Generar audio"
6. **Reproducir**: Usa el reproductor de audio integrado
7. **Descargar** (opcional): Haz clic en "💾 Descargar audio"

### Modelos utilizados

| Modelo | Propósito | Tamaño |
|--------|-----------|--------|
| `Salesforce/blip-image-captioning-base` | Image Captioning | ~1GB |
| Kokoro TTS | Text-to-Speech | ~100MB |

### Notas de rendimiento

- **Primera carga**: ~30-60 segundos (descarga/carga de modelos)
- **Siguientes usos**: ~2-5 segundos (modelos cacheados)
- **Memoria RAM**: ~2-4 GB recomendados
- **CPU**: Funciona en CPU, no requiere GPU

### Estructura de archivos

```
streamlit_app/
├── app.py                      # 🖼️ Image to Speech App
├── test_app.py                 # 📝 Word Counter Demo
├── streamlit_preinstalation.py # 📦 Script de instalación
├── Dockerfile                  # 🐳 Imagen Docker
├── docker-compose.yml          # 🐳 Orquestación
├── README.md                   # 📖 Esta guía
└── tmp/                        # 🗂️ Archivos temporales (auto-creado)
```

---

## 🛠️ Run Your Own Streamlit App

### Estructura básica de una app Streamlit

Crea un archivo Python (ej: `mi_app.py`):

```python
import streamlit as st

st.set_page_config(page_title="Mi App", layout="centered")

st.title("🎯 Mi Aplicación Streamlit")

st.write("¡Hola desde Docker!")

# Tu código aquí...
```

### Pasos para ejecutar tu propia app

1. **Crea tu archivo** en el directorio de trabajo de JupyterLab

2. **Abre una Terminal** en JupyterLab

3. **Navega al directorio** donde está tu archivo:
   ```bash
   cd /ruta/a/tu/directorio
   ```

4. **Ejecuta tu app**:
   ```bash
   streamlit run mi_app.py
   ```

5. **Abre la SEGUNDA URL** en tu navegador

### Tips para desarrollo

```bash
# Ejecutar con auto-reload desactivado (útil para debugging)
streamlit run mi_app.py --server.runOnSave false

# Especificar un puerto diferente
streamlit run mi_app.py --server.port 8502

# Modo headless (para entornos sin GUI)
streamlit run mi_app.py --server.headless true
```

### Ejemplo: App con las dependencias preinstaladas

Las dependencias instaladas por `streamlit_preinstalation.py` incluyen:

| Librería | Uso |
|----------|-----|
| `transformers` | Modelos de NLP de Hugging Face |
| `torch` | Framework de deep learning |
| `kokoro` | Text-to-Speech |
| `soundfile` | Manipulación de archivos de audio |
| `datasets` | Datasets de Hugging Face |
| `librosa` | Análisis de audio |
| `Pillow` | Procesamiento de imágenes |

**Ejemplo usando sentiment analysis**:

```python
import streamlit as st
from transformers import pipeline

st.title("😊 Sentiment Analyzer")

classifier = pipeline("sentiment-analysis")

text = st.text_input("Enter text to analyze:")

if st.button("Analyze"):
    if text:
        result = classifier(text)[0]
        st.write(f"**Label**: {result['label']}")
        st.write(f"**Score**: {result['score']:.2%}")
```

---

## 📝 Resumen Rápido

```
┌─────────────────────────────────────────────────────────┐
│  1. docker-compose up -d --build                        │
│  2. Abrir http://localhost:18888                        │
│  3. Abrir Terminal en JupyterLab                        │
│  4. cd "/app/Materiales docentes/2526"                  │
│  5. python streamlit_preinstalation.py                  │
│  6. (Si error kokoro) → ejecutar de nuevo el script     │
│  7. streamlit run test_app.py                           │
│  8. Abrir http://localhost:8501 en el navegador         │
└─────────────────────────────────────────────────────────┘
```

---

## 🆘 Soporte

Si encuentras problemas no cubiertos en esta guía:

1. Verifica que el contenedor Docker está corriendo
2. Revisa los logs del contenedor: `docker logs <container_id>`
3. Asegúrate de estar usando la Terminal de JupyterLab, no una externa
4. Confirma que no has creado ningún entorno virtual

---

> **Última actualización**: Diciembre 2025  
> **Versión**: 1.0
# nlp_madi
