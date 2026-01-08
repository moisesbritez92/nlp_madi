# ============================================
# Dockerfile para JupyterLab + Streamlit
# ============================================
# Este contenedor proporciona un entorno completo
# para ejecutar aplicaciones Streamlit desde JupyterLab
# SIN necesidad de crear entornos virtuales.
# ============================================

FROM python:3.11-slim

# Evitar prompts interactivos durante instalación
ENV DEBIAN_FRONTEND=noninteractive

# Variables de entorno para Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Directorio de trabajo
WORKDIR /app

# ============================================
# 1. Instalar dependencias del sistema
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Herramientas básicas
    build-essential \
    curl \
    wget \
    git \
    # Para audio (requerido por librosa, soundfile)
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    # Para espeak-ng (TTS)
    espeak-ng \
    # Para Pillow
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    # Limpieza
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# 2. Instalar JupyterLab y dependencias base
# ============================================
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    jupyterlab \
    notebook \
    ipywidgets

# ============================================
# 3. Instalar Streamlit y dependencias ML
# ============================================
# Primero instalar PyTorch CPU desde su índice específico
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Luego instalar el resto de paquetes desde PyPI
RUN pip install --no-cache-dir \
    streamlit \
    transformers \
    soundfile \
    datasets \
    librosa \
    Pillow \
    pandas \
    numpy \
    matplotlib \
    plotly

# ============================================
# 4. Instalar kokoro (TTS) - puede requerir
#    ejecución adicional del script
# ============================================
RUN pip install --no-cache-dir "kokoro>=0.9.4" || true

# ============================================
# 5. Configuración de Streamlit
# ============================================
RUN mkdir -p /root/.streamlit

RUN echo '\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
address = "0.0.0.0"\n\
port = 8501\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /root/.streamlit/config.toml

# ============================================
# 6. Configuración de JupyterLab
# ============================================
RUN mkdir -p /root/.jupyter

RUN echo '\
c.ServerApp.ip = "0.0.0.0"\n\
c.ServerApp.port = 8888\n\
c.ServerApp.open_browser = False\n\
c.ServerApp.allow_root = True\n\
c.ServerApp.token = ""\n\
c.ServerApp.password = ""\n\
c.ServerApp.allow_origin = "*"\n\
c.ServerApp.terminado_settings = {"shell_command": ["/bin/bash"]}\n\
c.ServerApp.root_dir = "/app/Materiales docentes/2526"\n\
' > /root/.jupyter/jupyter_server_config.py

# ============================================
# 7. Crear directorio para materiales
# ============================================
RUN mkdir -p "/app/Materiales docentes/2526"

# Establecer el directorio compartido como directorio de trabajo
WORKDIR "/app/Materiales docentes/2526"

# ============================================
# 8. Exponer puertos
# ============================================
# Puerto 8888: JupyterLab
# Puerto 8501: Streamlit
EXPOSE 8888 8501

# ============================================
# 9. Comando de inicio
# ============================================
CMD ["jupyter", "lab", "--allow-root", "--no-browser"]
