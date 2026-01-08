import subprocess
import sys


def pip_install(package: str):
    """Install a package using pip."""
    print(f"\n[INFO] Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def apt_install(package: str):
    """Install a system package using apt-get (no sudo, for Docker)."""
    print(f"\n[INFO] Installing system package {package} with apt-get...")
    cmd = ["apt-get", "-qq", "-y", "install", package]
    subprocess.check_call(cmd)


def main():
    # ---- 1. pip installs ----
    pip_packages = [
        "transformers",
        "torch",
        "kokoro>=0.9.4",
        "soundfile",
        "torchcodec",
        "datasets",
        "librosa",
        "Pillow",
        "streamlit",
    ]

    for pkg in pip_packages:
        pip_install(pkg)

    # ---- 2. apt-get install ----
    try:
        apt_install("espeak-ng")
    except Exception as e:
        print("\n[WARNING] Could not install 'espeak-ng' via apt-get.")
        print("          Make sure your Docker base image supports apt-get.")
        print(f"          Error details: {e}")

    # ---- 3. Run your Python code ----
    print("\n[INFO] Importing libraries and loading models...")

    from transformers import pipeline as hf_pipeline
    from kokoro import KPipeline
    import soundfile as sf  # noqa: F401
    import torch  # noqa: F401

    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image  # noqa: F401

    # Sentiment analysis
    sentiment_classifier = hf_pipeline("sentiment-analysis")

    # Kokoro TTS
    tts_pipeline = KPipeline(lang_code="a")

    # BLIP captioning
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    print("\n[OK] All packages installed and models loaded successfully.")

    # Optional example:
    # print(sentiment_classifier("This is running inside a Docker container!"))


if __name__ == "__main__":
    main()