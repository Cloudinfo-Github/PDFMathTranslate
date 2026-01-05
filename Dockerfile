FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HTTPX_TIMEOUT=120

# System deps (font/render) + Python
RUN apt-get update && \
     apt-get install --no-install-recommends -y python3 python3-pip python3-venv libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 && \
     rm -rf /var/lib/apt/lists/*

COPY . .

# Base Python deps and GPU ONNX runtime (installed last to override CPU build)
# Also install rapidocr for table text translation support
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
     python3 -m pip install --no-cache-dir . && \
     python3 -m pip install --no-cache-dir "onnxruntime-gpu>=1.18.0" && \
     python3 -m pip install --no-cache-dir -U "babeldoc>=0.1.22,<0.3.0" "pymupdf<1.25.3" "pdfminer-six==20250416" && \
     python3 -m pip install --no-cache-dir rapidocr_onnxruntime || true

# Apply BabelDOC patch to fix missing text issue
# Problem: DocLayout-YOLO may not detect all text regions, causing text like "NO.1 What practices..."
# to be skipped during translation. This patch treats undetected text as plain text.
RUN python3 patches/fix_missing_text.py

# Pre-download all fonts and models during build to avoid runtime network issues
# This bakes all assets into the image so translation works offline
RUN python3 patches/warmup_assets.py

EXPOSE 7860

CMD ["pdf2zh", "-i"]
