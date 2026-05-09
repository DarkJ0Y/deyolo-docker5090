# ============================================================
# DEYOLO — Dual-modal YOLOv8 for VIP Cup 2026
# Target GPU : NVIDIA RTX 5090 (Blackwell, sm_120, CUDA 12.8)
# Base image : PyTorch 2.7 + CUDA 12.8 (official nightly)
# ============================================================

FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# ── System packages ──────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        unzip \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /workspace

# ── Python dependencies (RTX 5090 / Blackwell-compatible) ────
# PyTorch is already in the base image (cu128 build).
# torchvision must match the torch version.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Torchvision 0.22 ships with PyTorch 2.7 wheels for cu128
RUN pip install --no-cache-dir \
        torchvision==0.22.0 \
        torchaudio==2.7.0

# Core ML / CV stack
RUN pip install --no-cache-dir \
        einops \
        opencv-python-headless \
        matplotlib \
        numpy \
        scipy \
        pillow \
        tqdm \
        pyyaml \
        pandas \
        seaborn

# Experiment tracking
RUN pip install --no-cache-dir wandb

# ── Clone & install DEYOLO (custom ultralytics fork) ─────────
RUN git clone https://github.com/chips96/DEYOLO.git /workspace/DEYOLO
WORKDIR /workspace/DEYOLO
RUN pip install --no-cache-dir -e .

# ── Kaggle CLI (for dataset download) ────────────────────────
RUN pip install --no-cache-dir kaggle

# ── Copy project files ────────────────────────────────────────
#WORKDIR /workspace
#COPY download_dataset.sh /workspace/download_dataset.sh
#COPY run_notebook.py     /workspace/run_notebook.py
#COPY notebook_converted.py /workspace/notebook_converted.py

RUN chmod +x /workspace/download_dataset.sh

# ── Entrypoint ────────────────────────────────────────────────
# Expects: KAGGLE_USERNAME and KAGGLE_KEY env vars for dataset download
CMD ["/bin/bash", "sleep", "infinity"]
