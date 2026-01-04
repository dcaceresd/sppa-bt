FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tmux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sppa-bt

COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel packaging \
    && pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir -r requirements.txt

RUN pip install --upgrade transformers

COPY . .