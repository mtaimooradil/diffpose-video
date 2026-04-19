FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies + Python 3.13 via deadsnakes PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    unzip \
    git \
    libgl1 \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap pip for Python 3.13
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

# Make python3.13 the default python/python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# Install latest PyTorch with CUDA 12.8 support
RUN python -m pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Install all project dependencies
RUN python -m pip install --no-cache-dir \
    numpy \
    scipy \
    einops \
    "timm>=0.9.0" \
    tqdm \
    pyyaml \
    matplotlib \
    tensorboard \
    opencv-python-headless \
    rtmlib \
    "onnxruntime-gpu==1.20.1" \
    dash \
    plotly

WORKDIR /workspace

RUN mkdir -p data checkpoints exp

CMD ["bash"]
