# devel image: nvcc + toolchain required to build cumesh / flex_gemm / o_voxel (TRELLIS native extensions)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda-12.4 \
    PATH=/usr/local/cuda-12.4/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH} \
    MAX_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    git-lfs \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ca-certificates \
    curl \
    sudo \
    ninja-build \
    build-essential \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Normalize CUDA path naming to match the 12.4-only setup steps.
RUN if [ ! -e /usr/local/cuda-12.4 ]; then ln -s /usr/local/cuda /usr/local/cuda-12.4; fi

# Install Miniconda and create the `trellis` env (your “from scratch” flow).
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/bin:/opt/conda/envs/trellis/bin:${PATH}

RUN curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniconda.sh \
    && "${CONDA_DIR}/bin/conda" create -n trellis python=3.10 -y \
    && "${CONDA_DIR}/bin/conda" clean -afy

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch + torchvision (CUDA 12.4 wheels). TRELLIS/timm need torchvision.
RUN python -m pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install xformers --index-url https://download.pytorch.org/whl/cu124 \
    && python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)" \
    && python -c "import torch, os; p=os.path.join(os.path.dirname(torch.__file__), 'lib'); open('/etc/ld.so.conf.d/pytorch-torch-lib.conf','w').write(p + chr(10))" \
    && ldconfig

ARG TRELLIS_REPO=https://github.com/microsoft/TRELLIS.2.git
ARG TRELLIS_REF=main

# Needed for extension compilation (Docker build doesn't have a GPU to auto-detect).
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# STEP 5 — INSTALL DEPENDENCIES
RUN python -m pip install --no-cache-dir \
    ninja imageio imageio-ffmpeg tqdm easydict opencv-python-headless trimesh zstandard kornia timm

# STEP 6 — INSTALL TRANSFORMERS (VERY IMPORTANT)
RUN python -m pip install --upgrade "transformers<5" \
    && python - <<'PY'
from transformers import DINOv3ViTModel
print("DINOv3 OK")
PY

# STEP 7 — INSTALL FLASH-ATTN (use the same prebuilt wheel URL you provided)
RUN python -m pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# STEP 9 — CLONE TRELLIS
RUN git clone --recursive --branch "${TRELLIS_REF}" "${TRELLIS_REPO}" /opt/TRELLIS.2

# STEP 10 — BUILD EXTENSIONS
# - setup.sh checks for `nvidia-smi`; we add a stub so platform=CUDA in Docker build.
RUN rm -rf /tmp/extensions/* \
    && if ! command -v nvidia-smi >/dev/null 2>&1; then printf '%s\n' '#!/bin/sh' 'echo \"nvidia-smi stub\"' > /usr/local/bin/nvidia-smi && chmod +x /usr/local/bin/nvidia-smi; fi \
    && cd /opt/TRELLIS.2 \
    && . ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

# Install/upgrade gguf explicitly (used by GGUF pipelines).
RUN python -m pip install -U gguf

COPY app.py /app/app.py
COPY run.sh /app/run.sh
RUN sed -i '1s/^\xEF\xBB\xBF//' /app/run.sh && sed -i 's/\r$//' /app/run.sh && chmod +x /app/run.sh

ENV PYTHONPATH=/opt/TRELLIS.2:/opt/TRELLIS.2/o-voxel \
    PORT=8000 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TQDM_DISABLE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ATTN_BACKEND=flash_attn \
    ENABLE_S3=1 \
    TRELLIS_IMAGE_SIZE=1024 \
    GLB_TEXTURE_SIZE=2048 \
    MESH_SIMPLIFY_TARGET=500000 \
    MESH_SIMPLIFY_FACES=4000000 \
    MAX_WORKERS=2 \
    MAX_QUEUE_LENGTH=5 \
    MAX_PREVIEW_QUEUE_LENGTH=5

RUN mkdir -p /app/outputs /root/.cache/huggingface

EXPOSE 8000

CMD ["/app/run.sh"]
