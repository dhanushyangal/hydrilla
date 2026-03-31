#!/usr/bin/env bash
# TRELLIS.2 install path for Docker / headless CI: same steps as setup.sh, but:
# - No --new-env (use image Python + torch already installed).
# - No nvidia-smi required when TRELLIS_SETUP_ALLOW_CPU=1 (CUDA cross-compile for extensions).
#
# Usage:
#   export TRELLIS_SETUP_ALLOW_CPU=1
#   export TRELLIS2_ROOT=/opt/TRELLIS.2   # default if unset
#   export TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9
#   ./trellis-setup-docker.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --flexgemm --o-voxel

set -euo pipefail

# Python used for installs/compilation inside this script.
# In Docker we activate the conda env so `python` points to the right interpreter.
PYTHON_BIN="${PYTHON_BIN:-python}"

TEMP="$(getopt -o h --long help,new-env,basic,flash-attn,cumesh,o-voxel,flexgemm,nvdiffrast,nvdiffrec -n 'trellis-setup-docker.sh' -- "$@")" || exit 1
eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
FLASHATTN=false
CUMESH=false
OVOXEL=false
FLEXGEMM=false
NVDIFFRAST=false
NVDIFFREC=false
ERROR=false

while true; do
  case "$1" in
    -h|--help) HELP=true; shift ;;
    --new-env) NEW_ENV=true; shift ;;
    --basic) BASIC=true; shift ;;
    --flash-attn) FLASHATTN=true; shift ;;
    --cumesh) CUMESH=true; shift ;;
    --o-voxel) OVOXEL=true; shift ;;
    --flexgemm) FLEXGEMM=true; shift ;;
    --nvdiffrast) NVDIFFRAST=true; shift ;;
    --nvdiffrec) NVDIFFREC=true; shift ;;
    --) shift; break ;;
    *) ERROR=true; break ;;
  esac
done

if [ "$ERROR" = true ]; then
  echo "Error: Invalid argument"
  HELP=true
fi

if [ "$HELP" = true ]; then
  echo "Usage: trellis-setup-docker.sh [OPTIONS]"
  echo "  TRELLIS2_ROOT=/path         TRELLIS.2 clone (default: /opt/TRELLIS.2)"
  echo "  TRELLIS_SETUP_ALLOW_CPU=1   Required during docker build (no GPU / no nvidia-smi)."
  echo "  --new-env                   Ignored in Docker (conda not used); torch must be preinstalled."
  echo "Options match TRELLIS.2 setup.sh: --basic --flash-attn --cumesh --o-voxel --flexgemm --nvdiffrast --nvdiffrec"
  exit 0
fi

WORKDIR="$(cd "${TRELLIS2_ROOT:-/opt/TRELLIS.2}" && pwd)"
EXT_ROOT="${TRELLIS_EXTENSIONS_ROOT:-/tmp/trellis-extensions}"
mkdir -p "$EXT_ROOT"

# Platform: setup.sh exits if no GPU; we allow forcing CUDA for nvcc-only builds.
if [ "${TRELLIS_SETUP_ALLOW_CPU:-}" = "1" ]; then
  PLATFORM="cuda"
elif command -v nvidia-smi >/dev/null 2>&1; then
  PLATFORM="cuda"
elif command -v rocminfo >/dev/null 2>&1; then
  PLATFORM="hip"
else
  echo "Error: No supported GPU found. For Docker image builds set TRELLIS_SETUP_ALLOW_CPU=1"
  exit 1
fi

if [ "$NEW_ENV" = true ]; then
  echo "[trellis-setup-docker] Note: --new-env ignored (use preinstalled Python in container)."
fi

cd "$WORKDIR"

if [ "$BASIC" = true ]; then
  "$PYTHON_BIN" -m pip install --no-cache-dir \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers \
    gradio==6.0.1 tensorboard pandas lpips zstandard
  "$PYTHON_BIN" -m pip install --no-cache-dir \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
  apt-get update
  apt-get install -y --no-install-recommends libjpeg-dev
  rm -rf /var/lib/apt/lists/*
  "$PYTHON_BIN" -m pip install --no-cache-dir pillow-simd || "$PYTHON_BIN" -m pip install --no-cache-dir Pillow
  "$PYTHON_BIN" -m pip install --no-cache-dir kornia timm
fi

if [ "$FLASHATTN" = true ]; then
  if [ "$PLATFORM" = "cuda" ]; then
    # Prefer the prebuilt wheel for CUDA 12.4 / torch 2.6 / cp310 (faster than compiling).
    "$PYTHON_BIN" -m pip install --no-cache-dir \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
  elif [ "$PLATFORM" = "hip" ]; then
    echo "[FLASHATTN] ROCm path not used in hydrilla Docker image."
    exit 1
  fi
fi

if [ "$NVDIFFRAST" = true ]; then
  if [ "$PLATFORM" = "cuda" ]; then
    rm -rf "$EXT_ROOT/nvdiffrast"
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git "$EXT_ROOT/nvdiffrast"
    "$PYTHON_BIN" -m pip install --no-cache-dir "$EXT_ROOT/nvdiffrast" --no-build-isolation
  else
    echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
    exit 1
  fi
fi

if [ "$NVDIFFREC" = true ]; then
  if [ "$PLATFORM" = "cuda" ]; then
    rm -rf "$EXT_ROOT/nvdiffrec"
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git "$EXT_ROOT/nvdiffrec"
    "$PYTHON_BIN" -m pip install --no-cache-dir "$EXT_ROOT/nvdiffrec" --no-build-isolation
  else
    echo "[NVDIFFREC] Unsupported platform: $PLATFORM"
    exit 1
  fi
fi

if [ "$CUMESH" = true ]; then
  rm -rf "$EXT_ROOT/CuMesh"
  git clone --recursive https://github.com/JeffreyXiang/CuMesh.git "$EXT_ROOT/CuMesh"
  "$PYTHON_BIN" -m pip install --no-cache-dir "$EXT_ROOT/CuMesh" --no-build-isolation
fi

if [ "$FLEXGEMM" = true ]; then
  rm -rf "$EXT_ROOT/FlexGEMM"
  git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git "$EXT_ROOT/FlexGEMM"
  "$PYTHON_BIN" -m pip install --no-cache-dir "$EXT_ROOT/FlexGEMM" --no-build-isolation
fi

if [ "$OVOXEL" = true ]; then
  "$PYTHON_BIN" -m pip install --no-cache-dir --no-build-isolation -v -e "$WORKDIR/o-voxel"
fi

echo "[trellis-setup-docker] Done."
