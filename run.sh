#!/usr/bin/env bash
# Run hydrilla.co API (TRELLIS.2-4B). Set TRELLIS2_ROOT to repo root if app is not inside it.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer the Conda env created on the host (mounted into the container).
CONDA_ENV_NAME="${CONDA_ENV_NAME:-trellis2}"
CONDA_BASE="${CONDA_BASE:-/home/ubuntu/miniconda3}"
CONDA_BIN="$CONDA_BASE/bin/conda"

# Avoid `source conda.sh` because it can hardcode the original install prefix.
if [ -x "$CONDA_BIN" ]; then
  # Make /opt/conda available too, in case anything expects that path.
  if [ ! -e /opt/conda ]; then
    mkdir -p /opt || true
    ln -s "$CONDA_BASE" /opt/conda || true
  fi
else
  echo "ℹ️ [run.sh] conda not found at $CONDA_BIN; using system python."
fi

# Ensure app deps exist in the selected Python env.
if command -v pip >/dev/null 2>&1 && [ -f /app/requirements.txt ]; then
  if [ -x "$CONDA_BIN" ]; then
    "$CONDA_BIN" run -n "$CONDA_ENV_NAME" python -m pip install -r /app/requirements.txt || true
  else
    pip install -r /app/requirements.txt || true
  fi
fi

# Make sure dynamic linker can find PyTorch's shared libs when o_voxel loads its _C extension.
# In fresh containers, torch's lib directory is not always on LD_LIBRARY_PATH.
if [ -x "$CONDA_BIN" ]; then
  TORCH_LIB_DIR="$("$CONDA_BIN" run -n "$CONDA_ENV_NAME" python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || true)"
  if [ -n "$TORCH_LIB_DIR" ] && [ -d "$TORCH_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH}"
  fi
fi

# Ensure CUDA libs are also reachable.
for p in /usr/local/cuda/lib64 /usr/local/cuda-12.4/lib64; do
  if [ -d "$p" ]; then
    export LD_LIBRARY_PATH="$p:${LD_LIBRARY_PATH}"
  fi
done

if [ -n "$TRELLIS2_ROOT" ]; then
  export PYTHONPATH="$TRELLIS2_ROOT:$PYTHONPATH"
fi
# Use port from env or 8000
PORT="${PORT:-8000}"
echo "Starting hydrilla.co API on 0.0.0.0:$PORT (TRELLIS.2-4B)"
if [ -x "$CONDA_BIN" ]; then
  exec "$CONDA_BIN" run -n "$CONDA_ENV_NAME" python -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
fi

if command -v python >/dev/null 2>&1; then
  exec python -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
fi
exec python3 -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
