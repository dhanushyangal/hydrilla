"""
hydrilla.co API – Image-to-3D and Text-to-3D with Microsoft TRELLIS.2-4B.
Same API surface as app_improved (image gen, jobs, S3, preview, edit) but 3D via Trellis2.
Run from TRELLIS.2 repo root or set PYTHONPATH to include TRELLIS.2.
"""
import sys
import os

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
# Hugging Face token: set HF_TOKEN or use `huggingface-cli login` (stored in ~/.cache/huggingface/token)
if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])

# tqdm/HF progress uses CR/ANSI; journald often stores those lines as "[NB blob data]"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

# Avoid "Tensor.item() cannot be called on meta tensors" when TRELLIS.2 loads DinoV3/vision models
try:
    import transformers
    _cm = getattr(transformers.PreTrainedModel.from_pretrained, "__func__", None) or transformers.PreTrainedModel.from_pretrained
    _orig_fp = _cm
    def _from_pretrained_no_meta(cls, pretrained_model_name_or_path, *args, low_cpu_mem_usage=None, device_map=None, **kwargs):
        kwargs["low_cpu_mem_usage"] = False if low_cpu_mem_usage is None else low_cpu_mem_usage
        kwargs["device_map"] = None if device_map is None else device_map
        return _orig_fp(cls, pretrained_model_name_or_path, *args, **kwargs)
    transformers.PreTrainedModel.from_pretrained = classmethod(_from_pretrained_no_meta)
except Exception:
    pass
try:
    import torch
    _orig_item = torch.Tensor.item
    def _patched_item(self):
        if getattr(self, "is_meta", False):
            return 0
        return _orig_item(self)
    torch.Tensor.item = _patched_item
except Exception:
    pass

import uuid
import time
import requests
import asyncio
import json
import shutil
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
from fastapi import FastAPI, Form, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlparse
from typing import Dict, Optional
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Thread
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as BotocoreConfig

# -------------------------
# 3D: TRELLIS.2-4B (microsoft/TRELLIS.2-4B)
# -------------------------
TRELLIS2_PIPELINE = None
O_VOXEL_AVAILABLE = False
try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel
    O_VOXEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ [WARNING] Trellis2/o_voxel not found. Set PYTHONPATH to TRELLIS.2 repo root. Error: {e}")

# Background removal
rembg_session = None
REMBG_AVAILABLE = False
try:
    from rembg import new_session, remove
    REMBG_AVAILABLE = True
    print("✅ [SERVER] rembg loaded (GPU background removal)")
except ImportError:
    print("⚠️ [WARNING] rembg not available. pip install rembg[gpu]")

# Text-to-image (Z-Image-Turbo only; SDXL removed)
import torch
TORCH_AVAILABLE = True
Z_IMAGE_TURBO_AVAILABLE = False
Z_IMAGE_GGUF_AVAILABLE = False
_z_image_import_error = None
try:
    from diffusers import ZImagePipeline, ZImageImg2ImgPipeline
    Z_IMAGE_TURBO_AVAILABLE = True
    try:
        from diffusers import ZImageTransformer2DModel, GGUFQuantizationConfig
        Z_IMAGE_GGUF_AVAILABLE = True
    except Exception:
        pass
except Exception as e:
    _z_image_import_error = e

app = FastAPI(title="Hydrilla.co – TRELLIS.2-4B Image/Text to 3D API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    # Expose model readiness so you can confirm containers are correctly loading
    # Z-Image (image generation) + TRELLIS.2 (3D generation).
    z_image_enabled = (
        TEXT_TO_IMAGE_TYPE is not None
        and TEXT_TO_IMAGE_TYPE == "z_image_turbo"
        and z_image_pipeline is not None
    )
    trellis_enabled = TRELLIS2_PIPELINE is not None and O_VOXEL_AVAILABLE

    return {
        "status": "ok",
        "api": "hydrilla-co",
        # If TRELLIS isn't loaded, 3D endpoints will fail; keep this for quick monitoring.
        "fallback": not trellis_enabled,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "rembg_available": REMBG_AVAILABLE,
        "o_voxel_available": O_VOXEL_AVAILABLE,
        "z_image_turbo_loaded": z_image_enabled,
        "trellis2_pipeline_loaded": TRELLIS2_PIPELINE is not None,
        "attention_backend": os.getenv("ATTN_BACKEND", ""),
    }

OUTPUT_BASE = "outputs"
os.makedirs(OUTPUT_BASE, exist_ok=True)

# -------------------------
# Config – Quality presets (for text-to-image only; Trellis2 has no steps)
# -------------------------
# Resolution capped at 768 for small instances (no 1024 to avoid VRAM spike)
QUALITY_PRESETS = {
    "fast": {"t2i_steps": 40, "t2i_guidance": 8.0, "t2i_resolution": 512},
    "standard": {"t2i_steps": 60, "t2i_guidance": 9.0, "t2i_resolution": 768},
    "high": {"t2i_steps": 80, "t2i_guidance": 10.0, "t2i_resolution": 768},
    "ultra": {"t2i_steps": 100, "t2i_guidance": 11.0, "t2i_resolution": 768},
}
DEFAULT_QUALITY = os.getenv("DEFAULT_QUALITY", "standard")
HQ_CONFIG = QUALITY_PRESETS.get(DEFAULT_QUALITY, QUALITY_PRESETS["standard"])

OPTIMAL_IMAGE_SIZE = (768, 768)
MAX_IMAGE_SIZE = (2048, 2048)
MIN_IMAGE_SIZE = (256, 256)

# S3 – same bucket/region as flux_trellis_deploy gateway (hydrilla-outputs, ap-south-1).
# Credentials must come from environment variables (or IAM role).
S3_BUCKET = os.getenv("S3_BUCKET", "hydrilla-outputs")
S3_REGION = os.getenv("S3_REGION", "ap-south-1")
S3_PRESIGNED_URL_EXPIRY = int(os.getenv("S3_PRESIGNED_URL_EXPIRY", "3600"))
s3 = None
S3_ENABLED = False
# S3 timeouts so slow network doesn't block the single worker forever
_S3_CONFIG = BotocoreConfig(connect_timeout=15, read_timeout=120, retries={"max_attempts": 2, "mode": "standard"})
if os.getenv("ENABLE_S3", "1").strip().lower() in ("1", "true", "yes") and S3_BUCKET:
    try:
        # boto3 uses AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY from env (or IAM role)
        s3 = boto3.client("s3", region_name=S3_REGION, config=_S3_CONFIG)
        s3.head_bucket(Bucket=S3_BUCKET)
        S3_ENABLED = True
        print(f"✅ [SERVER] S3 connected: {S3_BUCKET} ({S3_REGION})")
    except Exception as e:
        print(f"⚠️ [WARNING] S3 disabled: {e}")
else:
    if not S3_BUCKET:
        print("ℹ️ [SERVER] S3 disabled (set S3_BUCKET and ENABLE_S3=1 to enable)")
    else:
        print("ℹ️ [SERVER] S3 disabled (set ENABLE_S3=1 and AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY to enable)")

# Log HF token status (never log the token value)
if os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"):
    print("✅ [SERVER] Hugging Face token set (faster downloads)")

def _get_trellis2_snapshot_path():
    """Return path to TRELLIS.2-4B snapshot in HF cache, or None."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snapshots_dir = os.path.join(hf_home, "hub", "models--microsoft--TRELLIS.2-4B", "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    revs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    return os.path.join(snapshots_dir, revs[0]) if revs else None


# Text-to-image config (pipelines loaded after TRELLIS so 3D gets GPU first)
z_image_pipeline = None
z_image_img2img_pipeline = None
TEXT_TO_IMAGE_TYPE = None
USE_Z_IMAGE_TURBO = os.getenv("USE_Z_IMAGE_TURBO", "1").strip().lower() in ("1", "true", "yes")
Z_IMAGE_USE_GGUF = os.getenv("Z_IMAGE_USE_GGUF", "1").strip().lower() in ("1", "true", "yes")
Z_IMAGE_GGUF_REPO = os.getenv("Z_IMAGE_GGUF_REPO", "unsloth/Z-Image-Turbo-GGUF").strip()
Z_IMAGE_GGUF_FILE = os.getenv("Z_IMAGE_GGUF_FILE", "z-image-turbo-Q4_K_S.gguf").strip()  # Q4_K_S uses less VRAM
Z_IMAGE_TURBO_MODEL = os.getenv("Z_IMAGE_TURBO_MODEL", "Tongyi-MAI/Z-Image-Turbo").strip()
Z_IMAGE_BASE_REPO = "Tongyi-MAI/Z-Image-Turbo"
# GGUF only: no full-model fallback (keeps startup light; set Z_IMAGE_USE_GGUF=1 and install gguf>=0.10)
# Cap at 768 for small GPU (1024 causes VRAM spike)
Z_IMAGE_T2I_RESOLUTION = max(512, min(768, int(os.getenv("Z_IMAGE_T2I_RESOLUTION", "768"))))
# Match api_trellis: 1024 for mesh quality; lower to 512/384 if GPU OOM
TRELLIS_IMAGE_SIZE = max(256, min(1024, int(os.getenv("TRELLIS_IMAGE_SIZE", "1024"))))
cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# rembg first (light), then image generation (Z-Image), then TRELLIS (3D)
if REMBG_AVAILABLE:
    for model in ("u2net", "isnet-general-use", "InSPyReNet"):
        try:
            rembg_session = new_session(model)
            print(f"✅ [SERVER] rembg ({model}) loaded")
            break
        except Exception as e:
            if model == "InSPyReNet":
                print(f"⚠️ [WARNING] rembg init failed: {e}")
            continue

# Load Z-Image (image generation) first
if Z_IMAGE_TURBO_AVAILABLE and TORCH_AVAILABLE and USE_Z_IMAGE_TURBO:
    if TORCH_AVAILABLE:
        try:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
    gguf_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    _z_load_error = None
    if Z_IMAGE_USE_GGUF and Z_IMAGE_GGUF_AVAILABLE:
        try:
            print("🚀 [SERVER] Loading Z-Image-Turbo (GGUF) for text-to-image...")
            print(f"   📦 GGUF repo: {Z_IMAGE_GGUF_REPO}, file: {Z_IMAGE_GGUF_FILE}")
            from huggingface_hub import hf_hub_download
            gguf_path = hf_hub_download(
                repo_id=Z_IMAGE_GGUF_REPO,
                filename=Z_IMAGE_GGUF_FILE,
                cache_dir=cache_dir,
                local_files_only=False,
            )
            transformer = ZImageTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=gguf_dtype),
                torch_dtype=gguf_dtype,
            )
            transformer = transformer.to("cuda")
            z_image_pipeline = ZImagePipeline.from_pretrained(
                Z_IMAGE_BASE_REPO,
                transformer=transformer,
                torch_dtype=gguf_dtype,
                cache_dir=cache_dir,
            )
            z_image_pipeline = z_image_pipeline.to("cuda")
            z_image_img2img_pipeline = None
            TEXT_TO_IMAGE_TYPE = "z_image_turbo"
            print("✅ [SERVER] Z-Image-Turbo (GGUF) loaded - text-to-image enabled")
        except Exception as e:
            _z_load_error = e
            print(f"⚠️ [WARNING] Z-Image-Turbo GGUF load failed: {e}")
            print("   💡 GGUF only (no full-model fallback). pip install -U gguf; use diffusers with Z-Image GGUF support.")
            z_image_pipeline = None
            z_image_img2img_pipeline = None
    if not Z_IMAGE_USE_GGUF or not Z_IMAGE_GGUF_AVAILABLE:
        if Z_IMAGE_TURBO_AVAILABLE and USE_Z_IMAGE_TURBO and TEXT_TO_IMAGE_TYPE is None:
            print("ℹ️ [SERVER] Z-Image: GGUF only. Set Z_IMAGE_USE_GGUF=1 and ensure GGUF support (diffusers + gguf>=0.10).")

if TEXT_TO_IMAGE_TYPE is None:
    if not Z_IMAGE_TURBO_AVAILABLE:
        print("ℹ️ [SERVER] Image generation not loaded: Z-Image-Turbo not available.")
        if _z_image_import_error is not None:
            print(f"   Reason: diffusers ZImagePipeline import failed: {_z_image_import_error}")
    elif not USE_Z_IMAGE_TURBO:
        print("ℹ️ [SERVER] Image generation not loaded: USE_Z_IMAGE_TURBO=0")
    else:
        print("ℹ️ [SERVER] Image generation not loaded (Z-Image-Turbo load failed). Text-to-3D needs image_url.")

# Load TRELLIS.2-4B (3D) after image generation
if O_VOXEL_AVAILABLE:
    try:
        if TORCH_AVAILABLE:
            try:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        print("🚀 [SERVER] Loading TRELLIS.2-4B pipeline...")
        snapshot_path = _get_trellis2_snapshot_path()
        if snapshot_path and os.path.isfile(os.path.join(snapshot_path, "pipeline.json")):
            snapshot_path = os.path.abspath(snapshot_path)
            print(f"   From cache: {snapshot_path}")
            TRELLIS2_PIPELINE = Trellis2ImageTo3DPipeline.from_pretrained(snapshot_path)
        else:
            TRELLIS2_PIPELINE = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        TRELLIS2_PIPELINE.cuda()
        print("✅ [SERVER] TRELLIS.2-4B loaded on GPU")
    except Exception as e:
        print(f"❌ [ERROR] Failed to load TRELLIS.2-4B: {e}")
        TRELLIS2_PIPELINE = None
else:
    print("⚠️ [WARNING] TRELLIS.2 not available; 3D endpoints will fail.")

# -------------------------
# Threading & GPU
# -------------------------
# Worker threads for background jobs (set MAX_WORKERS=1 on small instance to avoid CPU thrash)
EXECUTOR = ThreadPoolExecutor(max_workers=max(1, int(os.getenv("MAX_WORKERS", "2"))))
# Cap queue length so server doesn't accept unbounded work (prevents OOM and stress)
MAX_QUEUE_LENGTH = max(1, int(os.getenv("MAX_QUEUE_LENGTH", "5")))
MAX_PREVIEW_QUEUE_LENGTH = max(1, int(os.getenv("MAX_PREVIEW_QUEUE_LENGTH", "5")))
THREE_D_LOCK = Lock()
PREVIEW_LOCK = Lock()
JOBS_LOCK = Lock()
cancelled_jobs = set()
CANCELLED_JOBS_LOCK = Lock()
QUEUE_LOCK = Lock()
job_queue = []
currently_processing_3d = None
preview_queue = []
currently_generating_preview = None
ESTIMATED_3D_TIME = int(os.environ.get("ESTIMATED_3D_TIME_SECONDS", "90"))  # Trellis2 ~3–60s
ESTIMATED_PREVIEW_TIME = int(os.environ.get("ESTIMATED_PREVIEW_TIME_SECONDS", "12" if TEXT_TO_IMAGE_TYPE == "z_image_turbo" else "25"))
# 3D/GLB quality (env overrides); 500k/4M/3072 balance quality vs VRAM
MESH_SIMPLIFY_TARGET = max(50000, min(8000000, int(os.environ.get("MESH_SIMPLIFY_TARGET", "500000"))))
MESH_SIMPLIFY_FACES = int(os.environ.get("MESH_SIMPLIFY_FACES", "4000000"))
GLB_TEXTURE_SIZE = max(256, min(4096, int(os.environ.get("GLB_TEXTURE_SIZE", "2048"))))

def get_queue_position(job_id: str):
    with QUEUE_LOCK:
        if job_id == currently_processing_3d:
            return (0, 0, 0)
        if job_id in job_queue:
            position = job_queue.index(job_id) + 1
            jobs_ahead = position - 1 + (1 if currently_processing_3d else 0)
            return (position, jobs_ahead * ESTIMATED_3D_TIME, jobs_ahead)
        return (-1, 0, 0)

def add_to_queue(job_id: str):
    with QUEUE_LOCK:
        if job_id not in job_queue:
            job_queue.append(job_id)

def remove_from_queue(job_id: str):
    global currently_processing_3d
    with QUEUE_LOCK:
        if job_id in job_queue:
            job_queue.remove(job_id)
        if currently_processing_3d == job_id:
            currently_processing_3d = None

def start_processing_3d(job_id: str):
    global currently_processing_3d
    with QUEUE_LOCK:
        if job_id in job_queue:
            job_queue.remove(job_id)
        currently_processing_3d = job_id

def get_preview_queue_position(preview_id: str):
    with QUEUE_LOCK:
        if preview_id == currently_generating_preview:
            return (0, 0, 0)
        if preview_id in preview_queue:
            position = preview_queue.index(preview_id) + 1
            ahead = position - 1 + (1 if currently_generating_preview else 0)
            return (position, ahead * ESTIMATED_PREVIEW_TIME, ahead)
        return (-1, 0, 0)

def add_to_preview_queue(preview_id: str):
    with QUEUE_LOCK:
        if preview_id not in preview_queue:
            preview_queue.append(preview_id)

def remove_from_preview_queue(preview_id: str):
    global currently_generating_preview
    with QUEUE_LOCK:
        if preview_id in preview_queue:
            preview_queue.remove(preview_id)
        if currently_generating_preview == preview_id:
            currently_generating_preview = None

def start_processing_preview(preview_id: str):
    global currently_generating_preview
    with QUEUE_LOCK:
        if preview_id in preview_queue:
            preview_queue.remove(preview_id)
        currently_generating_preview = preview_id

def _run_with_gpu_cleanup(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and always clear GPU in finally to avoid OOM on next job."""
    try:
        return fn(*args, **kwargs)
    finally:
        if TORCH_AVAILABLE:
            try:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

def get_queue_info():
    with QUEUE_LOCK:
        jobs_ahead_3d = len(job_queue) + (1 if currently_processing_3d else 0)
        return {
            "queue_length": len(job_queue),
            "currently_processing": currently_processing_3d is not None,
            "processing_job_id": currently_processing_3d,
            "waiting_jobs": len(job_queue),
            "jobs_ahead_for_new": jobs_ahead_3d,
            "estimated_wait_for_new_job": jobs_ahead_3d * ESTIMATED_3D_TIME,
            "estimated_total_seconds": jobs_ahead_3d * ESTIMATED_3D_TIME + ESTIMATED_3D_TIME,
            "estimated_time_per_job_seconds": ESTIMATED_3D_TIME,
            "preview_queue_length": len(preview_queue),
            "currently_generating_preview": currently_generating_preview is not None,
            "preview_waiting": len(preview_queue),
            "estimated_wait_for_preview": (len(preview_queue) + (1 if currently_generating_preview else 0)) * ESTIMATED_PREVIEW_TIME,
            "estimated_preview_time_seconds": ESTIMATED_PREVIEW_TIME,
        }

# -------------------------
# Jobs
# -------------------------
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

jobs: Dict[str, Dict] = {}
user_jobs: Dict[str, set] = {}
JOBS_PERSISTENCE_FILE = os.path.join(OUTPUT_BASE, "jobs.json")

def is_job_cancelled(job_id: str):
    with CANCELLED_JOBS_LOCK:
        return job_id in cancelled_jobs

def mark_job_cancelled(job_id: str):
    with CANCELLED_JOBS_LOCK:
        cancelled_jobs.add(job_id)

def load_jobs_from_disk():
    try:
        if os.path.exists(JOBS_PERSISTENCE_FILE):
            with open(JOBS_PERSISTENCE_FILE, "r") as f:
                data = json.load(f)
            with JOBS_LOCK:
                jobs.update(data)
                for jid, job in data.items():
                    uid = job.get("user_id")
                    if uid:
                        user_jobs.setdefault(uid, set()).add(jid)
            print(f"✅ [SERVER] Loaded {len(data)} jobs from disk")
            return len(data)
    except Exception as e:
        print(f"⚠️ [WARNING] Load jobs failed: {e}")
    return 0

def save_jobs_to_disk():
    try:
        with JOBS_LOCK:
            data = {}
            for jid, job in jobs.items():
                data[jid] = {k: job.get(k) for k in ["job_id", "user_id", "status", "progress", "message", "created_at", "updated_at", "result", "error"] if job.get(k) is not None}
        with open(JOBS_PERSISTENCE_FILE + ".tmp", "w") as f:
            json.dump(data, f, indent=2)
        if os.path.exists(JOBS_PERSISTENCE_FILE):
            os.replace(JOBS_PERSISTENCE_FILE + ".tmp", JOBS_PERSISTENCE_FILE)
        else:
            os.rename(JOBS_PERSISTENCE_FILE + ".tmp", JOBS_PERSISTENCE_FILE)
    except Exception as e:
        print(f"⚠️ [WARNING] Save jobs failed: {e}")

def update_job_status(job_id: str, status: JobStatus, progress: int = 0, message: str = "", result: Optional[Dict] = None, error: Optional[str] = None, user_id: Optional[str] = None):
    with JOBS_LOCK:
        if job_id not in jobs:
            jobs[job_id] = {"job_id": job_id, "user_id": user_id, "status": status.value, "progress": 0, "message": "", "created_at": time.time(), "result": None, "error": None}
            if user_id:
                user_jobs.setdefault(user_id, set()).add(job_id)
        jobs[job_id]["status"] = status.value
        jobs[job_id]["progress"] = progress
        jobs[job_id]["message"] = message
        jobs[job_id]["updated_at"] = time.time()
        if result:
            jobs[job_id]["result"] = result
        if error:
            jobs[job_id]["error"] = error
            if status != JobStatus.CANCELLED:
                jobs[job_id]["status"] = JobStatus.FAILED.value
    save_jobs_to_disk()

# -------------------------
# S3
# -------------------------
def upload_and_presign(local_path: str, s3_key: str, expires: int = None):
    if not S3_ENABLED or s3 is None:
        return None
    if expires is None:
        expires = S3_PRESIGNED_URL_EXPIRY
    try:
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        return s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": s3_key}, ExpiresIn=expires)
    except Exception as e:
        print(f"❌ [ERROR] S3 upload failed: {e}")
        return None

# -------------------------
# Image helpers (from app_improved)
# -------------------------
def is_valid_url(url: str):
    try:
        r = urlparse(url)
        return bool(r.scheme and r.netloc)
    except Exception:
        return False

def optimize_image_for_3d(image: Image.Image) -> Image.Image:
    orig_mode = image.mode
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    w, h = image.size
    if w > MAX_IMAGE_SIZE[0] or h > MAX_IMAGE_SIZE[1]:
        image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    elif w < MIN_IMAGE_SIZE[0] or h < MIN_IMAGE_SIZE[1]:
        ratio = max(MIN_IMAGE_SIZE[0] / w, MIN_IMAGE_SIZE[1] / h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    if image.size != OPTIMAL_IMAGE_SIZE:
        cw, ch = image.size
        tw, th = OPTIMAL_IMAGE_SIZE
        img_aspect, ta = cw / ch, tw / th
        if abs(img_aspect - ta) > 0.01:
            if img_aspect > ta:
                nw, nh = tw, int(ch * tw / cw)
            else:
                nw, nh = int(cw * th / ch), th
            image = image.resize((nw, nh), Image.Resampling.LANCZOS)
            bg = Image.new("RGBA" if image.mode == "RGBA" else "RGB", OPTIMAL_IMAGE_SIZE, (240, 240, 240, 0) if image.mode == "RGBA" else (240, 240, 240))
            x, y = (tw - nw) // 2, (th - nh) // 2
            if image.mode == "RGBA":
                bg.paste(image, (x, y), image)
            else:
                bg.paste(image, (x, y))
            image = bg
        else:
            image = image.resize(OPTIMAL_IMAGE_SIZE, Image.Resampling.LANCZOS)
    try:
        from PIL import ImageStat
        stat = ImageStat.Stat(image)
        mean_b = sum(stat.mean) / len(stat.mean)
        cf = 1.15 if mean_b < 100 else (1.05 if mean_b > 200 else 1.1)
        image = ImageEnhance.Contrast(image).enhance(cf)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=2))
        image = ImageEnhance.Color(image).enhance(1.1)
    except Exception:
        pass
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image

def apply_exif_orientation(image: Image.Image) -> Image.Image:
    try:
        exif = image.getexif()
        if not exif:
            return image
        o = exif.get(274)
        if not o or o == 1:
            return image
        trans = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
                 5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}
        if o in trans:
            image = image.transpose(trans[o])
        return image
    except Exception:
        return image

# Max download size and max dimension to avoid OOM on small instances
MAX_IMAGE_DOWNLOAD_BYTES = int(os.getenv("MAX_IMAGE_DOWNLOAD_BYTES", str(20 * 1024 * 1024)))  # 20MB
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "2048"))
# On minimal EC2/Ubuntu, CA certs can be missing → SSL verify fails. Set IMAGE_DOWNLOAD_VERIFY_SSL=0 to skip.
IMAGE_DOWNLOAD_VERIFY_SSL = os.getenv("IMAGE_DOWNLOAD_VERIFY_SSL", "1").strip().lower() not in ("0", "false", "no")
if not IMAGE_DOWNLOAD_VERIFY_SSL:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_image_from_url(url: str) -> Image.Image:
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL: {url}")
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}, stream=True, verify=IMAGE_DOWNLOAD_VERIFY_SSL)
    r.raise_for_status()
    content = b""
    for chunk in r.iter_content(chunk_size=65536):
        content += chunk
        if len(content) > MAX_IMAGE_DOWNLOAD_BYTES:
            raise ValueError(f"Image larger than {MAX_IMAGE_DOWNLOAD_BYTES // (1024*1024)}MB")
    image = Image.open(BytesIO(content))
    image.verify()
    image = Image.open(BytesIO(content))
    image = apply_exif_orientation(image)
    if max(image.size) > MAX_IMAGE_DIMENSION:
        ratio = MAX_IMAGE_DIMENSION / max(image.size)
        new_size = (max(1, int(image.size[0] * ratio)), max(1, int(image.size[1] * ratio)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    return image

def download_image_url_to_temp_path(url: str) -> str:
    """Download image from URL with size/dimension limits; save to temp file; return path."""
    image = load_image_from_url(url)
    temp_dir = os.path.join(OUTPUT_BASE, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, f"{uuid.uuid4()}.png")
    image.save(path, "PNG", optimize=True)
    return path

def remove_background(image: Image.Image) -> Image.Image:
    if not REMBG_AVAILABLE or rembg_session is None:
        return image
    return remove(image, session=rembg_session)

def enhance_prompt_for_3d(prompt: str) -> str:
    return f"3D render of {prompt.strip()}, front view, centered, white background, studio lighting"

def text_to_image(prompt: str, job_id: str = None) -> Image.Image:
    """Generate image from text. Z-Image-Turbo only: 9 steps, guidance_scale=0."""
    if TEXT_TO_IMAGE_TYPE != "z_image_turbo" or z_image_pipeline is None:
        raise ValueError(
            "No text-to-image model available. Enable Z-Image-Turbo (USE_Z_IMAGE_TURBO=1) and ensure Z-Image-Turbo loads."
        )
    try:
        original_prompt = prompt.strip()
        enhanced_prompt = enhance_prompt_for_3d(original_prompt)
        print(f"🎨 [SERVER] Generating image from text using Z-Image-Turbo...")
        print(f"   📝 Prompt: '{enhanced_prompt[:150]}...'")
        preset_res = HQ_CONFIG.get("t2i_resolution", 768)
        target_size = max(512, min(preset_res, Z_IMAGE_T2I_RESOLUTION))
        print(f"   📐 Resolution: {target_size}x{target_size} (preset={preset_res}, cap={Z_IMAGE_T2I_RESOLUTION})")
        with torch.no_grad():
            result = z_image_pipeline(
                prompt=enhanced_prompt,
                height=target_size,
                width=target_size,
                num_inference_steps=9,
                guidance_scale=0.0,
                generator=torch.Generator("cuda").manual_seed(int(time.time() * 1000) % (2**32)) if job_id is None else torch.Generator("cuda").manual_seed(42),
            )
        image = result.images[0]
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        print(f"✅ [SERVER] Image generated (Z-Image-Turbo): {image.size}, mode: {image.mode}")
        return image
    except Exception as e:
        raise ValueError(f"Failed to generate image from text (Z-Image-Turbo): {str(e)}")

def edit_image_with_prompt(image: Image.Image, prompt: str, strength: float = 0.6) -> Image.Image:
    """Edit image using Z-Image-Turbo img2img. Same as app_improved."""
    if z_image_img2img_pipeline is None or TEXT_TO_IMAGE_TYPE != "z_image_turbo":
        raise ValueError(
            "Image edit requires Z-Image-Turbo. Set USE_Z_IMAGE_TURBO=1 and ensure Z-Image-Turbo (FP8) loaded."
        )
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Edit prompt cannot be empty")
    if image.size != (768, 768):
        image = image.resize((768, 768), Image.Resampling.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")
    with torch.no_grad():
        result = z_image_img2img_pipeline(
            prompt=prompt,
            image=image,
            strength=float(max(0.01, min(1.0, strength))),
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(int(time.time() * 1000) % (2**32)),
        )
    out = result.images[0]
    if out.mode != "RGBA":
        out = out.convert("RGBA")
    return out

# -------------------------
# 3D: Trellis2 – run pipeline and export GLB via o_voxel
# -------------------------
def _run_trellis2_and_export_glb(image: Image.Image, out_path: str) -> None:
    """Run TRELLIS.2-4B image-to-3D and save mesh as GLB. Requires TRELLIS2_PIPELINE and o_voxel."""
    if TRELLIS2_PIPELINE is None or not O_VOXEL_AVAILABLE:
        raise RuntimeError("TRELLIS.2 pipeline not loaded. Set PYTHONPATH to TRELLIS.2 repo.")
    image_rgb = image.convert("RGB")
    # Skip pipeline preprocess (rembg): we already did rembg + composite; pipeline.rembg_model may be None
    with torch.inference_mode():
        mesh = TRELLIS2_PIPELINE.run(image_rgb, preprocess_image=False)[0]
    mesh.simplify(MESH_SIMPLIFY_FACES)
    if TORCH_AVAILABLE:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    # remesh=True; extension_webp=False (PNG) so export works when Pillow has no WebP (e.g. ~/.local)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=MESH_SIMPLIFY_TARGET,
        texture_size=GLB_TEXTURE_SIZE,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    glb.export(out_path, extension_webp=False)


# -------------------------
# Background: image-to-3d
# -------------------------
def process_image_to_3d(job_id: str, image_url: str):
    start = time.time()
    add_to_queue(job_id)
    try:
        if is_job_cancelled(job_id):
            remove_from_queue(job_id)
            update_job_status(job_id, JobStatus.CANCELLED, 0, "Cancelled", error="Job cancelled by user")
            return
        update_job_status(job_id, JobStatus.PROCESSING, 10, "Downloading image...")
        image = load_image_from_url(image_url.strip())
        if is_job_cancelled(job_id):
            update_job_status(job_id, JobStatus.CANCELLED, 0, "Cancelled", error="Job cancelled by user")
            return
        update_job_status(job_id, JobStatus.PROCESSING, 30, "Removing background...")
        image = remove_background(image)
        if is_job_cancelled(job_id):
            update_job_status(job_id, JobStatus.CANCELLED, 0, "Cancelled", error="Job cancelled by user")
            return
        update_job_status(job_id, JobStatus.PROCESSING, 40, "Optimizing image...")
        image = optimize_image_for_3d(image)
        # RGB for pipeline (same as api_trellis .ai) so 3D mesh gets correct colors; composite on white
        if image.mode == "RGBA":
            bg_white = Image.new("RGB", image.size, (255, 255, 255))
            bg_white.paste(image, mask=image.split()[-1])
            image_rgb_for_trellis = bg_white.convert("RGB")
        else:
            image_rgb_for_trellis = image.convert("RGB")
        if max(image_rgb_for_trellis.size) > TRELLIS_IMAGE_SIZE:
            ratio = TRELLIS_IMAGE_SIZE / max(image_rgb_for_trellis.size)
            new_sz = (max(1, int(image_rgb_for_trellis.size[0] * ratio)), max(1, int(image_rgb_for_trellis.size[1] * ratio)))
            image_rgb_for_trellis = image_rgb_for_trellis.resize(new_sz, Image.Resampling.LANCZOS)
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (240, 240, 240))
            bg.paste(image, mask=image.split()[-1])
            image = bg.convert("RGB")
        out_dir = os.path.join(OUTPUT_BASE, "image", job_id)
        os.makedirs(out_dir, exist_ok=True)
        processed_image_path = os.path.join(out_dir, "processed_image.png")
        image.save(processed_image_path, "PNG", optimize=True)
        queue_pos, est_wait, jobs_ahead = get_queue_position(job_id)
        if jobs_ahead > 0:
            update_job_status(job_id, JobStatus.PROCESSING, 50, f"Waiting in queue ({jobs_ahead} ahead)...")
        else:
            update_job_status(job_id, JobStatus.PROCESSING, 50, "Generating 3D mesh (TRELLIS.2)...")
        if is_job_cancelled(job_id):
            remove_from_queue(job_id)
            update_job_status(job_id, JobStatus.CANCELLED, 50, "Cancelled", error="Job cancelled by user")
            return
        if TORCH_AVAILABLE:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        with THREE_D_LOCK:
            start_processing_3d(job_id)
            update_job_status(job_id, JobStatus.PROCESSING, 55, "Generating 3D mesh...")
            if is_job_cancelled(job_id):
                remove_from_queue(job_id)
                update_job_status(job_id, JobStatus.CANCELLED, 50, "Cancelled", error="Job cancelled by user")
                return
            mesh = None
            def run_trellis2():
                nonlocal mesh
                if TORCH_AVAILABLE:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                # Skip pipeline preprocess (rembg_model may be None); we already did rembg + composite
                with torch.inference_mode():
                    mesh = TRELLIS2_PIPELINE.run(image_rgb_for_trellis, preprocess_image=False)[0]
                mesh.simplify(MESH_SIMPLIFY_FACES)
            try:
                run_trellis2()
                if TORCH_AVAILABLE:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
            except InterruptedError:
                remove_from_queue(job_id)
                update_job_status(job_id, JobStatus.CANCELLED, 50, "Cancelled", error="Job cancelled by user")
                return
            except RuntimeError as e:
                err_msg = str(e).lower()
                if "out of memory" in err_msg or "cuda" in err_msg and "memory" in err_msg:
                    if TORCH_AVAILABLE:
                        try:
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    remove_from_queue(job_id)
                    update_job_status(
                        job_id, JobStatus.FAILED, 50, "GPU out of memory during 3D mesh generation.",
                        error="GPU out of memory. Set TRELLIS_IMAGE_SIZE=512 or 1024 (e.g. export TRELLIS_IMAGE_SIZE=1024) and retry."
                    )
                    return
                raise
            if mesh is None:
                raise RuntimeError("No mesh produced")
            update_job_status(job_id, JobStatus.PROCESSING, 85, "Exporting GLB...")
            out_path = os.path.join(out_dir, "mesh.glb")
            if TORCH_AVAILABLE:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            try:
                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                    attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
decimation_target=MESH_SIMPLIFY_TARGET, texture_size=GLB_TEXTURE_SIZE, remesh=True, remesh_band=1, remesh_project=0, verbose=False,
            )
                glb.export(out_path, extension_webp=False)
            except RuntimeError as e:
                if TORCH_AVAILABLE:
                    try:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                err_msg = str(e).lower()
                if "out of memory" in err_msg or "cuda" in err_msg or "cumesh" in err_msg:
                    remove_from_queue(job_id)
                    update_job_status(
                        job_id, JobStatus.FAILED, 85, "GPU out of memory during GLB export (CuMesh).",
                        error="Reduce MESH_SIMPLIFY_TARGET and MESH_SIMPLIFY_FACES (e.g. 100000) or TRELLIS_IMAGE_SIZE=384."
                    )
                    return
                raise
            if TORCH_AVAILABLE:
                import gc
                del mesh
                gc.collect()
                torch.cuda.empty_cache()
        if is_job_cancelled(job_id):
            update_job_status(job_id, JobStatus.CANCELLED, 0, "Cancelled", error="Job cancelled by user")
            return
        update_job_status(job_id, JobStatus.PROCESSING, 90, "Saving 3D model...")
        update_job_status(job_id, JobStatus.PROCESSING, 95, "Uploading to S3...")
        mesh_s3_url = upload_and_presign(out_path, f"image/{job_id}/mesh.glb")
        processed_s3_url = upload_and_presign(processed_image_path, f"image/{job_id}/processed_image.png")
        remove_from_queue(job_id)
        elapsed = round(time.time() - start, 2)
        result = {"job_id": job_id, "mode": "image-to-3d", "elapsed_seconds": elapsed}
        result["mesh_url"] = mesh_s3_url if mesh_s3_url else out_path
        result["processed_image_url"] = processed_s3_url if processed_s3_url else processed_image_path
        update_job_status(job_id, JobStatus.COMPLETED, 100, "Completed", result=result)
    except Exception as e:
        remove_from_queue(job_id)
        import traceback
        traceback.print_exc()
        update_job_status(job_id, JobStatus.FAILED, 0, str(e), error=str(e))
        if TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

# -------------------------
# Background: text-to-3d
# -------------------------
def process_text_to_3d(job_id: str, prompt: str):
    start = time.time()
    add_to_queue(job_id)
    try:
        if is_job_cancelled(job_id):
            remove_from_queue(job_id)
            update_job_status(job_id, JobStatus.CANCELLED, 0, "Cancelled", error="Job cancelled by user")
            return
        update_job_status(job_id, JobStatus.PROCESSING, 10, "Generating image from text...")
        try:
            image = text_to_image(prompt.strip(), job_id=job_id)
        except InterruptedError:
            remove_from_queue(job_id)
            update_job_status(job_id, JobStatus.CANCELLED, 0, "Cancelled", error="Job cancelled by user")
            return
        except ValueError as e:
            remove_from_queue(job_id)
            update_job_status(job_id, JobStatus.FAILED, 0, str(e), error=str(e))
            return
        out_dir = os.path.join(OUTPUT_BASE, "text", job_id)
        os.makedirs(out_dir, exist_ok=True)
        generated_image_path = os.path.join(out_dir, "generated_image.png")
        image.save(generated_image_path, "PNG", optimize=True)
        update_job_status(job_id, JobStatus.PROCESSING, 30, "Removing background...")
        image = remove_background(image)
        if is_job_cancelled(job_id):
            update_job_status(job_id, JobStatus.CANCELLED, 0, "Cancelled", error="Job cancelled by user")
            return
        update_job_status(job_id, JobStatus.PROCESSING, 40, "Optimizing image...")
        image = optimize_image_for_3d(image)
        # RGB for pipeline (same as api_trellis .ai) so 3D mesh gets correct colors; composite on white
        if image.mode == "RGBA":
            bg_white = Image.new("RGB", image.size, (255, 255, 255))
            bg_white.paste(image, mask=image.split()[-1])
            image_rgb_for_trellis = bg_white.convert("RGB")
        else:
            image_rgb_for_trellis = image.convert("RGB")
        if max(image_rgb_for_trellis.size) > TRELLIS_IMAGE_SIZE:
            ratio = TRELLIS_IMAGE_SIZE / max(image_rgb_for_trellis.size)
            new_sz = (max(1, int(image_rgb_for_trellis.size[0] * ratio)), max(1, int(image_rgb_for_trellis.size[1] * ratio)))
            image_rgb_for_trellis = image_rgb_for_trellis.resize(new_sz, Image.Resampling.LANCZOS)
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (240, 240, 240))
            bg.paste(image, mask=image.split()[-1])
            image = bg.convert("RGB")
        processed_image_path = os.path.join(out_dir, "processed_image.png")
        image.save(processed_image_path, "PNG", optimize=True)
        queue_pos, est_wait, jobs_ahead = get_queue_position(job_id)
        if jobs_ahead > 0:
            update_job_status(job_id, JobStatus.PROCESSING, 50, f"Waiting in queue ({jobs_ahead} ahead)...")
        else:
            update_job_status(job_id, JobStatus.PROCESSING, 50, "Generating 3D mesh (TRELLIS.2)...")
        if is_job_cancelled(job_id):
            remove_from_queue(job_id)
            update_job_status(job_id, JobStatus.CANCELLED, 50, "Cancelled", error="Job cancelled by user")
            return
        if TORCH_AVAILABLE:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        with THREE_D_LOCK:
            start_processing_3d(job_id)
            update_job_status(job_id, JobStatus.PROCESSING, 55, "Generating 3D mesh...")
            if is_job_cancelled(job_id):
                remove_from_queue(job_id)
                update_job_status(job_id, JobStatus.CANCELLED, 50, "Cancelled", error="Job cancelled by user")
                return
            mesh = None
            def run_trellis2():
                nonlocal mesh
                if TORCH_AVAILABLE:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                # Skip pipeline preprocess (rembg_model may be None); we already did rembg + composite
                with torch.inference_mode():
                    mesh = TRELLIS2_PIPELINE.run(image_rgb_for_trellis, preprocess_image=False)[0]
                mesh.simplify(MESH_SIMPLIFY_FACES)
            try:
                run_trellis2()
                if TORCH_AVAILABLE:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
            except InterruptedError:
                remove_from_queue(job_id)
                update_job_status(job_id, JobStatus.CANCELLED, 60, "Cancelled", error="Job cancelled by user")
                return
            except RuntimeError as e:
                err_msg = str(e).lower()
                if "out of memory" in err_msg or "cuda" in err_msg and "memory" in err_msg:
                    if TORCH_AVAILABLE:
                        try:
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    remove_from_queue(job_id)
                    update_job_status(
                        job_id, JobStatus.FAILED, 50, "GPU out of memory during 3D mesh generation.",
                        error="GPU out of memory. Set TRELLIS_IMAGE_SIZE=512 or 1024 (e.g. export TRELLIS_IMAGE_SIZE=1024) and retry."
                    )
                    return
                raise
            if mesh is None:
                raise RuntimeError("No mesh produced")
            update_job_status(job_id, JobStatus.PROCESSING, 85, "Exporting GLB...")
            out_path = os.path.join(out_dir, "mesh.glb")
            if TORCH_AVAILABLE:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            try:
                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                    attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
decimation_target=MESH_SIMPLIFY_TARGET, texture_size=GLB_TEXTURE_SIZE, remesh=True, remesh_band=1, remesh_project=0, verbose=False,
            )
                glb.export(out_path, extension_webp=False)
            except RuntimeError as e:
                if TORCH_AVAILABLE:
                    try:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                err_msg = str(e).lower()
                if "out of memory" in err_msg or "cuda" in err_msg or "cumesh" in err_msg:
                    remove_from_queue(job_id)
                    update_job_status(
                        job_id, JobStatus.FAILED, 85, "GPU out of memory during GLB export (CuMesh).",
                        error="Reduce MESH_SIMPLIFY_TARGET and MESH_SIMPLIFY_FACES (e.g. 100000) or TRELLIS_IMAGE_SIZE=384."
                    )
                    return
                raise
            if TORCH_AVAILABLE:
                import gc
                del mesh
                gc.collect()
                torch.cuda.empty_cache()
        if is_job_cancelled(job_id):
            update_job_status(job_id, JobStatus.CANCELLED, 0, "Cancelled", error="Job cancelled by user")
            return
        update_job_status(job_id, JobStatus.PROCESSING, 95, "Uploading to S3...")
        mesh_s3_url = upload_and_presign(out_path, f"text/{job_id}/mesh.glb")
        generated_s3_url = upload_and_presign(generated_image_path, f"text/{job_id}/generated_image.png")
        processed_s3_url = upload_and_presign(processed_image_path, f"text/{job_id}/processed_image.png")
        remove_from_queue(job_id)
        elapsed = round(time.time() - start, 2)
        result = {"job_id": job_id, "mode": "text-to-3d", "prompt": prompt.strip(), "elapsed_seconds": elapsed}
        result["mesh_url"] = mesh_s3_url if mesh_s3_url else out_path
        result["generated_image_url"] = generated_s3_url if generated_s3_url else generated_image_path
        result["processed_image_url"] = processed_s3_url if processed_s3_url else processed_image_path
        update_job_status(job_id, JobStatus.COMPLETED, 100, "Completed", result=result)
    except Exception as e:
        remove_from_queue(job_id)
        import traceback
        traceback.print_exc()
        update_job_status(job_id, JobStatus.FAILED, 0, str(e), error=str(e))
        if TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

# -------------------------
# Endpoints: jobs/active, image-to-3d, text-to-3d, text-to-image, edit-image, status, stream, cancel, queue, user jobs
# -------------------------
@app.get("/jobs/active")
def get_active_jobs():
    with JOBS_LOCK:
        active = sum(1 for j in jobs.values() if j.get("status") in (JobStatus.PENDING.value, JobStatus.PROCESSING.value))
        qi = get_queue_info()
        preview_active = qi["preview_queue_length"] + (1 if qi["currently_generating_preview"] else 0)
    return {
        "active_jobs": active,
        "total_jobs": len(jobs),
        "safe_to_shutdown": active == 0 and preview_active == 0,
        "preview_queue_length": qi["preview_queue_length"],
        "currently_generating_preview": qi["currently_generating_preview"],
    }

@app.post("/upload-image")
async def upload_image(image: UploadFile = File(..., alias="image")):
    """Upload a single image to S3; return presigned URL. Same contract as gateway for fallback."""
    if not S3_ENABLED or s3 is None:
        raise HTTPException(status_code=503, detail="S3 upload not available")
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type; image required")
    try:
        content = await image.read()
        img = Image.open(BytesIO(content))
        img = apply_exif_orientation(img)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        content = buf.getvalue()
        temp_dir = os.path.join(OUTPUT_BASE, "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.png")
        with open(temp_path, "wb") as f:
            f.write(content)
        s3_key = f"uploads/{os.path.basename(temp_path)}"
        url = upload_and_presign(temp_path, s3_key)
        try:
            os.remove(temp_path)
        except Exception:
            pass
        if not url:
            raise HTTPException(status_code=500, detail="S3 upload failed")
        return {"success": True, "url": url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/image-to-3d")
async def image_to_3d(
    request: Request,
    image_file: UploadFile = File(None),
    image: UploadFile = File(None),
    image_url: str = Form(None),
    image_path: str = Form(None),
    user_id: str = Form(None),
):
    if request.headers.get("content-type", "").strip().startswith("application/json"):
        try:
            body = await request.json()
        except Exception:
            body = {}
        image_url = image_url or body.get("image_url")
        image_path = image_path or body.get("image_path")
        user_id = user_id or body.get("user_id")
    uploaded = image_file or image
    if not uploaded and not image_url and not (image_path and os.path.isfile(image_path)):
        raise HTTPException(status_code=400, detail="Provide image_file, image, or image_url")
    if TRELLIS2_PIPELINE is None:
        raise HTTPException(status_code=503, detail="TRELLIS.2 pipeline not loaded")
    with QUEUE_LOCK:
        if len(job_queue) + (1 if currently_processing_3d else 0) >= MAX_QUEUE_LENGTH:
            raise HTTPException(status_code=503, detail=f"3D queue full (max {MAX_QUEUE_LENGTH})")
    final_image_url = None
    if uploaded:
        if not uploaded.content_type or not uploaded.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type")
        try:
            content = await uploaded.read()
            img = Image.open(BytesIO(content))
            img = apply_exif_orientation(img)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG", optimize=True)
            content = buf.getvalue()
            temp_dir = os.path.join(OUTPUT_BASE, "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.png")
            with open(temp_path, "wb") as f:
                f.write(content)
            if S3_ENABLED and s3:
                s3_key = f"uploads/{os.path.basename(temp_path)}"
                s3.upload_file(temp_path, S3_BUCKET, s3_key)
                final_image_url = s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": s3_key}, ExpiresIn=S3_PRESIGNED_URL_EXPIRY)
                os.remove(temp_path)
            else:
                raise HTTPException(status_code=400, detail="File upload requires S3. Use image_url or configure S3.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    elif image_path and os.path.isfile(image_path):
        # Local path (e.g. from gateway job payload)
        if S3_ENABLED and s3:
            s3_key = f"uploads/{uuid.uuid4().hex}.png"
            final_image_url = upload_and_presign(image_path, s3_key)
            if not final_image_url:
                raise HTTPException(status_code=500, detail="S3 upload failed")
        else:
            raise HTTPException(status_code=400, detail="image_path requires S3")
    else:
        if not image_url or not image_url.strip():
            raise HTTPException(status_code=400, detail="Provide image_url")
        if not is_valid_url(image_url.strip()):
            raise HTTPException(status_code=400, detail="Invalid URL")
        final_image_url = image_url.strip()
    job_id = str(uuid.uuid4())
    update_job_status(job_id, JobStatus.PENDING, 0, "Job created...", user_id=user_id)
    asyncio.get_event_loop().run_in_executor(EXECUTOR, _run_with_gpu_cleanup, process_image_to_3d, job_id, final_image_url)
    with QUEUE_LOCK:
        jobs_ahead = len(job_queue) + (1 if currently_processing_3d else 0)
    qwait = jobs_ahead * ESTIMATED_3D_TIME
    return {
        "job_id": job_id,
        "user_id": user_id,
        "status": "pending",
        "message": f"Job queued for 3D generation. {jobs_ahead} job(s) ahead.",
        "status_url": f"/status/{job_id}",
        "stream_url": f"/stream/{job_id}",
        "queue": {
            "position": jobs_ahead + 1,
            "jobs_ahead": jobs_ahead,
            "estimated_wait_seconds": qwait,
            "estimated_total_seconds": qwait + ESTIMATED_3D_TIME,
        },
    }

@app.post("/text-to-3d")
async def text_to_3d(request: Request, prompt: str = Form(None), user_id: str = Form(None)):
    if request.headers.get("content-type", "").strip().startswith("application/json"):
        try:
            body = await request.json()
        except Exception:
            body = {}
        prompt = prompt or body.get("prompt")
        user_id = user_id or body.get("user_id")
    if not prompt or not str(prompt).strip():
        raise HTTPException(status_code=400, detail="Prompt is required")
    prompt = str(prompt).strip()
    if TRELLIS2_PIPELINE is None:
        raise HTTPException(status_code=503, detail="TRELLIS.2 pipeline not loaded")
    with QUEUE_LOCK:
        if len(job_queue) + (1 if currently_processing_3d else 0) >= MAX_QUEUE_LENGTH:
            raise HTTPException(status_code=503, detail=f"3D queue full (max {MAX_QUEUE_LENGTH})")
    job_id = str(uuid.uuid4())
    update_job_status(job_id, JobStatus.PENDING, 0, "Job created...", user_id=user_id)
    asyncio.get_event_loop().run_in_executor(EXECUTOR, _run_with_gpu_cleanup, process_text_to_3d, job_id, prompt)
    with QUEUE_LOCK:
        jobs_ahead = len(job_queue) + (1 if currently_processing_3d else 0)
    qwait = jobs_ahead * ESTIMATED_3D_TIME
    return {
        "job_id": job_id,
        "user_id": user_id,
        "status": "pending",
        "message": f"Job queued for 3D generation. {jobs_ahead} job(s) ahead.",
        "status_url": f"/status/{job_id}",
        "stream_url": f"/stream/{job_id}",
        "queue": {
            "position": jobs_ahead + 1,
            "jobs_ahead": jobs_ahead,
            "estimated_wait_seconds": qwait,
            "estimated_total_seconds": qwait + ESTIMATED_3D_TIME,
        },
    }

def process_preview_generation(preview_id: str, prompt: str):
    try:
        start_processing_preview(preview_id)
        if TORCH_AVAILABLE:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        with PREVIEW_LOCK:
            image = text_to_image(prompt.strip(), job_id=None)
        out_dir = os.path.join(OUTPUT_BASE, "preview", preview_id)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "preview_image.png")
        image.save(path, "PNG", optimize=True)
        url = upload_and_presign(path, f"preview/{preview_id}/preview_image.png")
        remove_from_preview_queue(preview_id)
        if TORCH_AVAILABLE:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        return {"success": True, "preview_id": preview_id, "image_url": url or path, "prompt": prompt.strip()}
    except Exception as e:
        remove_from_preview_queue(preview_id)
        if TORCH_AVAILABLE:
            try:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        raise

def process_edit_generation(edit_id: str, input_image_path: str, prompt: str, strength: float = 0.6):
    try:
        with open(input_image_path, "rb") as f:
            img = Image.open(f).copy()
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        with PREVIEW_LOCK:
            image = edit_image_with_prompt(img, prompt, strength=strength)
        out_dir = os.path.join(OUTPUT_BASE, "edit", edit_id)
        os.makedirs(out_dir, exist_ok=True)
        edited_path = os.path.join(out_dir, "edited.png")
        image.save(edited_path, "PNG", optimize=True)
        url = upload_and_presign(edited_path, f"edit/{edit_id}/edited.png")
        if TORCH_AVAILABLE:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        return {"edit_id": edit_id, "image_url": url or edited_path, "prompt": prompt, "strength": strength}
    except Exception:
        if TORCH_AVAILABLE:
            try:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        raise

@app.post("/text-to-image")
async def text_to_image_preview(request: Request, prompt: str = Form(None), user_id: str = Form(None)):
    if request.headers.get("content-type", "").strip().startswith("application/json"):
        try:
            body = await request.json()
        except Exception:
            body = {}
        prompt = prompt or body.get("prompt")
        user_id = user_id or body.get("user_id")
    if not prompt or not str(prompt).strip():
        raise HTTPException(status_code=400, detail="Prompt is required")
    prompt = str(prompt).strip()
    with QUEUE_LOCK:
        if len(preview_queue) + (1 if currently_generating_preview else 0) >= MAX_PREVIEW_QUEUE_LENGTH:
            raise HTTPException(status_code=503, detail=f"Preview queue full (max {MAX_PREVIEW_QUEUE_LENGTH})")
    preview_id = str(uuid.uuid4())
    add_to_preview_queue(preview_id)
    # Create a job record so GET /status/{preview_id} returns 200 after completion (avoids 404 when client polls with preview_id)
    update_job_status(preview_id, JobStatus.PENDING, 0, "Generating preview...", user_id=user_id)
    queue_info = get_queue_info()
    preview_pos, est_wait, ahead = get_preview_queue_position(preview_id)
    try:
        result = await asyncio.get_event_loop().run_in_executor(EXECUTOR, _run_with_gpu_cleanup, process_preview_generation, preview_id, prompt)
        # Store completed preview in jobs so GET /status/{preview_id} returns 200 (backend/frontend may poll with this id)
        img_url = result["image_url"]
        update_job_status(
            preview_id,
            JobStatus.COMPLETED,
            100,
            "Completed",
            result={
                "preview_id": preview_id,
                "image_url": img_url,
                "prompt": result["prompt"],
                "processed_image_url": img_url,
                "generated_image_url": img_url,
            },
            user_id=user_id,
        )
        return {
            "success": True,
            "preview_id": result["preview_id"],
            "image_url": result["image_url"],
            "prompt": result["prompt"],
            "status": "completed",
            "message": "Preview completed",
            "status_url": f"/status/{preview_id}",
            "queue": {
                "position": preview_pos,
                "previews_ahead": ahead,
                "estimated_wait_seconds": est_wait,
                "queue_length": queue_info["preview_queue_length"],
                "currently_generating": queue_info["currently_generating_preview"],
            },
        }
    except ValueError as e:
        update_job_status(preview_id, JobStatus.FAILED, 0, str(e), error=str(e), user_id=user_id)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        update_job_status(preview_id, JobStatus.FAILED, 0, "Preview failed", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/edit-image")
async def edit_image_endpoint(
    request: Request,
    prompt: str = Form(None),
    strength: float = Form(0.6),
    image: UploadFile = File(None),
    image_url: str = Form(None),
):
    if request.headers.get("content-type", "").strip().startswith("application/json"):
        try:
            body = await request.json()
        except Exception:
            body = {}
        prompt = prompt or body.get("prompt")
        strength = body.get("strength", strength)
        image_url = image_url or body.get("image_url")
    if not prompt or not str(prompt).strip():
        raise HTTPException(status_code=400, detail="Prompt is required")
    prompt = str(prompt).strip()
    strength = max(0.01, min(1.0, float(strength)))
    if z_image_img2img_pipeline is None:
        raise HTTPException(status_code=503, detail="Image edit requires Z-Image-Turbo")
    edit_id = str(uuid.uuid4())
    out_dir = os.path.join(OUTPUT_BASE, "edit", edit_id)
    os.makedirs(out_dir, exist_ok=True)
    input_path = os.path.join(out_dir, "input.png")
    temp_path_from_url = None
    try:
        if image and image.filename:
            content = await image.read()
            with open(input_path, "wb") as f:
                f.write(content)
        elif image_url and image_url.strip():
            if not is_valid_url(image_url.strip()):
                raise HTTPException(status_code=400, detail="Invalid image_url")
            temp_path_from_url = download_image_url_to_temp_path(image_url.strip())
            shutil.copy(temp_path_from_url, input_path)
        else:
            raise HTTPException(status_code=400, detail="Provide image file or image_url")
        result = await asyncio.get_event_loop().run_in_executor(EXECUTOR, _run_with_gpu_cleanup, process_edit_generation, edit_id, input_path, prompt, strength)
        return {"success": True, "edit_id": result["edit_id"], "image_url": result["image_url"], "prompt": result["prompt"], "strength": result["strength"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path_from_url and os.path.isfile(temp_path_from_url):
            try:
                os.remove(temp_path_from_url)
            except Exception:
                pass

@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    with JOBS_LOCK:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = jobs[job_id]
        if job["status"] in (JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value):
            raise HTTPException(status_code=400, detail=f"Cannot cancel job with status {job['status']}")
    mark_job_cancelled(job_id)
    update_job_status(job_id, JobStatus.CANCELLED, job.get("progress", 0), "Cancellation requested", error="Job cancelled by user")
    return {"job_id": job_id, "status": "cancelled", "message": "Cancellation requested"}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    with JOBS_LOCK:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = jobs[job_id].copy()
    qpos, qwait, qahead = get_queue_position(job_id)
    qi = get_queue_info()
    estimated_total = (qwait + ESTIMATED_3D_TIME) if job["status"] in (JobStatus.PENDING.value, JobStatus.PROCESSING.value) else 0
    resp = {
        "job_id": job_id,
        "user_id": job.get("user_id"),
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "queue": {
            "position": qpos,
            "jobs_ahead": qahead,
            "estimated_wait_seconds": qwait,
            "estimated_total_seconds": estimated_total,
            "queue_length": qi["queue_length"],
            "currently_processing": qi["currently_processing"],
        },
    }
    if job["status"] == JobStatus.COMPLETED.value and job.get("result"):
        resp["result"] = job["result"]
    if job["status"] in (JobStatus.FAILED.value, JobStatus.CANCELLED.value) and job.get("error"):
        resp["error"] = job["error"]
    return resp

@app.get("/queue/info")
async def get_queue_status():
    qi = get_queue_info()
    return {
        "queue_length": qi["queue_length"],
        "currently_processing": qi["currently_processing"],
        "processing_job_id": qi["processing_job_id"],
        "waiting_jobs": qi["waiting_jobs"],
        "jobs_ahead_for_new": qi["jobs_ahead_for_new"],
        "estimated_wait_for_new_job_seconds": qi["estimated_wait_for_new_job"],
        "estimated_total_seconds": qi["estimated_total_seconds"],
        "estimated_time_per_job_seconds": qi["estimated_time_per_job_seconds"],
        "preview_queue_length": qi["preview_queue_length"],
        "currently_generating_preview": qi["currently_generating_preview"],
        "preview_waiting": qi["preview_waiting"],
        "estimated_wait_for_preview_seconds": qi["estimated_wait_for_preview"],
        "estimated_preview_time_seconds": qi["estimated_preview_time_seconds"],
    }

@app.get("/jobs/user/{user_id}")
async def get_user_jobs(user_id: str):
    with JOBS_LOCK:
        if user_id not in user_jobs:
            return {"jobs": [], "count": 0}
        list_ids = list(user_jobs[user_id])
        user_job_list = []
        for jid in list_ids:
            if jid not in jobs:
                continue
            j = jobs[jid].copy()
            user_job_list.append({k: j.get(k) for k in ["job_id", "user_id", "status", "progress", "message", "created_at", "updated_at", "result", "error"] if j.get(k) is not None})
        user_job_list.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return {"jobs": user_job_list, "count": len(user_job_list)}

@app.get("/stream/{job_id}")
async def stream_progress(job_id: str):
    with JOBS_LOCK:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
    async def gen():
        last_p, last_s = -1, None
        while True:
            with JOBS_LOCK:
                if job_id not in jobs:
                    yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                    return
                job = jobs[job_id].copy()
            s, p = job["status"], job["progress"]
            if s != last_s or p != last_p:
                data = {"job_id": job_id, "status": s, "progress": p, "message": job["message"]}
                if s == JobStatus.COMPLETED.value and job.get("result"):
                    data["result"] = job["result"]
                    yield f"data: {json.dumps(data)}\n\n"
                    return
                if s in (JobStatus.FAILED.value, JobStatus.CANCELLED.value) and job.get("error"):
                    data["error"] = job["error"]
                    yield f"data: {json.dumps(data)}\n\n"
                    return
                yield f"data: {json.dumps(data)}\n\n"
                last_p, last_s = p, s
            if s in (JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value):
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

@app.on_event("startup")
async def startup_event():
    load_jobs_from_disk()
    print(f"📊 [SERVER] Quality preset: {DEFAULT_QUALITY}, ESTIMATED_3D_TIME={ESTIMATED_3D_TIME}s, TRELLIS_IMAGE_SIZE={TRELLIS_IMAGE_SIZE}, MESH_SIMPLIFY_TARGET={MESH_SIMPLIFY_TARGET}, GLB_TEXTURE_SIZE={GLB_TEXTURE_SIZE} (set lower if OOM/killed)")
