# hydrilla.co - GPU Docker deployment (TRELLIS.2-4B)

This API runs on `0.0.0.0:8000` and is packaged for NVIDIA GPU hosts using Docker.

The Docker image is portable across EC2 instances and laptops if they have:

- NVIDIA GPU + compatible driver
- Docker Engine
- NVIDIA Container Toolkit (`--gpus all` support)

## Quick start (same machine)

Run commands from the directory that contains **`docker-compose.yml`** (and `Dockerfile`). In this repo that folder is named **`hydrilla.co`**. If you cloned or renamed it to something like **`~/hydrilla`**, stay in that folder — do **not** `cd hydrilla.co` unless that subfolder actually exists.

```bash
docker compose build
docker compose up -d
docker compose logs -f hydrilla-api
```

The image build can take **a long time** (often well over an hour on a typical CPU): `trellis-setup-docker.sh` mirrors TRELLIS.2 `setup.sh` with `--basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --flexgemm --o-voxel` (no conda; `TRELLIS_SETUP_ALLOW_CPU=1` so `nvidia-smi` is not required at build time). FlashAttention and the CUDA extensions compile during the build.

By default this build pulls TRELLIS from your fork:

- `https://github.com/dhanushyangal/TRELLIS.2`

If you want another repo/branch, set:

```bash
export TRELLIS_REPO=https://github.com/<user>/TRELLIS.2.git
export TRELLIS_REF=main
docker compose build --no-cache
```

Open:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## Required runtime notes

- This setup is GPU-only.
- First boot may take longer because model files are downloaded to Hugging Face cache.
- Cache and outputs are persisted in Docker volumes (`hf_cache`, `hydrilla_outputs`) so restart/migration is faster.

## Environment variables

Configure these by exporting env vars before `docker compose up` (or put them in a `.env` file next to `docker-compose.yml`):

- `HF_TOKEN` (recommended for faster/private/gated model pulls)
- `ENABLE_S3` (`1` or `0`)
- `S3_BUCKET`, `S3_REGION`, `S3_PRESIGNED_URL_EXPIRY`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (if S3 is enabled and you are not using IAM role)
- `TRELLIS_IMAGE_SIZE`, `GLB_TEXTURE_SIZE`
- `MESH_SIMPLIFY_TARGET`, `MESH_SIMPLIFY_FACES`
- `MAX_WORKERS`
- `TRELLIS_REPO`, `TRELLIS_REF` (for Docker build source)

## Move image to another instance or laptop

### Option A: Use Docker registry (recommended)

```bash
docker tag hydrilla/hydrilla-api:gpu <registry>/<namespace>/hydrilla-api:gpu
docker push <registry>/<namespace>/hydrilla-api:gpu
```

On target host:

```bash
docker pull <registry>/<namespace>/hydrilla-api:gpu
docker run -d --name hydrilla-api \
  --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN=<your_hf_token> \
  -v hydrilla_outputs:/app/outputs \
  -v hf_cache:/root/.cache/huggingface \
  --restart unless-stopped \
  <registry>/<namespace>/hydrilla-api:gpu
```

### Option B: Export/import image tar

On source host:

```bash
docker save hydrilla/hydrilla-api:gpu -o hydrilla-api-gpu.tar
```

Copy `hydrilla-api-gpu.tar` to target host, then:

```bash
docker load -i hydrilla-api-gpu.tar
docker run -d --name hydrilla-api \
  --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN=<your_hf_token> \
  -v hydrilla_outputs:/app/outputs \
  -v hf_cache:/root/.cache/huggingface \
  --restart unless-stopped \
  hydrilla/hydrilla-api:gpu
```

## Validation checklist

1. Check startup logs:

```bash
docker compose logs -f hydrilla-api
```

Expected line includes `TRELLIS.2-4B loaded on GPU`.

1. Health check:

```bash
curl http://localhost:8000/health
```

1. Submit image-to-3d job:

```bash
curl -X POST "http://localhost:8000/image-to-3d" \
  -F "image_url=https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=1024"
```

1. Poll status:

```bash
curl "http://localhost:8000/status/<job_id>"
```

1. Restart container and verify persisted data:

```bash
docker compose restart hydrilla-api
docker volume ls
```

`hf_cache` and `hydrilla_outputs` should still exist and previous outputs should remain available.

## Endpoints

|Endpoint|Description|
|---|---|
|`GET /health`|Health check|
|`GET /jobs/active`|Active job count, safe_to_shutdown|
|`POST /image-to-3d`|Image -> 3D (file or `image_url`)|
|`POST /text-to-3d`|Text -> 3D (uses t2i then Trellis2)|
|`POST /text-to-image`|Text -> preview image|
|`POST /edit-image`|Edit image with prompt (Z-Image img2img)|
|`GET /status/{job_id}`|Job status + result|
|`GET /stream/{job_id}`|SSE progress|
|`POST /cancel/{job_id}`|Cancel job|
|`GET /queue/info`|Queue lengths and wait estimates|
|`GET /jobs/user/{user_id}`|List jobs by user|
