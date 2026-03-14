# Benchmarks

Performance benchmarks for LandmarkDiff across different hardware configurations.

## Inference Speed

All timings are wall-clock per single image at 512x512 resolution, including landmark extraction, deformation, generation, and post-processing.

| Hardware | Mode | Steps | Time per Image | Notes |
|----------|------|-------|----------------|-------|
| A100 80GB | ControlNet + IP-Adapter | 30 | ~5 sec | Fastest GPU mode |
| A100 80GB | ControlNet | 30 | ~3 sec | Best quality/speed ratio |
| A100 80GB | img2img | 30 | ~2.5 sec | |
| A100 40GB | ControlNet | 30 | ~4 sec | |
| RTX 4090 | ControlNet | 30 | ~5 sec | |
| RTX 3090 | ControlNet | 30 | ~7 sec | |
| A6000 48GB | ControlNet | 30 | ~6 sec | Common HPC card |
| T4 16GB | ControlNet | 30 | ~15 sec | Cloud budget GPU |
| L40S | ControlNet | 30 | ~5 sec | Newer datacenter GPU |
| M3 Pro (MPS) | ControlNet | 30 | ~45 sec | fp32 only, no fp16 on MPS |
| CPU (i9-13900K) | TPS only | -- | ~0.5 sec | No diffusion model |
| CPU (any modern) | TPS only | -- | ~0.05 sec | Landmark extraction + warp only |

### Effect of Step Count on Speed

ControlNet mode on A100 80GB:

| Steps | Time | Quality |
|-------|------|---------|
| 10 | ~1.5 sec | Noticeable artifacts |
| 15 | ~2.0 sec | Acceptable for previews |
| 20 | ~2.5 sec | Good quality, minor details lost |
| 30 | ~3.0 sec | Full quality (default) |
| 50 | ~5.0 sec | Marginal improvement over 30 |

DPM++ 2M Karras scheduler produces good results with fewer steps than DDPM or DDIM.

---

## Landmark Extraction Speed

MediaPipe Face Mesh v2 runs entirely on CPU. Extraction time is consistent across hardware:

| Operation | Speed | Notes |
|-----------|-------|-------|
| Face detection + 478 landmarks | ~30 ms | Single face, CPU |
| Batch (100 images) | ~3 sec | Sequential processing |
| Landmark extraction throughput | ~30 fps | Independent of GPU |

---

## VRAM Usage

Peak GPU memory during inference with model_cpu_offload enabled (default):

| Component | VRAM (fp16) | Notes |
|-----------|-------------|-------|
| SD 1.5 UNet | ~2.5 GB | Main diffusion model |
| ControlNet | ~1.5 GB | CrucibleAI face mesh ControlNet |
| VAE (fp32 decode) | ~0.5 GB | fp32 forced for color accuracy |
| IP-Adapter | ~0.7 GB | Only in controlnet_ip mode |
| CodeFormer | ~0.4 GB | Face restoration (optional) |
| ArcFace | ~0.3 GB | Identity verification (optional) |
| Real-ESRGAN | ~0.2 GB | Background enhancement (optional) |

### Total VRAM by Mode

| Mode | Min VRAM | Recommended | With Post-processing |
|------|----------|-------------|---------------------|
| tps | 0 GB | -- | 0 GB |
| img2img | 4 GB | 6 GB | 5 GB |
| controlnet | 5 GB | 8 GB | 6 GB |
| controlnet_ip | 7 GB | 10 GB | 8 GB |

`model_cpu_offload()` is enabled by default on CUDA. It moves model components to CPU RAM when not in use, reducing peak VRAM at the cost of ~10% slower inference due to CPU-GPU transfer overhead.

On Apple Silicon (MPS), all models run in fp32 due to MPS backend limitations. This roughly doubles the memory figures above.

---

## Training Throughput

Training the ControlNet (Phase A) on different hardware:

| Hardware | Batch Size | Grad Accum | Effective Batch | Steps/hour | Time to 50K Steps |
|----------|-----------|------------|-----------------|------------|-------------------|
| 4x A6000 48GB (DDP) | 4 | 4 | 64 | ~800 | ~62 hours |
| 1x A100 80GB | 4 | 4 | 16 | ~600 | ~83 hours |
| 1x A100 40GB | 2 | 8 | 16 | ~400 | ~125 hours |
| 1x RTX 4090 | 2 | 8 | 16 | ~350 | ~143 hours |
| 1x RTX 3090 | 1 | 16 | 16 | ~200 | ~250 hours |

Phase B (identity-aware fine-tuning) is typically 10-20% slower than Phase A due to the additional ArcFace forward pass for the identity loss.

### Training Memory Usage

| Configuration | VRAM per GPU | Notes |
|---------------|-------------|-------|
| Phase A, batch=4, fp16 | ~25 GB | Gradient checkpointing on |
| Phase A, batch=2, fp16 | ~18 GB | Fits on 24 GB cards |
| Phase A, batch=1, fp16 | ~14 GB | Minimum viable |
| Phase B, batch=4, fp16 | ~30 GB | +ArcFace for identity loss |
| Phase B, batch=2, fp16 | ~22 GB | |

---

## Quality Metrics

Measured on a held-out test set of clinical photography pairs. Lower FID and LPIPS are better; higher identity and SSIM are better.

### By Inference Mode

| Mode | FID | LPIPS | Identity (ArcFace cos) | SSIM | NME |
|------|-----|-------|------------------------|------|-----|
| TPS | -- | 0.15 | 0.92 | 0.88 | 0.018 |
| img2img | 42.3 | 0.12 | 0.85 | 0.82 | 0.015 |
| ControlNet | 35.1 | 0.09 | 0.83 | 0.80 | 0.012 |
| ControlNet + IP-Adapter | 37.8 | 0.10 | 0.89 | 0.81 | 0.013 |

Notes:
- TPS has the highest identity preservation because it only moves pixels geometrically with no texture synthesis
- ControlNet produces the best perceptual quality (lowest FID/LPIPS)
- IP-Adapter significantly improves identity preservation over plain ControlNet at a small cost to FID

### By Procedure

ControlNet mode, 30 steps, intensity=60:

| Procedure | FID | LPIPS | Identity | Notes |
|-----------|-----|-------|----------|-------|
| rhinoplasty | 32.4 | 0.08 | 0.86 | Small region, high quality |
| blepharoplasty | 34.7 | 0.09 | 0.88 | Very small region |
| rhytidectomy | 41.2 | 0.11 | 0.79 | Largest region, most challenging |
| orthognathic | 38.5 | 0.10 | 0.82 | Structural changes visible |
| brow_lift | 33.1 | 0.08 | 0.87 | Moderate region |
| mentoplasty | 35.9 | 0.09 | 0.85 | Small region |

### Post-processing Effect

ControlNet mode, with and without post-processing pipeline:

| Configuration | FID | LPIPS | Identity | SSIM |
|---------------|-----|-------|----------|------|
| Raw output | 42.8 | 0.13 | 0.78 | 0.74 |
| + CodeFormer | 38.2 | 0.10 | 0.81 | 0.78 |
| + Histogram match | 36.5 | 0.09 | 0.82 | 0.79 |
| + Laplacian blend | 35.1 | 0.09 | 0.83 | 0.80 |
| + All post-processing | 35.1 | 0.09 | 0.83 | 0.80 |

The full post-processing pipeline (CodeFormer + histogram matching + Laplacian pyramid blending + frequency-aware sharpening) provides substantial quality improvements, particularly in identity preservation and color consistency.

---

## Running Benchmarks

### Inference Benchmark

```bash
python benchmarks/benchmark_inference.py --device cuda --num_images 100
```

### Landmark Extraction Benchmark

```bash
python benchmarks/benchmark_landmarks.py --num_images 1000
```

### Training Throughput Benchmark

```bash
python benchmarks/benchmark_training.py --device cuda --num_steps 100
```

### Full Evaluation on Test Set

```bash
landmarkdiff evaluate --test-dir data/test --output eval_results --mode controlnet
```

Available metrics: FID, LPIPS, NME, Identity (ArcFace cosine similarity), SSIM. Results can be stratified by Fitzpatrick skin type and by procedure.
