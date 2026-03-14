# Installation

## System Requirements

- **Python:** 3.10 or later (3.11 recommended)
- **OS:** Linux (primary), macOS (MPS backend for Apple Silicon), Windows (WSL2 recommended)
- **GPU:** NVIDIA GPU with 6+ GB VRAM for diffusion inference, 40+ GB for training
- **CPU-only:** TPS mode works without any GPU

## Quick Install

### From PyPI

```bash
pip install landmarkdiff
```

### From Source (recommended for development)

```bash
git clone https://github.com/dreamlessx/LandmarkDiff-public.git
cd LandmarkDiff-public
pip install -e .
```

## Install Options

LandmarkDiff uses optional dependency groups so you only install what you need.

### Core (inference only)

```bash
pip install -e .
```

Installs the base package with MediaPipe, PyTorch, diffusers, and transformers. Sufficient for running predictions in all four inference modes (TPS, img2img, ControlNet, ControlNet + IP-Adapter).

Core dependencies:
- `torch>=2.1.0`
- `diffusers>=0.27.0`
- `transformers>=4.38.0`
- `accelerate>=0.27.0`
- `mediapipe>=0.10.9`
- `opencv-python>=4.9.0`
- `numpy>=1.26.0`
- `Pillow>=10.0.0`
- `pyyaml>=6.0`

### Development

```bash
pip install -e ".[dev]"
pre-commit install
```

Includes testing (pytest), linting (ruff), type checking (mypy), and pre-commit hooks.

### Training

```bash
pip install -e ".[train]"
```

Adds training dependencies: wandb for experiment tracking, deepspeed for distributed training, webdataset for large-scale data loading, and insightface/onnxruntime for identity losses.

### Evaluation

```bash
pip install -e ".[eval]"
```

Adds evaluation metric libraries: torch-fidelity (FID), lpips, scikit-image (SSIM), and scipy.

### Gradio Demo

```bash
pip install -e ".[app]"
```

Adds Gradio for the interactive web demo, plus FastAPI and Uvicorn for the REST API server.

### GPU Acceleration

```bash
pip install -e ".[gpu]"
```

Adds xformers and triton for faster attention computation on NVIDIA GPUs.

### Everything

```bash
pip install -e ".[train,eval,app,dev,gpu]"
```

---

## Conda Environment

If you prefer conda for environment management:

```bash
conda create -n landmarkdiff python=3.11
conda activate landmarkdiff

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install LandmarkDiff
pip install -e .
```

---

## PyTorch with CUDA

LandmarkDiff requires PyTorch with CUDA support for diffusion-based inference modes. If you have not installed PyTorch with CUDA yet:

```bash
# Check your CUDA version
nvidia-smi

# Install PyTorch matching your CUDA version
# CUDA 12.1 (most common on recent systems)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For Apple Silicon (M1/M2/M3), PyTorch MPS backend is used automatically:

```bash
pip install torch torchvision
```

---

## Docker

### GPU Docker (recommended for deployment)

```bash
# Build the image
docker build -t landmarkdiff .

# Run the Gradio demo
docker run -p 7860:7860 --gpus all landmarkdiff
# Open http://localhost:7860

# Run with a specific GPU
docker run -p 7860:7860 --gpus '"device=0"' landmarkdiff
```

GPU passthrough requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Docker Compose

```bash
docker compose up landmarkdiff

# Run training (requires GPU)
docker compose --profile training up train
```

### CPU-only Docker

```bash
docker build -f Dockerfile.cpu -t landmarkdiff-cpu .
docker run -p 7860:7860 landmarkdiff-cpu
```

The Docker images use CUDA 12.1 + Python 3.11 and include all dependencies.

---

## Apptainer / Singularity (HPC)

For HPC environments that do not allow Docker:

```bash
apptainer build landmarkdiff.sif containers/landmarkdiff.def
apptainer exec --nv landmarkdiff.sif python scripts/app.py
```

---

## Verify Installation

Run this after installing to confirm everything is working:

```bash
python -c "
import landmarkdiff
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
print('LandmarkDiff installed successfully')
print(f'Version: {landmarkdiff.__version__}')
"
```

For a more thorough check that includes PyTorch device detection:

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    print('MPS backend available (Apple Silicon)')
else:
    print('CPU only (TPS mode will work, diffusion modes will be slow)')

from landmarkdiff.inference import LandmarkDiffPipeline, get_device
print(f'LandmarkDiff device: {get_device()}')
"
```

---

## Common Installation Issues

### MediaPipe fails on headless server

MediaPipe requires OpenGL libraries. On headless Linux servers:

```bash
# Debian / Ubuntu
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# RHEL / CentOS / Rocky
sudo dnf install mesa-libGL glib2
```

### PyTorch CUDA version mismatch

If you see errors about CUDA version incompatibility:

```bash
# Check system CUDA version
nvidia-smi

# Reinstall PyTorch for your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### MediaPipe version compatibility

LandmarkDiff supports both the new Tasks API (MediaPipe >= 0.10.20) and the legacy Solutions API. If you encounter issues with one API, the code automatically falls back to the other:

```bash
pip install mediapipe==0.10.14    # legacy Solutions API
pip install mediapipe>=0.10.20    # new Tasks API (recommended)
```

### Pre-commit hooks fail

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
```

---

## Next Steps

- [Quick Start](Quick-Start) for a guided walkthrough with code examples
- [API Reference](API-Reference) for the full module documentation
- [FAQ](FAQ) for common questions
