# landmarkdiff.inference

Full prediction pipeline combining landmark deformation, ControlNet generation, and compositing.

## Classes

### `LandmarkDiffPipeline`

Main inference pipeline.

**Class Methods:**
- `from_pretrained(checkpoint_dir, device="cuda") -> LandmarkDiffPipeline`

**Methods:**

#### `generate(image, procedure="rhinoplasty", intensity=50.0, ...) -> dict`

Generate a surgical prediction.

**Parameters:**
- `image` (np.ndarray): Input face photo (BGR numpy array, will be resized to 512x512)
- `procedure` (str): Surgical procedure name (default: `"rhinoplasty"`)
- `intensity` (float): Deformation strength, 0 to 100 (default: 50.0)
- `num_inference_steps` (int): Diffusion denoising steps (default: 30)
- `guidance_scale` (float): Classifier-free guidance strength (default: 9.0)
- `controlnet_conditioning_scale` (float): How strongly the wireframe controls generation (default: 0.9)
- `strength` (float): img2img denoising strength (default: 0.5)
- `seed` (int | None): Random seed for reproducibility (default: None)
- `clinical_flags` (ClinicalFlags | None): Clinical edge case flags (default: None)
- `postprocess` (bool): Run post-processing pipeline (default: True)
- `use_gfpgan` (bool): Use GFPGAN instead of CodeFormer for face restoration (default: False)

**Returns:** dict with keys:
- `"output"` - Final composited image
- `"output_raw"` - Raw diffusion output (before compositing)
- `"output_tps"` - TPS-only geometric warp
- `"conditioning"` - Wireframe fed to ControlNet
- `"mask"` - Surgical mask
- `"landmarks_original"` - Input landmarks
- `"landmarks_manipulated"` - Deformed landmarks
- `"identity_check"` - ArcFace similarity score

**Example:**
```python
from landmarkdiff.inference import LandmarkDiffPipeline

pipeline = LandmarkDiffPipeline(mode="controlnet", device="cuda")
pipeline.load()
result = pipeline.generate(
    image,
    procedure="rhinoplasty",
    intensity=60,
    seed=42
)
```

#### `mask_composite(original, generated, mask) -> PIL.Image`

Composite generated face onto original using feathered mask with LAB skin tone matching.
