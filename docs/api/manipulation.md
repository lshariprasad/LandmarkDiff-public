# landmarkdiff.manipulation

Gaussian RBF landmark deformation for surgical simulation.

## Classes

### `DeformationHandle`

A single deformation control point.

**Attributes:**
- `landmark_index` (int): MediaPipe landmark index (0-477)
- `displacement` (np.ndarray): Movement vector `[dx, dy, dz]` in pixels
- `influence_radius` (float): Gaussian influence radius in pixels

## Constants

### `PROCEDURE_LANDMARKS`

Pre-defined landmark index sets for supported procedures:
- `"rhinoplasty"` - nose reshaping
- `"blepharoplasty"` - eyelid surgery
- `"rhytidectomy"` - facelift
- `"orthognathic"` - jaw surgery
- `"brow_lift"` - brow elevation
- `"mentoplasty"` - chin surgery

## Functions

### `gaussian_rbf_deform(landmarks, handle) -> np.ndarray`

Apply a single Gaussian RBF deformation handle to a landmark array.

**Parameters:**
- `landmarks` (np.ndarray): Landmark coordinates, shape `(478, 2)` or `(478, 3)`
- `handle` (DeformationHandle): Single deformation control handle

**Returns:** New `np.ndarray` with deformed positions (copy of input)

### `apply_procedure_preset(face, procedure, intensity=50.0, ...) -> FaceLandmarks`

Apply a named procedure preset.

**Parameters:**
- `face` (FaceLandmarks): Input face landmarks
- `procedure` (str): One of `"rhinoplasty"`, `"blepharoplasty"`, `"rhytidectomy"`, `"orthognathic"`, `"brow_lift"`, `"mentoplasty"`
- `intensity` (float): Deformation strength, 0 to 100 (default: 50.0)
- `image_size` (int): Reference image size for displacement scaling (default: 512)
- `clinical_flags` (ClinicalFlags | None): Clinical edge case flags (default: None)
- `displacement_model_path` (str | None): Path to data-driven displacement model (default: None)
- `noise_scale` (float): Random noise added to displacements (default: 0.0)

**Returns:** New `FaceLandmarks` with deformed positions

**Example:**
```python
import numpy as np
from PIL import Image
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset

img = np.array(Image.open("face.jpg").convert("RGB").resize((512, 512)))
landmarks = extract_landmarks(img)
deformed = apply_procedure_preset(landmarks, "rhinoplasty", intensity=60)
```
