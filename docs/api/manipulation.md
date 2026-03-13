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

## Functions

### `gaussian_rbf_deform(landmarks, handles, intensity=1.0) -> FaceLandmarks`

Apply Gaussian RBF deformation to landmarks.

**Parameters:**
- `landmarks` (FaceLandmarks): Input landmarks
- `handles` (list[DeformationHandle]): Deformation control handles
- `intensity` (float): Deformation strength, 0 to 100

**Returns:** New `FaceLandmarks` with deformed positions

### `apply_procedure_preset(landmarks, procedure, intensity=1.0) -> FaceLandmarks`

Apply a named procedure preset.

**Parameters:**
- `landmarks` (FaceLandmarks): Input landmarks
- `procedure` (str): One of `"rhinoplasty"`, `"blepharoplasty"`, `"rhytidectomy"`, `"orthognathic"`
- `intensity` (float): Deformation strength, 0 to 100

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
