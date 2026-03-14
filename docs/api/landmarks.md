# landmarkdiff.landmarks

Face mesh extraction using MediaPipe Face Mesh v2.

## Classes

### `FaceLandmarks`

Dataclass holding 478-point facial landmark data.

**Attributes:**
- `landmarks` (np.ndarray): Shape `(478, 3)` - x, y, z coordinates normalized to [0, 1]
- `image_width` (int): Source image width
- `image_height` (int): Source image height

**Methods:**
- `pixel_coords -> np.ndarray`: Convert normalized coords to pixel coordinates (property)
- `get_region(name: str) -> np.ndarray`: Get landmark indices for a named region

### Region names

| Region | Indices | Count |
|--------|---------|-------|
| `left_eye` | 33, 133, 159, 145, ... | 16 |
| `right_eye` | 362, 263, 386, 374, ... | 16 |
| `nose` | 1, 4, 5, 6, 48, 278, ... | 20 |
| `mouth` | 13, 14, 17, 82, 312, ... | 20 |
| `left_brow` | 70, 63, 105, 66, 107 | 5 |
| `right_brow` | 300, 293, 334, 296, 336 | 5 |
| `jawline` | 234, 454, 132, 361, ... | 17 |
| `forehead` | Various | 10 |

## Functions

### `extract_landmarks(image, min_detection_confidence=0.5, min_tracking_confidence=0.5) -> FaceLandmarks | None`

Extract facial landmarks from an image using MediaPipe Face Mesh.

**Parameters:**
- `image` (np.ndarray): Input face image (BGR numpy array)
- `min_detection_confidence` (float): Minimum detection confidence (default: 0.5)
- `min_tracking_confidence` (float): Minimum tracking confidence (default: 0.5)

**Returns:** `FaceLandmarks` object, or `None` if no face detected

**Example:**
```python
from landmarkdiff.landmarks import extract_landmarks

import numpy as np
from PIL import Image

img = np.array(Image.open("photo.jpg").convert("RGB").resize((512, 512)))
landmarks = extract_landmarks(img)
print(f"Detected {len(landmarks.landmarks)} landmarks")
print(f"Nose tip: {landmarks.landmarks[4]}")
```
