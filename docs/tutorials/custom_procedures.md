# Custom Procedure Presets

Define your own surgical procedure by specifying which landmarks to move and how.

## How it works

LandmarkDiff uses **Gaussian RBF (Radial Basis Function) deformation** to move facial landmarks. Each procedure is defined by a list of **deformation handles** - anchor points on the face that get displaced in a specific direction.

A deformation handle has three components:
- **landmark_index**: Which MediaPipe landmark to move (0-477)
- **displacement**: Direction and magnitude of movement `[dx, dy, dz]` in pixels
- **influence_radius**: How far the influence spreads to neighboring landmarks

## Step 1: Identify landmarks

The MediaPipe Face Mesh provides 478 landmarks. Key regions:

| Region | Landmark indices |
|--------|-----------------|
| Nose bridge | 6, 197, 195, 5 |
| Nose tip | 4, 1 |
| Nose wings | 48, 278 |
| Left eye | 33, 133, 159, 145 |
| Right eye | 362, 263, 386, 374 |
| Upper lip | 13, 14, 82, 312 |
| Lower lip | 17, 15, 87, 317 |
| Chin | 152, 175, 148, 377 |
| Jawline left | 234, 132, 58 |
| Jawline right | 454, 361, 288 |
| Left brow | 70, 63, 105, 66 |
| Right brow | 300, 293, 334, 296 |

## Step 2: Define your procedure

```python
import numpy as np
from landmarkdiff.manipulation import DeformationHandle, gaussian_rbf_deform

# Example: mentoplasty (chin advancement)
mentoplasty_handles = [
    # Move chin tip forward and slightly down
    DeformationHandle(landmark_index=152, displacement=np.array([0, 5, -8]), influence_radius=30.0),
    # Move lower chin forward
    DeformationHandle(landmark_index=175, displacement=np.array([0, 3, -6]), influence_radius=25.0),
    # Adjust chin contour
    DeformationHandle(landmark_index=148, displacement=np.array([0, 2, -4]), influence_radius=20.0),
    DeformationHandle(landmark_index=377, displacement=np.array([0, 2, -4]), influence_radius=20.0),
]
```

## Step 3: Register the preset

Add your procedure's landmark indices and radius to `PROCEDURE_LANDMARKS` and `PROCEDURE_RADIUS` in `landmarkdiff/manipulation.py`, then add the displacement logic in `_get_procedure_handles()`:

```python
PROCEDURE_LANDMARKS["mentoplasty"] = [148, 152, 175, 377]
PROCEDURE_RADIUS["mentoplasty"] = 25.0
```

## Step 4: Test it

```python
import numpy as np
from PIL import Image
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.conditioning import render_wireframe

img = np.array(Image.open("face.jpg").convert("RGB").resize((512, 512)))
landmarks = extract_landmarks(img)
deformed = apply_procedure_preset(landmarks, "mentoplasty", intensity=60)

# Visualize the deformation
original_mesh = render_wireframe(landmarks, (512, 512))
deformed_mesh = render_wireframe(deformed, (512, 512))
```

## Step 5: Add a test

In `tests/test_manipulation.py`:

```python
def test_mentoplasty_preset():
    landmarks = create_dummy_landmarks()
    result = apply_procedure_preset(landmarks, "mentoplasty", intensity=50)
    assert result is not None
    # Chin landmarks should have moved
    assert not np.allclose(result.landmarks[152], landmarks.landmarks[152])
```

## Tips

- Start with small displacements (3-8 pixels) and adjust
- Use larger radii (25-40) for smooth, natural-looking deformations
- Use smaller radii (10-15) for localized changes
- Test with multiple face shapes - the same displacement can look different on different faces
- The intensity parameter (0-100) scales all displacements linearly
