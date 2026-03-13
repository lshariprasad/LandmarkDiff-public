# landmarkdiff.clinical

Clinical edge case handling for patients with special conditions.

## Classes

### `ClinicalFlags`

Dataclass for clinical condition flags.

**Attributes:**
- `vitiligo` (bool): Patient has vitiligo - reduces mask intensity in depigmented patches
- `bells_palsy` (bool): Patient has Bell's palsy - disables deformation on paralyzed side
- `bells_palsy_side` (str): Affected side (`"left"` or `"right"`)
- `keloid` (bool): Patient is keloid-prone - softens mask transitions
- `ehlers_danlos` (bool): Patient has Ehlers-Danlos syndrome - widens RBF influence radii (1.5x)

## Functions

### `detect_vitiligo_patches(image) -> np.ndarray`

Detect depigmented skin patches using LAB color space analysis.

**Returns:** Binary mask of detected vitiligo patches

### `adjust_mask_for_vitiligo(mask, vitiligo_patches) -> np.ndarray`

Reduce mask intensity in vitiligo-affected regions to prevent color artifacts.

### `disable_paralyzed_side(handles, side) -> list[DeformationHandle]`

Remove deformation handles on the paralyzed side for Bell's palsy patients.

### `soften_keloid_transitions(mask, sigma_multiplier=2.0) -> np.ndarray`

Apply additional Gaussian blur to mask boundaries in keloid-prone regions.

### `widen_influence_radii(handles, multiplier=1.5) -> list[DeformationHandle]`

Increase deformation handle radii for Ehlers-Danlos patients with hypermobile tissue.

## Usage

```python
from landmarkdiff.clinical import ClinicalFlags
from landmarkdiff.inference import LandmarkDiffPipeline

pipeline = LandmarkDiffPipeline(mode="controlnet", device="cuda")
pipeline.load()

flags = ClinicalFlags(
    vitiligo=True,
    bells_palsy=False,
    keloid=True,
    ehlers_danlos=False
)

img = np.array(Image.open("patient.jpg").convert("RGB").resize((512, 512)))
result = pipeline.generate(
    img,
    procedure="rhinoplasty",
    intensity=60,
    clinical_flags=flags
)
```
