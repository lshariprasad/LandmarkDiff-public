"""Compute baseline metrics on HDA test set for paper Table 1.

Evaluates TPS-only and morphing baselines against real after-surgery targets.
Generates per-procedure metrics and overall averages.

Usage:
    python scripts/compute_baselines_hda.py --test_dir data/hda_splits/test
"""

from __future__ import annotations

import argparse
import json

# Add project root
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.evaluation import (
    compute_identity_similarity,
    compute_lpips,
    compute_nme,
    compute_ssim,
)
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.synthetic.tps_warp import warp_image_tps


def discover_test_pairs(test_dir: Path) -> list[dict]:
    """Discover input/target pairs and their procedures."""
    prefixes = sorted(set(f.stem.rsplit("_", 1)[0] for f in test_dir.glob("*_input.png")))
    pairs = []
    for prefix in prefixes:
        input_path = test_dir / f"{prefix}_input.png"
        target_path = test_dir / f"{prefix}_target.png"
        conditioning_path = test_dir / f"{prefix}_conditioning.png"
        mask_path = test_dir / f"{prefix}_mask.png"

        if not target_path.exists():
            continue

        # Extract procedure from prefix (e.g., "rhinoplasty_Nose_123")
        procedure = prefix.split("_")[0]
        pairs.append(
            {
                "prefix": prefix,
                "procedure": procedure,
                "input": str(input_path),
                "target": str(target_path),
                "conditioning": str(conditioning_path),
                "mask": str(mask_path),
            }
        )
    return pairs


def compute_tps_baseline(input_img: np.ndarray, target_img: np.ndarray) -> dict:
    """Compute TPS-only baseline metrics.

    TPS warp from input landmarks to target landmarks, then compare.
    """

    input_lm = extract_landmarks(input_img)
    target_lm = extract_landmarks(target_img)

    if input_lm is None or target_lm is None:
        return None

    # TPS warp input to match target landmarks
    try:
        warped = warp_image_tps(input_img, input_lm.pixel_coords, target_lm.pixel_coords)
    except Exception:
        return None

    ssim = compute_ssim(warped, target_img)
    lpips = compute_lpips(warped, target_img)
    # Detect landmarks on warped image and compare to target
    warped_lm = extract_landmarks(warped)
    if warped_lm is not None:
        nme = compute_nme(warped_lm.pixel_coords, target_lm.pixel_coords)
    else:
        nme = float("nan")

    # Identity similarity
    id_sim = compute_identity_similarity(warped, target_img)

    return {
        "ssim": ssim,
        "lpips": lpips,
        "nme": nme,
        "identity_sim": id_sim,
        "warped": warped,
    }


def compute_morphing_baseline(input_img: np.ndarray, target_img: np.ndarray) -> dict:
    """Compute affine morphing baseline (Crisalix-style).

    Simple affine warp using 3 control points (eyes + nose tip).
    """

    input_lm = extract_landmarks(input_img)
    target_lm = extract_landmarks(target_img)

    if input_lm is None or target_lm is None:
        return None

    h, w = input_img.shape[:2]

    # Use key facial points for affine transform
    # Left eye center (avg of landmarks 33, 133), right eye (263, 362), nose tip (1)
    src_pts = np.float32(
        [
            input_lm.pixel_coords[33],
            input_lm.pixel_coords[263],
            input_lm.pixel_coords[1],
        ]
    )
    dst_pts = np.float32(
        [
            target_lm.pixel_coords[33],
            target_lm.pixel_coords[263],
            target_lm.pixel_coords[1],
        ]
    )

    M = cv2.getAffineTransform(src_pts, dst_pts)
    morphed = cv2.warpAffine(input_img, M, (w, h))

    ssim = compute_ssim(morphed, target_img)
    lpips = compute_lpips(morphed, target_img)

    # Detect landmarks on morphed image and compare to target
    morphed_lm = extract_landmarks(morphed)
    if morphed_lm is not None:
        nme = compute_nme(morphed_lm.pixel_coords, target_lm.pixel_coords)
    else:
        nme = float("nan")

    # Identity similarity
    id_sim = compute_identity_similarity(morphed, target_img)

    return {
        "ssim": ssim,
        "lpips": lpips,
        "nme": nme,
        "identity_sim": id_sim,
        "morphed": morphed,
    }


def compute_direct_metrics(input_img: np.ndarray, target_img: np.ndarray) -> dict:
    """Compute metrics between input and target directly (no transformation)."""

    ssim = compute_ssim(input_img, target_img)
    lpips = compute_lpips(input_img, target_img)

    # NME between input and target landmarks
    input_lm = extract_landmarks(input_img)
    target_lm = extract_landmarks(target_img)
    if input_lm is not None and target_lm is not None:
        nme = compute_nme(input_lm.pixel_coords, target_lm.pixel_coords)
    else:
        nme = float("nan")

    # Identity similarity
    id_sim = compute_identity_similarity(input_img, target_img)

    return {"ssim": ssim, "lpips": lpips, "nme": nme, "identity_sim": id_sim}


def main():
    parser = argparse.ArgumentParser(description="Compute baseline metrics on HDA test set")
    parser.add_argument("--test_dir", default="data/hda_splits/test")
    parser.add_argument("--output", default="paper/baseline_results.json")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    pairs = discover_test_pairs(test_dir)
    print(f"Found {len(pairs)} test pairs")

    # Group by procedure
    proc_counts = defaultdict(int)
    for p in pairs:
        proc_counts[p["procedure"]] += 1
    print(f"Procedures: {dict(proc_counts)}")

    # Collect metrics
    results = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)

    for i, pair in enumerate(pairs):
        input_img = cv2.imread(pair["input"])
        target_img = cv2.imread(pair["target"])
        proc = pair["procedure"]

        if input_img is None or target_img is None:
            continue

        # Resize both to 512x512
        input_img = cv2.resize(input_img, (512, 512))
        target_img = cv2.resize(target_img, (512, 512))

        # Direct (no transform) baseline
        direct = compute_direct_metrics(input_img, target_img)
        if direct:
            results[proc]["direct_ssim"].append(direct["ssim"])
            results[proc]["direct_lpips"].append(direct["lpips"])
            if not np.isnan(direct.get("nme", float("nan"))):
                results[proc]["direct_nme"].append(direct["nme"])
            if not np.isnan(direct.get("identity_sim", float("nan"))):
                results[proc]["direct_identity_sim"].append(direct["identity_sim"])

        # TPS baseline
        tps = compute_tps_baseline(input_img, target_img)
        if tps:
            results[proc]["tps_ssim"].append(tps["ssim"])
            results[proc]["tps_lpips"].append(tps["lpips"])
            if not np.isnan(tps.get("nme", float("nan"))):
                results[proc]["tps_nme"].append(tps["nme"])
            if not np.isnan(tps.get("identity_sim", float("nan"))):
                results[proc]["tps_identity_sim"].append(tps["identity_sim"])

        # Morphing baseline
        morph = compute_morphing_baseline(input_img, target_img)
        if morph:
            results[proc]["morph_ssim"].append(morph["ssim"])
            results[proc]["morph_lpips"].append(morph["lpips"])
            if not np.isnan(morph.get("nme", float("nan"))):
                results[proc]["morph_nme"].append(morph["nme"])
            if not np.isnan(morph.get("identity_sim", float("nan"))):
                results[proc]["morph_identity_sim"].append(morph["identity_sim"])

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} pairs")

    # Aggregate
    output = {}
    for proc in sorted(results.keys()):
        output[proc] = {}
        for metric, values in sorted(results[proc].items()):
            arr = np.array(values)
            output[proc][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "n": len(arr),
            }

    # Overall
    for metric_key in [
        "tps_ssim",
        "tps_lpips",
        "tps_nme",
        "tps_identity_sim",
        "morph_ssim",
        "morph_lpips",
        "morph_nme",
        "morph_identity_sim",
        "direct_ssim",
        "direct_lpips",
        "direct_nme",
        "direct_identity_sim",
    ]:
        all_vals = []
        for proc in results:
            all_vals.extend(results[proc].get(metric_key, []))
        if all_vals:
            arr = np.array(all_vals)
            output.setdefault("overall", {})[metric_key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "n": len(arr),
            }

    # Print results
    print("\n" + "=" * 80)
    print("BASELINE RESULTS ON HDA TEST SET")
    print("=" * 80)

    for proc in sorted(output.keys()):
        print(f"\n--- {proc.upper()} ---")
        for metric, stats in sorted(output[proc].items()):
            print(f"  {metric:20s}: {stats['mean']:.4f} +/- {stats['std']:.4f} (n={stats['n']})")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
