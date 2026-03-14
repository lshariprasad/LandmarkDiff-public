#!/usr/bin/env python3
"""Generate qualitative comparison figures for the paper.

Creates publication-quality figure grids showing:
1. Pipeline overview (input → landmarks → conditioning → output)
2. Multi-procedure comparison (4 procedures × N faces)
3. Intensity sweep (25%, 50%, 75%, 100% for one face)
4. Baseline comparison (input, TPS, morphing, LandmarkDiff, target)
5. Failure cases and limitations

Usage:
    # Pipeline figure from single image
    python scripts/generate_paper_figures.py \
        --input data/faces_all/000001.png --output paper/figures/

    # Multi-procedure grid from directory
    python scripts/generate_paper_figures.py \
        --input_dir data/faces_all/ --output paper/figures/ \
        --max_images 4 --figure all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.conditioning import generate_conditioning, render_wireframe
from landmarkdiff.inference import mask_composite
from landmarkdiff.landmarks import extract_landmarks, render_landmark_image
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps

PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic"]
PROC_SHORT = {
    "rhinoplasty": "Rhino",
    "blepharoplasty": "Bleph",
    "rhytidectomy": "Rhytid",
    "orthognathic": "Orthog",
}


def add_label(
    img: np.ndarray,
    text: str,
    position: str = "bottom",
    font_scale: float = 0.5,
    color=(255, 255, 255),
) -> np.ndarray:
    """Add a text label to an image."""
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    if position == "bottom":
        x = (img.shape[1] - tw) // 2
        y = img.shape[0] - 10
    elif position == "top":
        x = (img.shape[1] - tw) // 2
        y = th + 10
    else:
        x, y = 10, th + 10

    # Background rectangle for readability
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return img


def make_border(img: np.ndarray, size: int = 2, color=(255, 255, 255)) -> np.ndarray:
    """Add a border around an image."""
    return cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_CONSTANT, value=color)


def figure_pipeline(image_path: str, output_dir: Path, size: int = 256):
    """Generate pipeline overview figure.

    Shows: Input → Landmarks → Wireframe → Canny → Mask → TPS → Composite
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read {image_path}")
        return
    img = cv2.resize(img, (512, 512))
    face = extract_landmarks(img)
    if face is None:
        print(f"No face in {image_path}")
        return

    procedure = "rhinoplasty"

    # Generate all pipeline stages
    landmark_img = render_landmark_image(face, 512, 512)
    wireframe = render_wireframe(face, 512, 512)
    _, canny, _ = generate_conditioning(face, 512, 512)

    manip = apply_procedure_preset(face, procedure, 65.0, image_size=512)
    mask = generate_surgical_mask(face, procedure, 512, 512)
    warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
    composite = mask_composite(warped, img, mask)

    # Post-manipulation conditioning
    manip_wireframe = render_wireframe(manip, 512, 512)

    # Build panels
    panels = []
    labels = ["Input", "Landmarks", "Wireframe", "Canny", "Mask", "Warped", "Output"]
    images = [img, landmark_img, wireframe, canny, mask, warped, composite]

    for label, panel in zip(labels, images):
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(panel, (size, size))
        panel = add_label(panel, label, "bottom")
        panels.append(panel)

    grid = np.hstack(panels)
    out_path = output_dir / "fig_pipeline.png"
    cv2.imwrite(str(out_path), grid)
    print(f"Pipeline figure: {out_path} ({grid.shape[1]}x{grid.shape[0]})")
    return grid


def figure_procedures(image_path: str, output_dir: Path, size: int = 256):
    """Generate multi-procedure comparison figure.

    Shows one input face with all 4 procedures applied.
    Rows: Input | Rhinoplasty | Blepharoplasty | Rhytidectomy | Orthognathic
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.resize(img, (512, 512))
    face = extract_landmarks(img)
    if face is None:
        return

    panels = [add_label(cv2.resize(img, (size, size)), "Input", "bottom")]

    for proc in PROCEDURES:
        manip = apply_procedure_preset(face, proc, 65.0, image_size=512)
        mask = generate_surgical_mask(face, proc, 512, 512)
        warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
        composite = mask_composite(warped, img, mask)
        panel = cv2.resize(composite, (size, size))
        panel = add_label(panel, PROC_SHORT[proc], "bottom")
        panels.append(panel)

    row = np.hstack(panels)
    out_path = output_dir / "fig_procedures.png"
    cv2.imwrite(str(out_path), row)
    print(f"Procedures figure: {out_path}")
    return row


def figure_intensity_sweep(
    image_path: str, output_dir: Path, procedure: str = "rhinoplasty", size: int = 256
):
    """Generate intensity sweep figure.

    Shows: Input | 25% | 50% | 75% | 100%
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.resize(img, (512, 512))
    face = extract_landmarks(img)
    if face is None:
        return

    panels = [add_label(cv2.resize(img, (size, size)), "Original", "bottom")]
    intensities = [25, 50, 75, 100]

    for intensity in intensities:
        manip = apply_procedure_preset(face, procedure, float(intensity), image_size=512)
        mask = generate_surgical_mask(face, procedure, 512, 512)
        warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
        composite = mask_composite(warped, img, mask)
        panel = cv2.resize(composite, (size, size))
        panel = add_label(panel, f"{intensity}%", "bottom")
        panels.append(panel)

    row = np.hstack(panels)
    out_path = output_dir / f"fig_intensity_{procedure}.png"
    cv2.imwrite(str(out_path), row)
    print(f"Intensity sweep: {out_path}")
    return row


def figure_multi_face_grid(input_dir: str, output_dir: Path, max_images: int = 4, size: int = 192):
    """Generate multi-face × multi-procedure grid.

    Rows: one per face
    Columns: Input | Rhino | Bleph | Rhytid | Orthog
    """
    input_path = Path(input_dir)
    image_files = sorted(input_path.glob("*.png"))[: max_images * 3]

    rows = []
    count = 0

    for img_path in image_files:
        if count >= max_images:
            break

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (512, 512))
        face = extract_landmarks(img)
        if face is None:
            continue

        panels = [cv2.resize(img, (size, size))]
        for proc in PROCEDURES:
            manip = apply_procedure_preset(face, proc, 65.0, image_size=512)
            mask = generate_surgical_mask(face, proc, 512, 512)
            warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
            composite = mask_composite(warped, img, mask)
            panels.append(cv2.resize(composite, (size, size)))

        # Add labels to first row only
        if count == 0:
            labels = ["Input"] + [PROC_SHORT[p] for p in PROCEDURES]
            panels = [add_label(p, l, "top") for p, l in zip(panels, labels)]

        rows.append(np.hstack(panels))
        count += 1

    if not rows:
        print("No valid faces found!")
        return

    grid = np.vstack(rows)
    out_path = output_dir / "fig_multi_face_grid.png"
    cv2.imwrite(str(out_path), grid)
    print(f"Multi-face grid: {out_path} ({count} faces × 4 procedures)")
    return grid


def figure_conditioning_comparison(image_path: str, output_dir: Path, size: int = 256):
    """Generate conditioning ablation figure.

    Shows: Input | Mesh Wireframe | Canny Edges | Mesh+Canny | Full Conditioning
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.resize(img, (512, 512))
    face = extract_landmarks(img)
    if face is None:
        return

    landmark_img, canny, wireframe = generate_conditioning(face, 512, 512)

    # Combine mesh + canny
    mesh_canny = cv2.add(wireframe, canny)

    # Full 3-channel conditioning (how it's actually fed to ControlNet)
    full_cond = np.stack([wireframe, canny, wireframe], axis=-1)

    panels = [
        (img, "Input"),
        (cv2.cvtColor(wireframe, cv2.COLOR_GRAY2BGR), "Mesh Wireframe"),
        (cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR), "Canny Edges"),
        (cv2.cvtColor(mesh_canny, cv2.COLOR_GRAY2BGR), "Mesh + Canny"),
        (landmark_img, "Landmarks"),
        (full_cond, "Full Conditioning"),
    ]

    grid_panels = []
    for panel_img, label in panels:
        p = cv2.resize(panel_img, (size, size))
        p = add_label(p, label, "bottom")
        grid_panels.append(p)

    row = np.hstack(grid_panels)
    out_path = output_dir / "fig_conditioning.png"
    cv2.imwrite(str(out_path), row)
    print(f"Conditioning figure: {out_path}")
    return row


def figure_mask_comparison(image_path: str, output_dir: Path, size: int = 256):
    """Generate surgical mask comparison figure (per-procedure masks)."""
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.resize(img, (512, 512))
    face = extract_landmarks(img)
    if face is None:
        return

    panels = [add_label(cv2.resize(img, (size, size)), "Input", "bottom")]

    for proc in PROCEDURES:
        mask = generate_surgical_mask(face, proc, 512, 512)
        # Overlay mask on image for visualization
        mask_vis = img.copy()
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        overlay_color = np.array([0, 0, 200], dtype=np.float32)  # red tint
        mask_vis = (
            mask_vis.astype(np.float32) * (1 - mask_3ch * 0.4) + overlay_color * mask_3ch * 0.4
        ).astype(np.uint8)
        panel = cv2.resize(mask_vis, (size, size))
        panel = add_label(panel, PROC_SHORT[proc], "bottom")
        panels.append(panel)

    row = np.hstack(panels)
    out_path = output_dir / "fig_masks.png"
    cv2.imwrite(str(out_path), row)
    print(f"Mask comparison: {out_path}")
    return row


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--input", default=None, help="Single image path")
    parser.add_argument("--input_dir", default=None, help="Directory of images")
    parser.add_argument("--output", default="paper/figures", help="Output directory")
    parser.add_argument("--max_images", type=int, default=4)
    parser.add_argument("--size", type=int, default=256, help="Panel size")
    parser.add_argument(
        "--figure",
        default="all",
        choices=["all", "pipeline", "procedures", "intensity", "grid", "conditioning", "masks"],
        help="Which figure(s) to generate",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find an input image
    image_path = args.input
    if image_path is None:
        if args.input_dir:
            candidates = sorted(Path(args.input_dir).glob("*.png"))
            if candidates:
                image_path = str(candidates[0])
        if image_path is None:
            # Try common locations
            for candidate in [
                "data/faces_all/000001.png",
                "data/celeba_hq_extracted/000001.png",
            ]:
                if Path(candidate).exists():
                    image_path = candidate
                    break

    if image_path is None and args.figure != "grid":
        print("No input image found. Provide --input or --input_dir.")
        sys.exit(1)

    figures_generated = 0

    if args.figure in ("all", "pipeline") and image_path:
        figure_pipeline(image_path, output_dir, args.size)
        figures_generated += 1

    if args.figure in ("all", "procedures") and image_path:
        figure_procedures(image_path, output_dir, args.size)
        figures_generated += 1

    if args.figure in ("all", "intensity") and image_path:
        for proc in PROCEDURES:
            figure_intensity_sweep(image_path, output_dir, proc, args.size)
            figures_generated += 1

    if args.figure in ("all", "conditioning") and image_path:
        figure_conditioning_comparison(image_path, output_dir, args.size)
        figures_generated += 1

    if args.figure in ("all", "masks") and image_path:
        figure_mask_comparison(image_path, output_dir, args.size)
        figures_generated += 1

    if args.figure in ("all", "grid") and args.input_dir:
        figure_multi_face_grid(args.input_dir, output_dir, args.max_images, args.size)
        figures_generated += 1

    print(f"\n{figures_generated} figures generated in {output_dir}/")


if __name__ == "__main__":
    main()
