#!/usr/bin/env python3
"""Reddit posting automation for LandmarkDiff project.

Automates posting research announcements and helpful comments across
relevant subreddits. All content is genuine and technically accurate,
sharing a real MICCAI 2026 research project.

Usage:
    # Dry run a post to r/MachineLearning
    python scripts/reddit_poster.py \
        --client-id YOUR_ID --client-secret YOUR_SECRET \
        --password YOUR_PASSWORD \
        --mode post --subreddit MachineLearning --dry-run

    # Post to r/computervision
    python scripts/reddit_poster.py \
        --client-id YOUR_ID --client-secret YOUR_SECRET \
        --password YOUR_PASSWORD \
        --mode post --subreddit computervision

    # Leave helpful comments on trending CV posts
    python scripts/reddit_poster.py \
        --client-id YOUR_ID --client-secret YOUR_SECRET \
        --password YOUR_PASSWORD \
        --mode comment --subreddit computervision --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import praw  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USERNAME = "dreamlessxx"
USER_AGENT = "LandmarkDiff-poster/0.1.0 by u/dreamlessxx"

GITHUB_URL = "https://github.com/dreamlessx/LandmarkDiff-public"
HF_SPACE_URL = "https://huggingface.co/spaces/dreamlessx/LandmarkDiff"
WIKI_URL = "https://github.com/dreamlessx/LandmarkDiff-public/wiki"

# Rate-limiting: minimum seconds between posts
MIN_POST_INTERVAL_SECONDS = 600  # 10 minutes
MIN_COMMENT_INTERVAL_SECONDS = 120  # 2 minutes

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "reddit_poster.log"

SUPPORTED_SUBREDDITS = [
    "MachineLearning",
    "computervision",
    "StableDiffusion",
    "PlasticSurgery",
    "opensource",
]

# ---------------------------------------------------------------------------
# Post templates
# ---------------------------------------------------------------------------

POST_TEMPLATES: dict[str, dict[str, Any]] = {
    "MachineLearning": {
        "title": (
            "[R] LandmarkDiff: Anatomically-Conditioned Diffusion for"
            " Surgical Outcome Prediction from Clinical Photography"
        ),
        "body": f"""\
Hi r/MachineLearning,

I've been working on LandmarkDiff, an open-source system for predicting \
facial surgery outcomes from standard clinical photographs. Sharing it \
here for feedback from the ML community.

**What it does**

Given a single frontal face photo and a selected surgical procedure, \
LandmarkDiff synthesizes a plausible post-operative appearance while \
preserving patient identity. It supports six procedures: rhinoplasty, \
blepharoplasty (eyelid), rhytidectomy (facelift), orthognathic (jaw), \
brow lift, and mentoplasty (chin).

**Technical contribution**

The pipeline has three stages:

1. **Landmark extraction** -- MediaPipe FaceMesh extracts 478 3D facial \
landmarks that define anatomically meaningful control points across the face.

2. **Procedure-specific deformation** -- Gaussian RBF (radial basis function) \
interpolation drives landmark displacements that correspond to each \
procedure's anatomical effect. A data-driven displacement mode fits \
displacement fields from real before/after surgery pairs.

3. **Conditioned synthesis** -- Deformed landmark positions are rendered as a \
wireframe and fed as conditioning input to a ControlNet-conditioned Stable \
Diffusion 1.5 pipeline (CrucibleAI backbone). The wireframe spatially \
grounds the diffusion process to the predicted anatomy.

Post-processing: CodeFormer face restoration -> Real-ESRGAN upscaling -> \
histogram matching -> unsharp sharpening -> Laplacian blend.

The system also includes a **facial symmetry analysis** module that \
quantifies left-right facial symmetry before and after simulated procedures.

**Inference modes**

- `tps` (thin-plate spline, CPU-only): pure warp, no diffusion, very fast
- `img2img`: SD1.5 img2img conditioned on the warped image
- `controlnet`: ControlNet conditioning on the deformed wireframe
- `controlnet_ip`: ControlNet + IP-Adapter for stronger identity preservation

The CPU TPS demo is live without any GPU requirement.

**Links**

- GitHub (MIT license): {GITHUB_URL}
- Live demo (HF Spaces, CPU): {HF_SPACE_URL}
- Wiki / docs: {WIKI_URL}

**Disclaimer**: This is a research tool, not medical advice. Outputs are \
synthetic visualizations intended for research and educational purposes only.

Happy to discuss the deformation model, conditioning strategy, or the \
facial symmetry analysis in the comments.
""",
        "flair": "Research",
    },
    "computervision": {
        "title": (
            "LandmarkDiff: MediaPipe 478-point mesh + Gaussian RBF"
            " deformation + ControlNet for surgical face prediction"
        ),
        "body": f"""\
Built a system that uses MediaPipe FaceMesh (478 3D landmarks) as the \
geometric backbone for predicting facial surgery outcomes from a single \
clinical photo.

**Core vision pipeline:**

- Extract 478 anatomical control points with MediaPipe FaceMesh
- Apply procedure-specific Gaussian RBF displacements (rhinoplasty, brow \
lift, facelift, etc.)
- Fit thin-plate spline (TPS) warp from original to displaced control points
- Apply warp to the full image -- fast, smooth, and anatomically consistent
- Optionally pass the warped result through ControlNet + SD1.5 for texture \
synthesis

The system also provides a **facial symmetry analysis** feature, quantifying \
left-right asymmetry before and after simulated deformations using the \
landmark mesh.

The TPS CPU demo runs in-browser on HF Spaces, no GPU needed. The ControlNet \
GPU pipeline produces more photorealistic outputs by grounding diffusion to \
the deformed wireframe.

The project is open source (MIT) and CPU-runnable in its lightweight mode.

Demo: {HF_SPACE_URL}
Code: {GITHUB_URL}

Disclaimer: research tool, not medical advice.
""",
        "flair": None,
    },
    "StableDiffusion": {
        "title": (
            "Using facial mesh wireframes as ControlNet conditioning for"
            " surgery outcome prediction -- LandmarkDiff"
        ),
        "body": f"""\
Sharing a project that uses ControlNet conditioning in an unusual way: \
facial anatomy as a spatial prior for surgical outcome synthesis.

**The ControlNet pipeline**

Standard ControlNet-based generation takes a conditioning image (edges, \
pose, depth, etc.) and guides the diffusion process spatially. In \
LandmarkDiff, the conditioning input is a rendered wireframe of *deformed* \
facial landmarks -- the face mesh after procedure-specific landmark \
displacements are applied.

The deformation is driven by Gaussian RBF interpolation on MediaPipe's \
478-point 3D face mesh. Procedure presets encode anatomically meaningful \
control point movements for six surgeries (rhinoplasty, blepharoplasty, \
rhytidectomy, orthognathic, brow lift, mentoplasty).

**Full pipeline:**

1. MediaPipe extracts 478 3D landmarks from a clinical photo
2. Procedure displacements are applied to relevant landmark subsets (e.g., \
nasal tip + alar landmarks for rhinoplasty, orbital landmarks for \
blepharoplasty)
3. Deformed landmarks are projected back to 2D and rendered as a mesh wireframe
4. Wireframe -> ControlNet (CrucibleAI backbone) -> SD1.5 with the original \
photo as init
5. Optional IP-Adapter layer for stronger identity preservation \
(`controlnet_ip` mode)
6. CodeFormer + Real-ESRGAN post-processing

The wireframe conditioning means the diffusion model is spatially anchored \
to the predicted anatomy rather than being free to move facial features \
arbitrarily. Also includes facial symmetry analysis for quantitative \
assessment.

There is also a lightweight CPU mode (`tps`) that skips diffusion entirely \
and produces a pure TPS warp -- useful for fast preview without GPU. It's \
fully open source (MIT) and CPU-runnable.

Code (MIT): {GITHUB_URL}
Demo (CPU TPS + GPU ControlNet): {HF_SPACE_URL}

Disclaimer: research tool, not medical advice.
""",
        "flair": None,
    },
    "PlasticSurgery": {
        "title": (
            "Open-source research tool for visualizing facial surgery"
            " outcomes from a single photo -- LandmarkDiff"
        ),
        "body": f"""\
Hi r/PlasticSurgery,

I want to share an open-source research project that may be of interest to \
people here who are curious about what technology can (and cannot) do for \
surgical planning visualization.

**What it is**

LandmarkDiff is a research system that takes a standard facial photo and \
a selected procedure, and synthesizes a plausible visualization of what a \
patient might look like post-operatively. It supports six procedures:

- Rhinoplasty (nose reshaping)
- Blepharoplasty (eyelid surgery)
- Rhytidectomy (facelift)
- Orthognathic surgery (jaw repositioning)
- Brow lift
- Mentoplasty (chin augmentation/reduction)

**How it works (non-technical)**

The system uses a 478-point face mesh to map the geometry of the face, \
applies anatomically informed deformations for the selected procedure, and \
then uses an AI image synthesis model (Stable Diffusion with ControlNet \
conditioning) to generate a photorealistic visualization of the deformed \
face.

There is also a fast, lightweight mode (thin-plate spline warping) that \
runs without a GPU -- this is available directly in the browser demo. \
Additionally, the tool includes a facial symmetry analysis feature that \
measures left-right facial symmetry.

**Important caveats**

This is a research prototype, not a clinical tool. The outputs are synthetic \
visualizations based on generalized procedure models, not a prediction \
tailored to any individual's anatomy or surgical plan. Real surgical outcomes \
depend on a surgeon's technique, individual anatomy, healing, and many other \
factors that this model does not capture.

**This is a research tool, not medical advice.** Please consult a \
board-certified plastic surgeon for any decisions about surgery.

That said, I hope the demo is interesting for those who want to explore how \
this kind of technology works, and I welcome feedback from anyone in the \
surgical community on how this type of system could be made more realistic \
or useful for patient communication.

Demo: {HF_SPACE_URL}
Code: {GITHUB_URL}
""",
        "flair": None,
    },
    "opensource": {
        "title": (
            "LandmarkDiff: Open-source facial surgery outcome prediction"
            " with MediaPipe + ControlNet + SD1.5"
        ),
        "body": f"""\
Releasing LandmarkDiff, an MIT-licensed system for synthesizing plausible \
post-operative facial appearances from a single clinical photograph.

**Why open source?**

Surgical planning visualization is currently locked behind proprietary \
clinical software. We believe an open research tool can accelerate work \
in this space and make the technology accessible for educational purposes.

**Architecture overview:**

- MediaPipe FaceMesh (478 3D landmarks) for facial geometry extraction
- Gaussian RBF interpolation for procedure-specific landmark deformation
- ControlNet-conditioned Stable Diffusion 1.5 (CrucibleAI backbone) for \
photorealistic synthesis
- CodeFormer + Real-ESRGAN for post-processing
- Facial symmetry analysis for quantitative left-right assessment

Six procedures are supported: rhinoplasty, blepharoplasty, rhytidectomy, \
orthognathic, brow lift, and mentoplasty.

**What's included:**

- Python package with CLI and Python API
- Gradio web demo (CPU and GPU modes)
- Docker setup (CPU and GPU Dockerfiles, docker-compose)
- Comprehensive test suite
- Documentation and wiki
- Four inference modes including a CPU-only thin-plate spline mode

The CPU TPS demo runs on HuggingFace Spaces with no GPU needed, so \
anyone can try it immediately.

GitHub: {GITHUB_URL}
Demo: {HF_SPACE_URL}
License: MIT

Contributions welcome. Disclaimer: research tool, not medical advice.
""",
        "flair": None,
    },
}

# ---------------------------------------------------------------------------
# Comment templates for helpful, genuine contributions
# ---------------------------------------------------------------------------

COMMENT_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "MachineLearning": [
        {
            "trigger": "controlnet|conditioning|spatial control",
            "comment": (
                "One thing I've found useful with ControlNet conditioning is "
                "using anatomical wireframes rather than edge maps when the "
                "task requires geometric precision. For facial applications "
                "especially, a 478-point mesh from MediaPipe gives you dense "
                "enough control that the diffusion model stays spatially "
                "anchored to the intended anatomy. We explored this in "
                f"LandmarkDiff ({GITHUB_URL}) -- the wireframe-conditioned outputs "
                "were more geometrically faithful than edge-conditioned ones."
            ),
        },
        {
            "trigger": "face|facial|landmark|mediapipe",
            "comment": (
                "On the landmark side, MediaPipe FaceMesh's 478-point 3D mesh "
                "is surprisingly dense and reliable for downstream tasks. "
                "We've been using it as a geometric backbone for facial "
                f"deformation modeling in LandmarkDiff ({GITHUB_URL}), where the "
                "landmark density is critical for making procedure-specific "
                "displacements anatomically meaningful. Gaussian RBF "
                "interpolation on that mesh produces smooth warps without the "
                "artifacts you get from sparser keypoint sets."
            ),
        },
        {
            "trigger": "medical|clinical|surgery|surgical",
            "comment": (
                "Medical imaging applications for diffusion models are really "
                "promising. One challenge we've worked on is grounding the "
                "synthesis spatially -- for surgical visualization you need "
                "the model to respect anatomical constraints, not just "
                "generate plausible textures. We addressed this with "
                "ControlNet conditioning on deformed facial landmark "
                f"wireframes. Happy to share details: {GITHUB_URL}"
            ),
        },
    ],
    "computervision": [
        {
            "trigger": "face|landmark|mesh|keypoint",
            "comment": (
                "For dense facial landmark applications, MediaPipe's 478-point "
                "3D mesh is a solid starting point. We've been combining it "
                "with Gaussian RBF interpolation for anatomically-constrained "
                f"face warping in LandmarkDiff ({GITHUB_URL}). The density of the "
                "mesh matters a lot -- sparser sets (68 or 98 points) miss "
                "important anatomical regions like the alar base and orbital "
                "rim that are critical for realistic deformation."
            ),
        },
        {
            "trigger": "warp|deformation|tps|thin.plate|rbf",
            "comment": (
                "For image warping with anatomical constraints, we compared "
                "TPS and Gaussian RBF interpolation in our facial deformation "
                "pipeline. RBF gave us better locality control -- you can "
                "deform the nasal region without dragging the eyes along. "
                "TPS is faster but the global coupling can be a problem for "
                f"localized deformations. Details in our codebase: {GITHUB_URL}"
            ),
        },
        {
            "trigger": "symmetry|asymmetry|bilateral",
            "comment": (
                "Facial symmetry analysis is a useful diagnostic tool. In "
                "LandmarkDiff we built a symmetry module that reflects "
                "landmarks across the facial midline and computes per-region "
                "asymmetry scores from the 478-point MediaPipe mesh. It's "
                "useful for quantifying before/after changes in surgical "
                f"visualization. Code is MIT licensed: {GITHUB_URL}"
            ),
        },
    ],
    "StableDiffusion": [
        {
            "trigger": "controlnet|conditioning|control",
            "comment": (
                "Interesting approach. If you want really precise spatial "
                "control, facial landmark wireframes work well as ControlNet "
                "conditioning inputs. We've been using MediaPipe's 478-point "
                "mesh rendered as a wireframe to spatially anchor diffusion "
                "to predicted facial anatomy. The model respects the geometry "
                "much better than with edge maps alone. Built this for "
                f"surgical visualization: {GITHUB_URL}"
            ),
        },
        {
            "trigger": "face|portrait|identity|ip.adapter",
            "comment": (
                "For identity preservation in facial generation, combining "
                "ControlNet with IP-Adapter gives solid results. In our "
                "LandmarkDiff pipeline we use ControlNet for spatial anatomy "
                "control (wireframe conditioning) and IP-Adapter for identity "
                "features. The combination keeps the face looking like the "
                "same person while applying geometric changes. "
                f"Code: {GITHUB_URL}"
            ),
        },
    ],
    "PlasticSurgery": [
        {
            "trigger": "visuali|predict|expectation|result|outcome",
            "comment": (
                "There are some interesting research tools being developed "
                f"for surgical outcome visualization. LandmarkDiff ({HF_SPACE_URL}) is "
                "one open-source option that can generate plausible "
                "post-operative visualizations from a single photo. "
                "Important caveat: these are research prototypes, not "
                "clinical predictions. They don't account for individual "
                "anatomy, surgeon technique, or healing. Always consult a "
                "board-certified surgeon for actual planning."
            ),
        },
    ],
    "opensource": [
        {
            "trigger": "medical|health|clinical|research",
            "comment": (
                "Open source is especially important in medical research "
                "tools where reproducibility and transparency matter. "
                f"We released LandmarkDiff ({GITHUB_URL}) under MIT for exactly "
                "this reason -- surgical planning visualization shouldn't "
                "be locked behind proprietary walls. The CPU-only mode makes "
                "it accessible without requiring expensive hardware."
            ),
        },
    ],
}

# Karma-mode comments: generic, helpful CV/ML insights (no self-promotion)
KARMA_COMMENTS: dict[str, list[dict[str, str]]] = {
    "MachineLearning": [
        {
            "trigger": "diffusion|ddpm|score.based",
            "comment": (
                "One practical tip for conditioning diffusion models on "
                "spatial inputs: rendering your control signal as a clean "
                "image (wireframes, skeletons, depth maps) and using "
                "ControlNet often outperforms trying to inject spatial "
                "information through cross-attention alone. The spatial "
                "inductive bias from the encoder copy is hard to replicate "
                "with attention-based conditioning."
            ),
        },
        {
            "trigger": "evaluation|metric|fid|lpips",
            "comment": (
                "For evaluating identity preservation in face generation, "
                "ArcFace cosine similarity is more informative than FID or "
                "LPIPS. FID captures distributional quality but misses "
                "identity-specific features. LPIPS is good for perceptual "
                "similarity but doesn't specifically measure whether two "
                "faces look like the same person."
            ),
        },
    ],
    "computervision": [
        {
            "trigger": "landmark|keypoint|detection",
            "comment": (
                "For dense facial landmark detection, the key tradeoff is "
                "between number of points and reliability. 68-point models "
                "(dlib) are rock solid but miss important regions. 478-point "
                "models (MediaPipe) cover the full face surface but can be "
                "noisier on occluded regions. For downstream tasks that need "
                "anatomical precision, the density is worth the noise."
            ),
        },
        {
            "trigger": "warp|transform|registration",
            "comment": (
                "For image warping with control point constraints, the "
                "choice between TPS and RBF interpolation matters more than "
                "people realize. TPS gives you a globally smooth warp but "
                "has non-local coupling -- moving one point can shift distant "
                "regions. RBF with a localized kernel (Gaussian, Wendland) "
                "gives you locality control via the bandwidth parameter, "
                "which is critical for localized deformations."
            ),
        },
    ],
    "StableDiffusion": [
        {
            "trigger": "quality|artifact|face|detail",
            "comment": (
                "For face quality in SD outputs, post-processing with "
                "CodeFormer followed by Real-ESRGAN makes a big difference. "
                "CodeFormer handles facial restoration (fixing eyes, teeth, "
                "skin texture) and Real-ESRGAN handles upscaling. Running "
                "them in that order avoids the common issue of upscaling "
                "facial artifacts."
            ),
        },
    ],
}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(log_file: Path) -> logging.Logger:
    """Configure file and console logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("reddit_poster")
    logger.setLevel(logging.DEBUG)

    # File handler -- verbose
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

    # Console handler -- info+
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple file-backed rate limiter."""

    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self._state: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if self.state_file.exists():
            try:
                self._state = json.loads(self.state_file.read_text())
            except (json.JSONDecodeError, OSError):
                self._state = {}

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self._state, indent=2))

    def check(self, action: str, min_interval: float) -> float:
        """Return seconds remaining before the action is allowed.

        Returns 0.0 if the action is allowed now.
        """
        last = self._state.get(action, 0.0)
        elapsed = time.time() - last
        remaining = min_interval - elapsed
        return max(0.0, remaining)

    def record(self, action: str) -> None:
        """Record that the action was performed now."""
        self._state[action] = time.time()
        self._save()


# ---------------------------------------------------------------------------
# Reddit client
# ---------------------------------------------------------------------------


def create_reddit_client(
    client_id: str,
    client_secret: str,
    password: str,
) -> praw.Reddit:
    """Create and return an authenticated PRAW Reddit instance."""
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        username=USERNAME,
        password=password,
        user_agent=USER_AGENT,
    )
    # Verify authentication
    _ = reddit.user.me()
    return reddit


# ---------------------------------------------------------------------------
# Post mode
# ---------------------------------------------------------------------------


def do_post(
    reddit: praw.Reddit,
    subreddit_name: str,
    dry_run: bool,
    rate_limiter: RateLimiter,
    logger: logging.Logger,
) -> None:
    """Submit a self-post to the specified subreddit."""
    if subreddit_name not in POST_TEMPLATES:
        logger.error(
            "No post template for r/%s. Available: %s",
            subreddit_name,
            ", ".join(sorted(POST_TEMPLATES)),
        )
        sys.exit(1)

    template = POST_TEMPLATES[subreddit_name]
    title = template["title"]
    body = template["body"]

    # Rate limit check
    action_key = f"post:{subreddit_name}"
    wait = rate_limiter.check(action_key, MIN_POST_INTERVAL_SECONDS)
    if wait > 0:
        logger.warning(
            "Rate limited. Must wait %.0f more seconds before posting to r/%s.",
            wait,
            subreddit_name,
        )
        sys.exit(1)

    if dry_run:
        logger.info("=== DRY RUN: Post to r/%s ===", subreddit_name)
        logger.info("Title: %s", title)
        logger.info("Flair: %s", template.get("flair", "None"))
        logger.info("Body:\n%s", body)
        logger.info("=== END DRY RUN ===")
        return

    subreddit = reddit.subreddit(subreddit_name)
    submission = subreddit.submit(
        title=title,
        selftext=body,
        flair_id=None,  # flair must be set manually or via flair_id lookup
    )
    rate_limiter.record(action_key)
    logger.info(
        "Posted to r/%s: %s (ID: %s)",
        subreddit_name,
        submission.url,
        submission.id,
    )


# ---------------------------------------------------------------------------
# Comment mode
# ---------------------------------------------------------------------------


def _find_matching_template(
    title: str,
    templates: list[dict[str, str]],
) -> str | None:
    """Return the first matching comment template for a post title."""
    import re

    lower_title = title.lower()
    for tmpl in templates:
        pattern = tmpl["trigger"]
        if re.search(pattern, lower_title, re.IGNORECASE):
            return tmpl["comment"]
    return None


def do_comment(
    reddit: praw.Reddit,
    subreddit_name: str,
    dry_run: bool,
    rate_limiter: RateLimiter,
    logger: logging.Logger,
    karma_mode: bool = False,
) -> None:
    """Find trending posts and leave helpful comments."""
    templates = (
        KARMA_COMMENTS.get(subreddit_name, [])
        if karma_mode
        else COMMENT_TEMPLATES.get(subreddit_name, [])
    )

    if not templates:
        logger.error(
            "No %s templates for r/%s.",
            "karma" if karma_mode else "comment",
            subreddit_name,
        )
        sys.exit(1)

    subreddit = reddit.subreddit(subreddit_name)
    hot_posts = list(subreddit.hot(limit=25))
    logger.info(
        "Scanning %d hot posts in r/%s for matching topics...",
        len(hot_posts),
        subreddit_name,
    )

    matched = 0
    for post in hot_posts:
        # Skip stickied/mod posts
        if post.stickied:
            continue

        comment_text = _find_matching_template(post.title, templates)
        if comment_text is None:
            continue

        action_key = f"comment:{post.id}"
        wait = rate_limiter.check(action_key, MIN_COMMENT_INTERVAL_SECONDS)
        if wait > 0:
            logger.info(
                "Rate limited for post %s, skipping (%.0fs remaining).",
                post.id,
                wait,
            )
            continue

        if dry_run:
            logger.info("--- DRY RUN: Comment on r/%s ---", subreddit_name)
            logger.info("Post: %s", post.title)
            logger.info("URL: %s", post.url)
            logger.info("Comment:\n%s", comment_text)
            logger.info("--- END DRY RUN ---")
        else:
            post.reply(comment_text)
            rate_limiter.record(action_key)
            logger.info(
                "Commented on '%s' in r/%s (post ID: %s)",
                post.title[:60],
                subreddit_name,
                post.id,
            )

        matched += 1

    if matched == 0:
        logger.info("No matching posts found in r/%s hot feed.", subreddit_name)
    else:
        logger.info("Processed %d matching post(s) in r/%s.", matched, subreddit_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reddit posting automation for LandmarkDiff",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --client-id ID --client-secret SECRET "
            "--password PW --mode post --subreddit MachineLearning --dry-run\n"
            "  %(prog)s --client-id ID --client-secret SECRET "
            "--password PW --mode comment --subreddit computervision\n"
            "  %(prog)s --client-id ID --client-secret SECRET "
            "--password PW --mode karma --subreddit StableDiffusion --dry-run"
        ),
    )

    # Authentication
    parser.add_argument(
        "--client-id",
        required=True,
        help="Reddit API client ID",
    )
    parser.add_argument(
        "--client-secret",
        required=True,
        help="Reddit API client secret",
    )
    parser.add_argument(
        "--password",
        required=True,
        help="Reddit account password",
    )

    # Mode and target
    parser.add_argument(
        "--mode",
        choices=["post", "comment", "karma"],
        required=True,
        help=(
            "Action mode: 'post' submits a post, 'comment' replies to "
            "trending posts with project-related comments, 'karma' leaves "
            "genuinely helpful CV/ML comments (less self-promotional)"
        ),
    )
    parser.add_argument(
        "--subreddit",
        required=True,
        choices=SUPPORTED_SUBREDDITS,
        help="Target subreddit",
    )

    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without posting",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=LOG_FILE,
        help=f"Log file path (default: {LOG_FILE})",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    logger = setup_logging(args.log_file)

    logger.info(
        "LandmarkDiff Reddit poster starting at %s",
        datetime.now(timezone.utc).isoformat(),
    )
    logger.info("Mode: %s | Subreddit: r/%s | Dry run: %s", args.mode, args.subreddit, args.dry_run)

    state_file = args.log_file.parent / "reddit_poster_state.json"
    rate_limiter = RateLimiter(state_file)

    if args.dry_run:
        logger.info("DRY RUN mode -- no Reddit API calls will be made.")
        # In dry-run we still build the client config but skip auth
        reddit = None  # type: ignore[assignment]
    else:
        try:
            reddit = create_reddit_client(
                client_id=args.client_id,
                client_secret=args.client_secret,
                password=args.password,
            )
            logger.info("Authenticated as u/%s", USERNAME)
        except Exception:
            logger.exception("Failed to authenticate with Reddit API")
            sys.exit(1)

    if args.mode == "post":
        if args.dry_run:
            # Show the post without needing auth
            template = POST_TEMPLATES.get(args.subreddit)
            if template:
                logger.info("=== DRY RUN: Post to r/%s ===", args.subreddit)
                logger.info("Title: %s", template["title"])
                logger.info("Flair: %s", template.get("flair", "None"))
                logger.info("Body:\n%s", template["body"])
                logger.info("=== END DRY RUN ===")
            else:
                logger.error("No template for r/%s", args.subreddit)
                sys.exit(1)
        else:
            do_post(reddit, args.subreddit, False, rate_limiter, logger)

    elif args.mode == "comment":
        if args.dry_run:
            templates = COMMENT_TEMPLATES.get(args.subreddit, [])
            logger.info(
                "=== DRY RUN: %d comment templates for r/%s ===",
                len(templates),
                args.subreddit,
            )
            for i, tmpl in enumerate(templates, 1):
                logger.info(
                    "Template %d (trigger: %s):\n%s",
                    i,
                    tmpl["trigger"],
                    tmpl["comment"],
                )
            logger.info("=== END DRY RUN ===")
        else:
            do_comment(reddit, args.subreddit, False, rate_limiter, logger)

    elif args.mode == "karma":
        if args.dry_run:
            templates = KARMA_COMMENTS.get(args.subreddit, [])
            logger.info(
                "=== DRY RUN: %d karma templates for r/%s ===",
                len(templates),
                args.subreddit,
            )
            for i, tmpl in enumerate(templates, 1):
                logger.info(
                    "Template %d (trigger: %s):\n%s",
                    i,
                    tmpl["trigger"],
                    tmpl["comment"],
                )
            logger.info("=== END DRY RUN ===")
        else:
            do_comment(
                reddit,
                args.subreddit,
                False,
                rate_limiter,
                logger,
                karma_mode=True,
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
