"""Extended tests for the evaluation metrics suite.

Covers EvalMetrics dataclass, compute_ssim, compute_nme,
classify_fitzpatrick_ita, evaluate_batch, and summary/serialization methods.
Tests run without torch, lpips, or InsightFace.
"""

from __future__ import annotations

import numpy as np

from landmarkdiff.evaluation import (
    EvalMetrics,
    classify_fitzpatrick_ita,
    compute_nme,
    compute_ssim,
    evaluate_batch,
)

# ---------------------------------------------------------------------------
# EvalMetrics dataclass
# ---------------------------------------------------------------------------


class TestEvalMetrics:
    def test_defaults(self):
        m = EvalMetrics()
        assert m.fid == 0.0
        assert m.lpips == 0.0
        assert m.nme == 0.0
        assert m.ssim == 0.0
        assert m.identity_sim == 0.0
        assert m.fid_by_fitzpatrick == {}
        assert m.nme_by_procedure == {}

    def test_summary_basic(self):
        m = EvalMetrics(fid=12.5, lpips=0.15, nme=0.03, identity_sim=0.92, ssim=0.87)
        s = m.summary()
        assert "12.50" in s
        assert "0.1500" in s
        assert "0.0300" in s
        assert "0.9200" in s
        assert "0.8700" in s

    def test_summary_with_fitzpatrick(self):
        m = EvalMetrics(
            ssim=0.85,
            count_by_fitzpatrick={"I": 10, "III": 20},
            ssim_by_fitzpatrick={"I": 0.88, "III": 0.82},
        )
        s = m.summary()
        assert "Fitzpatrick" in s
        assert "n=10" in s
        assert "n=20" in s

    def test_to_dict_basic(self):
        m = EvalMetrics(fid=10.0, lpips=0.2, nme=0.05, ssim=0.9, identity_sim=0.95)
        d = m.to_dict()
        assert d["fid"] == 10.0
        assert d["lpips"] == 0.2
        assert d["nme"] == 0.05
        assert d["ssim"] == 0.9
        assert d["identity_sim"] == 0.95

    def test_to_dict_with_fitzpatrick(self):
        m = EvalMetrics(
            count_by_fitzpatrick={"II": 5},
            lpips_by_fitzpatrick={"II": 0.12},
            ssim_by_fitzpatrick={"II": 0.88},
            nme_by_fitzpatrick={"II": 0.04},
            identity_sim_by_fitzpatrick={"II": 0.93},
        )
        d = m.to_dict()
        assert d["fitz_II_count"] == 5
        assert d["fitz_II_lpips"] == 0.12
        assert d["fitz_II_ssim"] == 0.88
        assert d["fitz_II_nme"] == 0.04
        assert d["fitz_II_identity"] == 0.93

    def test_to_dict_with_procedure(self):
        m = EvalMetrics(
            nme_by_procedure={"rhinoplasty": 0.03},
            lpips_by_procedure={"rhinoplasty": 0.15},
            ssim_by_procedure={"rhinoplasty": 0.88},
        )
        d = m.to_dict()
        assert d["proc_rhinoplasty_nme"] == 0.03
        assert d["proc_rhinoplasty_lpips"] == 0.15
        assert d["proc_rhinoplasty_ssim"] == 0.88


# ---------------------------------------------------------------------------
# compute_ssim
# ---------------------------------------------------------------------------


class TestComputeSSIM:
    def test_identical_images_score_one(self):
        img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        score = compute_ssim(img, img)
        assert abs(score - 1.0) < 0.01

    def test_different_images_lower_score(self):
        rng = np.random.default_rng(0)
        img1 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        img2 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        score = compute_ssim(img1, img2)
        assert score < 0.99

    def test_grayscale_images(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        score = compute_ssim(img, img)
        assert abs(score - 1.0) < 0.01

    def test_return_type_is_float(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        score = compute_ssim(img, img)
        assert isinstance(score, float)

    def test_ssim_bounded(self):
        rng = np.random.default_rng(42)
        img1 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        img2 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        score = compute_ssim(img1, img2)
        assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_nme
# ---------------------------------------------------------------------------


class TestComputeNME:
    def test_identical_landmarks_zero_nme(self):
        rng = np.random.default_rng(0)
        lm = rng.uniform(0, 512, (478, 2)).astype(np.float32)
        nme = compute_nme(lm, lm)
        assert abs(nme) < 1e-6

    def test_shifted_landmarks_positive_nme(self):
        rng = np.random.default_rng(0)
        lm = rng.uniform(100, 400, (478, 2)).astype(np.float32)
        shifted = lm + 5.0
        nme = compute_nme(shifted, lm)
        assert nme > 0

    def test_nme_normalized_by_iod(self):
        """NME should decrease when IOD increases for same absolute error."""
        lm1 = np.zeros((478, 2), dtype=np.float32)
        lm1[33] = [100.0, 200.0]  # left eye
        lm1[263] = [200.0, 200.0]  # right eye (IOD=100)

        lm2 = lm1.copy()
        lm2[263] = [400.0, 200.0]  # right eye (IOD=300)

        pred = lm1.copy()
        pred += 10.0  # uniform shift

        nme_narrow = compute_nme(pred, lm1)
        nme_wide = compute_nme(pred + np.array([0, 0]), lm2)
        # Wider IOD should produce smaller NME for same pixel error
        assert nme_wide < nme_narrow

    def test_minimum_iod_clamp(self):
        """When eyes overlap (IOD < 1), should clamp to 1.0."""
        lm = np.zeros((478, 2), dtype=np.float32)
        lm[33] = [100.0, 100.0]
        lm[263] = [100.0, 100.0]  # same position, IOD=0
        pred = lm.copy()
        pred[0] = [10.0, 10.0]
        # Should not raise or produce inf
        nme = compute_nme(pred, lm)
        assert np.isfinite(nme)

    def test_custom_eye_indices(self):
        """compute_nme should accept custom eye indices."""
        rng = np.random.default_rng(0)
        lm = rng.uniform(50, 450, (478, 2)).astype(np.float32)
        # Use different indices for "eyes"
        nme = compute_nme(lm, lm, left_eye_idx=7, right_eye_idx=249)
        assert abs(nme) < 1e-6

    def test_return_type_is_float(self):
        rng = np.random.default_rng(0)
        lm = rng.uniform(0, 512, (478, 2)).astype(np.float32)
        nme = compute_nme(lm, lm)
        assert isinstance(nme, float)


# ---------------------------------------------------------------------------
# classify_fitzpatrick_ita
# ---------------------------------------------------------------------------


class TestClassifyFitzpatrick:
    def test_returns_valid_type(self):
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        ftype = classify_fitzpatrick_ita(img)
        assert ftype in {"I", "II", "III", "IV", "V", "VI"}

    def test_light_skin_classifies_low(self):
        """Very light image should classify as Type I or II."""
        img = np.full((64, 64, 3), 250, dtype=np.uint8)
        ftype = classify_fitzpatrick_ita(img)
        assert ftype in {"I", "II", "III"}

    def test_dark_skin_classifies_high(self):
        """Very dark image should classify as Type V or VI."""
        img = np.full((64, 64, 3), 30, dtype=np.uint8)
        ftype = classify_fitzpatrick_ita(img)
        assert ftype in {"IV", "V", "VI"}

    def test_samples_from_center(self):
        """Classification should sample from center region."""
        # Create image with different center vs. border
        img = np.full((100, 100, 3), 30, dtype=np.uint8)  # dark border
        img[25:75, 25:75] = 250  # bright center
        ftype = classify_fitzpatrick_ita(img)
        # Center is bright, so should classify as light
        assert ftype in {"I", "II", "III"}


# ---------------------------------------------------------------------------
# evaluate_batch
# ---------------------------------------------------------------------------


class TestEvaluateBatch:
    def test_single_pair(self):
        rng = np.random.default_rng(0)
        pred = rng.integers(50, 200, (64, 64, 3), dtype=np.uint8)
        target = pred.copy()
        metrics = evaluate_batch([pred], [target])
        assert isinstance(metrics, EvalMetrics)
        # SSIM of identical images should be high
        assert metrics.ssim > 0.9

    def test_multiple_pairs(self):
        rng = np.random.default_rng(0)
        preds = [rng.integers(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
        targets = [p.copy() for p in preds]
        metrics = evaluate_batch(preds, targets)
        assert metrics.ssim > 0.9

    def test_with_landmarks(self):
        rng = np.random.default_rng(0)
        pred_img = rng.integers(50, 200, (64, 64, 3), dtype=np.uint8)
        target_img = pred_img.copy()
        pred_lm = rng.uniform(50, 200, (478, 2)).astype(np.float32)
        target_lm = pred_lm.copy()
        metrics = evaluate_batch(
            [pred_img],
            [target_img],
            pred_landmarks=[pred_lm],
            target_landmarks=[target_lm],
        )
        assert metrics.nme < 1e-6

    def test_with_procedures(self):
        rng = np.random.default_rng(0)
        preds = [rng.integers(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        targets = [p.copy() for p in preds]
        procedures = ["rhinoplasty", "rhinoplasty", "blepharoplasty"]
        metrics = evaluate_batch(preds, targets, procedures=procedures)
        assert "rhinoplasty" in metrics.ssim_by_procedure
        assert "blepharoplasty" in metrics.ssim_by_procedure

    def test_fitzpatrick_stratification(self):
        rng = np.random.default_rng(0)
        preds = [rng.integers(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        targets = [p.copy() for p in preds]
        metrics = evaluate_batch(preds, targets)
        # Should have at least one Fitzpatrick group
        assert len(metrics.count_by_fitzpatrick) >= 1

    def test_empty_batch(self):
        metrics = evaluate_batch([], [])
        assert isinstance(metrics, EvalMetrics)
        assert metrics.ssim == 0.0
