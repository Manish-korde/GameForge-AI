"""
GameForge AI — Deformation Evaluation & Comparative Benchmark Suite
Stage 3 implementation: Evaluates deformation models (Geometric Baseline, Neural STN,
and Hybrid VAE) across standardized metrics on held-out test character identities.
"""

import os
import json
import time
import math
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any


def compute_photometric_metrics(pred_arr: np.ndarray, gt_arr: np.ndarray) -> Dict[str, float]:
    """
    Computes standard image reconstruction metrics between predicted and ground-truth RGBA arrays:
    - L1 Error (mean absolute error normalized to [0.0, 1.0])
    - Mean Squared Error (MSE)
    - Peak Signal-to-Noise Ratio (PSNR in dB)
    """
    pred_f = pred_arr.astype(np.float32) / 255.0
    gt_f = gt_arr.astype(np.float32) / 255.0

    l1_err = float(np.mean(np.abs(pred_f - gt_f)))
    mse_err = float(np.mean((pred_f - gt_f) ** 2))

    if mse_err < 1e-10:
        psnr = 100.0
    else:
        psnr = float(10.0 * math.log10(1.0 / mse_err))

    return {
        "l1_error": round(l1_err, 5),
        "mse": round(mse_err, 6),
        "psnr_db": round(psnr, 2)
    }


def compute_silhouette_metrics(
    pred_arr: np.ndarray,
    gt_arr: np.ndarray,
    alpha_threshold: int = 15
) -> Dict[str, Any]:
    """
    Computes silhouette overlap IoU, alpha pixel mass retention, and canvas clipping.
    """
    pred_alpha = pred_arr[:, :, 3] > alpha_threshold
    gt_alpha = gt_arr[:, :, 3] > alpha_threshold

    intersection = np.sum(pred_alpha & gt_alpha)
    union = np.sum(pred_alpha | gt_alpha)
    iou = float(intersection / union) if union > 0 else 1.0

    gt_count = int(np.sum(gt_alpha))
    pred_count = int(np.sum(pred_alpha))
    alpha_retention = (pred_count / gt_count * 100.0) if gt_count > 0 else 100.0
    alpha_retention = min(100.0, max(0.0, alpha_retention))

    h, w = pred_arr.shape[:2]
    clipping = bool(
        np.any(pred_alpha[0, :]) or
        np.any(pred_alpha[h - 1, :]) or
        np.any(pred_alpha[:, 0]) or
        np.any(pred_alpha[:, w - 1])
    )

    return {
        "target_overlap_iou": round(iou, 4),
        "alpha_integrity_pct": round(alpha_retention, 2),
        "clipping_detected": clipping
    }


class DeformationBenchmarkRunner:
    """
    Runs systematic evaluation of deformation models on the held-out test split.
    """

    def __init__(self, dataset_dir: Optional[str] = None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.dataset_dir = dataset_dir or os.path.join(base_dir, "data", "synthetic_poses")
        self.manifest_path = os.path.join(self.dataset_dir, "manifest.json")

    def load_test_samples(self) -> List[Dict]:
        """Loads entries belonging to the 'test' split from manifest.json."""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Dataset manifest not found at {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        test_samples = [s for s in manifest.get("samples", []) if s.get("split") == "test"]
        return test_samples

    def evaluate_geometric_baseline(self) -> Dict[str, Any]:
        """
        Evaluates the Geometric Baseline deformer on all held-out test character samples.
        Note: The synthetic targets were rendered by the geometric deformer, so geometric
        reconstruction metrics establish the reference ceiling (self-consistency check),
        while IoU, alpha integrity, clipping, and latency reflect operational characteristics.
        """
        from deformation.geometry import GeometricDeformer
        from deformation.schemas import Skeleton2D, ResamplingPolicy

        deformer = GeometricDeformer()
        test_samples = self.load_test_samples()

        if not test_samples:
            raise ValueError("No test samples found in manifest.json")

        latencies = []
        target_ious = []
        alpha_integrities = []
        clipping_flags = []
        l1_errors = []
        psnr_values = []

        for sample in test_samples:
            src_full_path = os.path.join(self.dataset_dir, sample["source_image_path"])
            gt_full_path = os.path.join(self.dataset_dir, sample["target_image_path"])

            src_img = Image.open(src_full_path).convert("RGBA")
            gt_img = Image.open(gt_full_path).convert("RGBA")
            gt_arr = np.array(gt_img)

            src_skel = Skeleton2D(**sample["source_skeleton"])
            tgt_skel = Skeleton2D(**sample["target_skeleton"])

            t0 = time.perf_counter()
            pred_img, val_result = deformer.deform(
                image=src_img,
                source_skeleton=src_skel,
                target_skeleton=tgt_skel,
                intensity=1.0,
                resampling=ResamplingPolicy.NEAREST,
                preserve_pixel_art=True
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            pred_arr = np.array(pred_img)

            photometric = compute_photometric_metrics(pred_arr, gt_arr)
            silhouette = compute_silhouette_metrics(pred_arr, gt_arr)

            target_ious.append(silhouette["target_overlap_iou"])
            alpha_integrities.append(silhouette["alpha_integrity_pct"])
            clipping_flags.append(silhouette["clipping_detected"])
            l1_errors.append(photometric["l1_error"])
            psnr_values.append(photometric["psnr_db"])

        return {
            "model_name": "Geometric Baseline (LBS)",
            "test_sample_count": len(test_samples),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "mean_target_iou": round(float(np.mean(target_ious)), 4),
            "mean_alpha_integrity_pct": round(float(np.mean(alpha_integrities)), 2),
            "clipping_rate_pct": round(float(np.mean(clipping_flags) * 100.0), 2),
            "mean_l1_error": round(float(np.mean(l1_errors)), 5),
            "mean_psnr_db": round(float(np.mean(psnr_values)), 2),
            "failure_rate_pct": 0.0
        }

    def evaluate_model_pipeline(self, model_infer_func, model_name: str) -> Dict[str, Any]:
        """
        Generic evaluator comparing any inference function against the held-out test ground truth.
        """
        from deformation.schemas import Skeleton2D

        test_samples = self.load_test_samples()
        latencies = []
        target_ious = []
        alpha_integrities = []
        clipping_flags = []
        l1_errors = []
        psnr_values = []
        failures = 0

        for sample in test_samples:
            src_full_path = os.path.join(self.dataset_dir, sample["source_image_path"])
            gt_full_path = os.path.join(self.dataset_dir, sample["target_image_path"])

            src_img = Image.open(src_full_path).convert("RGBA")
            gt_img = Image.open(gt_full_path).convert("RGBA")
            gt_arr = np.array(gt_img)

            src_skel = Skeleton2D(**sample["source_skeleton"])
            tgt_skel = Skeleton2D(**sample["target_skeleton"])

            t0 = time.perf_counter()
            try:
                pred_img = model_infer_func(src_img, src_skel, tgt_skel)
            except Exception as e:
                failures += 1
                continue
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            pred_arr = np.array(pred_img)
            photometric = compute_photometric_metrics(pred_arr, gt_arr)
            silhouette = compute_silhouette_metrics(pred_arr, gt_arr)

            target_ious.append(silhouette["target_overlap_iou"])
            alpha_integrities.append(silhouette["alpha_integrity_pct"])
            clipping_flags.append(silhouette["clipping_detected"])
            l1_errors.append(photometric["l1_error"])
            psnr_values.append(photometric["psnr_db"])

        return {
            "model_name": model_name,
            "test_sample_count": len(test_samples),
            "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "mean_target_iou": round(float(np.mean(target_ious)), 4) if target_ious else 0.0,
            "mean_alpha_integrity_pct": round(float(np.mean(alpha_integrities)), 2) if alpha_integrities else 0.0,
            "clipping_rate_pct": round(float(np.mean(clipping_flags) * 100.0), 2) if clipping_flags else 0.0,
            "mean_l1_error": round(float(np.mean(l1_errors)), 5) if l1_errors else 1.0,
            "mean_psnr_db": round(float(np.mean(psnr_values)), 2) if psnr_values else 0.0,
            "failure_rate_pct": round(float(failures / len(test_samples) * 100.0), 2)
        }


benchmark_runner = DeformationBenchmarkRunner()
