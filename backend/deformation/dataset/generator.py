"""
GameForge AI — Synthetic Paired-Pose Dataset Generator
Stage 3 implementation: Produces reproducible, identity-partitioned paired pose datasets
(I_src, S_src, S_tgt) -> I_tgt using the deterministic geometric baseline deformer.
"""

import os
import json
import time
import random
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional

from deformation.schemas import (
    Skeleton2D,
    Joint2D,
    BoundingBox,
    DeformationRequest,
    DeformationMethod,
    ResamplingPolicy
)
from deformation.skeleton import (
    get_default_skeleton,
    get_pose_templates,
    get_template_by_name,
    fit_skeleton_to_bbox,
    JOINT_NAMES
)
from deformation.geometry import GeometricDeformer


class SyntheticDatasetGenerator:
    """
    Generates synthetic paired character pose datasets for neural deformation experiments.
    Strictly enforces zero character identity leakage across train, val, and test splits.
    """

    def __init__(
        self,
        samples_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        seed: int = 42
    ):
        self.seed = seed
        self.deformer = GeometricDeformer()

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.samples_dir = samples_dir or os.path.join(base_dir, "alucard_samples")
        self.output_dir = output_dir or os.path.join(base_dir, "data", "synthetic_poses")

    def load_available_characters(self) -> List[Dict]:
        """Loads available character sprite metadata from alucard_samples/manifest.json."""
        manifest_path = os.path.join(self.samples_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Samples manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        # Verify images exist on disk
        valid_characters = []
        for s in samples:
            img_path = os.path.join(self.samples_dir, s["filename"])
            if os.path.exists(img_path):
                valid_characters.append({
                    "character_id": s["sample_id"],
                    "label": s.get("label", f"Character_{s['sample_id']}"),
                    "filename": s["filename"],
                    "filepath": img_path
                })
        return valid_characters

    def partition_character_splits(
        self,
        characters: List[Dict],
        train_ratio: float = 0.60,
        val_ratio: float = 0.20
    ) -> Dict[str, List[Dict]]:
        """
        Partitions characters strictly by character identity to prevent near-duplicate leakage.
        Default: 60% Train, 20% Val, 20% Test.
        """
        rng = random.Random(self.seed)
        shuffled = list(characters)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_chars = shuffled[:n_train]
        val_chars = shuffled[n_train:n_train + n_val]
        test_chars = shuffled[n_train + n_val:]

        return {
            "train": sorted(train_chars, key=lambda x: x["character_id"]),
            "val": sorted(val_chars, key=lambda x: x["character_id"]),
            "test": sorted(test_chars, key=lambda x: x["character_id"])
        }

    def generate_dataset(
        self,
        target_pose_names: Optional[List[str]] = None,
        max_samples_per_split: Optional[int] = None
    ) -> Dict:
        """
        Executes reproducible synthetic dataset generation.
        Renders target poses for each character using GeometricDeformer, saving PNGs and manifest.
        """
        start_time = time.perf_counter()
        target_poses = target_pose_names or ["walk_1", "walk_2", "run", "jump", "attack_slash"]

        characters = self.load_available_characters()
        splits = self.partition_character_splits(characters)

        os.makedirs(self.output_dir, exist_ok=True)

        manifest_entries = []
        split_counts = {"train": 0, "val": 0, "test": 0}
        failure_count = 0
        total_bytes_written = 0

        sample_counter = 0

        for split_name, char_list in splits.items():
            split_dir = os.path.join(self.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            for char_info in char_list:
                src_path = char_info["filepath"]
                src_img = Image.open(src_path).convert("RGBA")

                # Detect bounding box and fit source skeleton
                bbox = self.deformer.detect_bounding_box(src_img)
                if bbox.non_transparent_pixels == 0:
                    failure_count += 1
                    continue

                default_skel = get_default_skeleton()
                src_skel = fit_skeleton_to_bbox(default_skel, bbox, src_img.width, src_img.height)

                for pose_name in target_poses:
                    if max_samples_per_split and split_counts[split_name] >= max_samples_per_split:
                        break

                    tpl = get_template_by_name(pose_name)
                    if not tpl:
                        failure_count += 1
                        continue

                    tgt_skel = fit_skeleton_to_bbox(tpl.skeleton, bbox, src_img.width, src_img.height)

                    # Render target sprite using geometric baseline
                    tgt_img, val_result = self.deformer.deform(
                        image=src_img,
                        source_skeleton=src_skel,
                        target_skeleton=tgt_skel,
                        intensity=1.0,
                        resampling=ResamplingPolicy.NEAREST,
                        preserve_pixel_art=True
                    )

                    sample_id = f"{split_name}_{sample_counter:04d}_c{char_info['character_id']}_{pose_name}"
                    src_out_filename = f"{sample_id}_src.png"
                    tgt_out_filename = f"{sample_id}_tgt.png"

                    src_out_path = os.path.join(split_dir, src_out_filename)
                    tgt_out_path = os.path.join(split_dir, tgt_out_filename)

                    src_img.save(src_out_path, format="PNG")
                    tgt_img.save(tgt_out_path, format="PNG")

                    bytes_sample = os.path.getsize(src_out_path) + os.path.getsize(tgt_out_path)
                    total_bytes_written += bytes_sample

                    entry = {
                        "sample_id": sample_id,
                        "split": split_name,
                        "character_id": char_info["character_id"],
                        "character_label": char_info["label"],
                        "target_pose_name": pose_name,
                        "source_image_path": os.path.relpath(src_out_path, self.output_dir).replace("\\", "/"),
                        "target_image_path": os.path.relpath(tgt_out_path, self.output_dir).replace("\\", "/"),
                        "source_skeleton": src_skel.model_dump(),
                        "target_skeleton": tgt_skel.model_dump(),
                        "validation": val_result.model_dump()
                    }

                    manifest_entries.append(entry)
                    split_counts[split_name] += 1
                    sample_counter += 1

        duration_sec = round(time.perf_counter() - start_time, 2)
        summary = {
            "dataset_name": "GameForge_Synthetic_Pose_Pairs_v1",
            "seed": self.seed,
            "creation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(manifest_entries),
            "split_counts": split_counts,
            "character_split_ids": {
                split: [c["character_id"] for c in char_list]
                for split, char_list in splits.items()
            },
            "failures": failure_count,
            "total_bytes_written": total_bytes_written,
            "generation_duration_sec": duration_sec,
            "target_poses_used": target_poses
        }

        # Write manifest
        manifest_payload = {
            "summary": summary,
            "samples": manifest_entries
        }
        manifest_file = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)

        return summary


dataset_generator = SyntheticDatasetGenerator()
