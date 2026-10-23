"""
GameForge AI — Pixel-Art Resampling & Alpha Integrity Module
Preserves discrete pixel-art palettes, strict nearest-neighbor sampling,
and clean transparent alpha borders without bilinear color bleeding.
"""

import numpy as np
from PIL import Image
from typing import Tuple
from .schemas import ResamplingPolicy


def clean_alpha_channel(rgba_array: np.ndarray, threshold: int = 20) -> np.ndarray:
    """
    Snaps small fractional alpha bleeding values below the threshold to 0,
    maintaining crisp boundaries for retro sprite art.
    """
    cleaned = rgba_array.copy()
    alpha_ch = cleaned[:, :, 3]
    alpha_ch[alpha_ch < threshold] = 0
    cleaned[:, :, 3] = alpha_ch
    return cleaned


def resample_sprite_grid(
    src_array: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    width: int,
    height: int,
    resampling: ResamplingPolicy = ResamplingPolicy.NEAREST,
    preserve_pixel_art: bool = True
) -> np.ndarray:
    """
    Samples pixel values from src_array using backward coordinate mapping (src_x, src_y).
    When preserve_pixel_art is True or resampling is NEAREST, enforces discrete integer grid snapping.
    """
    out_array = np.zeros((height, width, 4), dtype=np.uint8)

    if resampling == ResamplingPolicy.NEAREST or preserve_pixel_art:
        # Discrete integer grid snapping (Strict Pixel-Art Preservation)
        near_x = np.rint(src_x).astype(np.int32)
        near_y = np.rint(src_y).astype(np.int32)

        valid_mask = (near_x >= 0) & (near_x < width) & (near_y >= 0) & (near_y < height)
        out_array[valid_mask] = src_array[near_y[valid_mask], near_x[valid_mask]]

        # Alpha thresholding
        out_array = clean_alpha_channel(out_array, threshold=20)

    else:
        # Bilinear interpolation
        x0 = np.floor(src_x).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(src_y).astype(np.int32)
        y1 = y0 + 1

        wa = (x1 - src_x) * (y1 - src_y)
        wb = (src_x - x0) * (y1 - src_y)
        wc = (x1 - src_x) * (src_y - y0)
        wd = (src_x - x0) * (src_y - y0)

        valid_mask = (x0 >= 0) & (x1 < width) & (y0 >= 0) & (y1 < height)

        for c in range(4):
            ch_out = np.zeros((height, width), dtype=np.float32)
            ch_out[valid_mask] = (
                wa[valid_mask] * src_array[y0[valid_mask], x0[valid_mask], c] +
                wb[valid_mask] * src_array[y0[valid_mask], x1[valid_mask], c] +
                wc[valid_mask] * src_array[y1[valid_mask], x0[valid_mask], c] +
                wd[valid_mask] * src_array[y1[valid_mask], x1[valid_mask], c]
            )
            out_array[:, :, c] = np.clip(ch_out, 0, 255).astype(np.uint8)

    return out_array
