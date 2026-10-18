"""
Lesion masking / skin-tone-bias reduction.

Isolates the lesion region in a dermoscopic image so that downstream feature
extraction focuses on lesion-specific visual patterns rather than surrounding
skin tone. Two color-space strategies are implemented, matching the two
approaches compared in the project report (Section VII.3):

    * HSV thresholding
    * CIELAB (L*a*b*) thresholding

Both are followed by simple morphological cleanup (closing + largest-contour
selection) to produce a clean binary lesion mask.

Usage
-----
    python lesion_masking.py --input_dir data/raw --output_dir data/masked \
        --method hsv
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def _largest_component_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return binary_mask

    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(binary_mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean


def mask_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """Segment the lesion using HSV saturation/value thresholds + Otsu."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    _, mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return _largest_component_mask(mask)


def mask_cielab(image_bgr: np.ndarray) -> np.ndarray:
    """Segment the lesion using the CIELAB a*-channel + Otsu."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]

    _, mask = cv2.threshold(
        a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return _largest_component_mask(mask)


MASKERS = {"hsv": mask_hsv, "cielab": mask_cielab}


def apply_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out everything outside the lesion mask (background -> black)."""
    return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)


def process_directory(input_dir: str, output_dir: str, method: str) -> None:
    masker = MASKERS[method]
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in Path(input_dir).rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️  Skipping unreadable file: {img_path}")
            continue

        mask = masker(image)
        lesion_only = apply_mask(image, mask)

        cv2.imwrite(str(out_path / img_path.name), lesion_only)

    print(f"✅ Processed {len(image_paths)} images with '{method}' masking -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Lesion masking preprocessing")
    parser.add_argument("--input_dir", required=True, help="Directory of raw dermoscopic images")
    parser.add_argument("--output_dir", required=True, help="Where to write masked images")
    parser.add_argument(
        "--method", choices=list(MASKERS.keys()), default="hsv",
        help="Color-space masking strategy (default: hsv)"
    )
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.method)


if __name__ == "__main__":
    main()
