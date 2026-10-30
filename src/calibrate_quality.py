"""
Module 1: Image Quality Metric Extraction and Statistical Calibration
Branch: palak-quality
File: src/calibrate_quality.py

Provides classical OpenCV and NumPy functions to evaluate image quality
(blur, illumination, glare, obstruction/hair/vignette) and a calibration
routine to establish empirical WARN and FAIL boundaries from training data.
"""

import os
import sys
import glob
import random
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2


# ============================================================================
# 1. STANDALONE METRIC EXTRACTION FUNCTIONS
# ============================================================================

def measure_blur(img: np.ndarray) -> float:
    """
    Computes the blur metric using the variance of the Laplacian operator on grayscale.
    
    A higher variance indicates sharp edges and fine details.
    A lower variance indicates motion blur, out-of-focus blur, or optical degradation.

    Parameters
    ----------
    img : np.ndarray
        Input image as a NumPy array (BGR, RGB, or single-channel Grayscale).

    Returns
    -------
    float
        Variance of the Laplacian (higher = sharper, lower = blurrier).
    """
    if img is None or img.size == 0:
        raise ValueError("Invalid or empty image provided to measure_blur.")

    # Convert to grayscale if image has color channels
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img[:, :, 0]
    else:
        gray = img

    # Compute Laplacian with 64-bit float precision to avoid overflow
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return variance


def measure_illumination(img: np.ndarray) -> Dict[str, float]:
    """
    Measures illumination characteristics:
      1. Mean brightness across the luminance channel (or grayscale).
      2. Fraction of underexposed pixels (luminance < 30).
      3. Fraction of overexposed pixels (luminance > 225).

    Parameters
    ----------
    img : np.ndarray
        Input image as a NumPy array (BGR, RGB, or single-channel Grayscale).

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
          - 'mean_brightness': Average luminance in range [0.0, 255.0]
          - 'underexposed_fraction': Fraction of dark pixels in range [0.0, 1.0]
          - 'overexposed_fraction': Fraction of clipped bright pixels in range [0.0, 1.0]
    """
    if img is None or img.size == 0:
        raise ValueError("Invalid or empty image provided to measure_illumination.")

    # Extract luminance / brightness channel (Y channel in YCrCb or Grayscale)
    if len(img.shape) == 3:
        # Grayscale / luminance transformation (BT.601 standard)
        if img.shape[2] == 4:
            y_channel = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            # Y channel from YCrCb provides standard perceptual luminance
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y_channel = ycrcb[:, :, 0]
        else:
            y_channel = img[:, :, 0]
    else:
        y_channel = img

    total_pixels = float(y_channel.size)
    mean_brightness = float(np.mean(y_channel))
    underexposed_fraction = float(np.count_nonzero(y_channel < 30) / total_pixels)
    overexposed_fraction = float(np.count_nonzero(y_channel > 225) / total_pixels)

    return {
        "mean_brightness": mean_brightness,
        "underexposed_fraction": underexposed_fraction,
        "overexposed_fraction": overexposed_fraction,
    }


def measure_glare(img: np.ndarray) -> float:
    """
    Measures specular glare and reflection as the percentage of pixels that
    exhibit high brightness (Value > 230) and low saturation (Saturation < 30)
    in HSV color space.

    Parameters
    ----------
    img : np.ndarray
        Input image as a NumPy array (BGR or RGB).

    Returns
    -------
    float
        Percentage of glare pixels in range [0.0, 100.0].
    """
    if img is None or img.size == 0:
        raise ValueError("Invalid or empty image provided to measure_glare.")

    # Single-channel grayscale fallback
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        # In pure grayscale, saturation is 0, so glare corresponds to V > 230
        gray = img if len(img.shape) == 2 else img[:, :, 0]
        glare_pct = float(np.mean(gray > 230) * 100.0)
        return glare_pct

    # BGR to HSV conversion
    # Note: OpenCV HSV ranges: H in [0, 179], S in [0, 255], V in [0, 255]
    if img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        bgr = img

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Glare condition: high brightness and washed-out/desaturated color
    glare_mask = (val > 230) & (sat < 30)
    glare_pct = float(np.mean(glare_mask) * 100.0)
    return glare_pct


def measure_obstruction(
    img: np.ndarray,
    corner_box_ratio: float = 0.15,
    corner_dark_thresh: int = 30,
    blackhat_kernel_size: int = 17,
    hair_thresh: int = 15,
) -> Dict[str, float]:
    """
    Measures optical and physical obstructions in the field of view:
      a) Dark corner/vignette coverage: Fraction of dark pixels in corner bounding boxes.
      b) Hair mask coverage: Surface coverage of hair fibers segmented via
         morphological Black-Hat filtering with an elliptical structuring element.

    Parameters
    ----------
    img : np.ndarray
        Input image as a NumPy array (BGR, RGB, or single-channel Grayscale).
    corner_box_ratio : float, default=0.15
        Proportion of image width and height allocated to each corner box (e.g. 15%).
    corner_dark_thresh : int, default=30
        Pixel intensity threshold below which a corner pixel is deemed dark/vignetted.
    blackhat_kernel_size : int, default=17
        Structuring element size for morphological black-hat filter.
    hair_thresh : int, default=15
        Intensity threshold on black-hat response for hair fiber segmentation.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
          - 'vignette_coverage': Fraction of dark pixels across the 4 corners [0.0, 1.0]
          - 'hair_coverage': Fraction of image surface covered by hair [0.0, 1.0]
          - 'total_obstruction': Combined obstruction estimate [0.0, 1.0]
    """
    if img is None or img.size == 0:
        raise ValueError("Invalid or empty image provided to measure_obstruction.")

    h, w = img.shape[:2]

    # Convert to grayscale for geometric and morphological analysis
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img[:, :, 0]
    else:
        gray = img

    # ------------------------------------------------------------------------
    # a) Dark Corner / Vignette Coverage
    # ------------------------------------------------------------------------
    cw = max(1, int(w * corner_box_ratio))
    ch = max(1, int(h * corner_box_ratio))

    # Define 4 corner regions
    tl = gray[0:ch, 0:cw]
    tr = gray[0:ch, w - cw:w]
    bl = gray[h - ch:h, 0:cw]
    br = gray[h - ch:h, w - cw:w]

    total_corner_pixels = float(cw * ch * 4)
    dark_corner_count = (
        np.count_nonzero(tl < corner_dark_thresh)
        + np.count_nonzero(tr < corner_dark_thresh)
        + np.count_nonzero(bl < corner_dark_thresh)
        + np.count_nonzero(br < corner_dark_thresh)
    )
    vignette_coverage = float(dark_corner_count / total_corner_pixels)

    # ------------------------------------------------------------------------
    # b) Hair Mask Coverage using Morphological Black-Hat Filter
    # ------------------------------------------------------------------------
    # Hair appears as thin, dark linear structures against a brighter background.
    # The Black-Hat operation: Closing(I) - I isolates dark structures smaller
    # than the structuring element.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (blackhat_kernel_size, blackhat_kernel_size)
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold the black-hat filter output to create binary hair mask
    _, hair_mask = cv2.threshold(blackhat, hair_thresh, 255, cv2.THRESH_BINARY)
    hair_coverage = float(np.count_nonzero(hair_mask > 0) / float(gray.size))

    # Total combined obstruction (capped at 1.0)
    total_obstruction = float(np.clip(vignette_coverage + hair_coverage, 0.0, 1.0))

    return {
        "vignette_coverage": vignette_coverage,
        "hair_coverage": hair_coverage,
        "total_obstruction": total_obstruction,
    }


# ============================================================================
# 2. BATCH METRIC EXTRACTION ON A SINGLE IMAGE
# ============================================================================

def extract_all_quality_metrics(img: np.ndarray) -> Dict[str, float]:
    """
    Runs all 4 standalone metric extractors on a single image and aggregates
    the resulting metrics into a flat dictionary.

    Parameters
    ----------
    img : np.ndarray
        Image array.

    Returns
    -------
    Dict[str, float]
        Dictionary with all computed quality metrics.
    """
    blur_score = measure_blur(img)
    illum_metrics = measure_illumination(img)
    glare_pct = measure_glare(img)
    obstruction_metrics = measure_obstruction(img)

    return {
        "blur_laplacian_var": blur_score,
        "illumination_mean_brightness": illum_metrics["mean_brightness"],
        "illumination_underexposed_fraction": illum_metrics["underexposed_fraction"],
        "illumination_overexposed_fraction": illum_metrics["overexposed_fraction"],
        "glare_percentage": glare_pct,
        "vignette_coverage": obstruction_metrics["vignette_coverage"],
        "hair_coverage": obstruction_metrics["hair_coverage"],
    }


# ============================================================================
# 3. STATISTICAL PERCENTILE COMPUTATION
# ============================================================================

def compute_metric_percentiles(
    samples_dict: Dict[str, List[float]]
) -> Dict[str, Dict[str, float]]:
    """
    Computes summary statistics and key percentiles (1st, 5th, 50th, 95th, 99th)
    across the collected metric samples to establish WARN and FAIL boundaries.

    Parameters
    ----------
    samples_dict : Dict[str, List[float]]
        Dictionary mapping metric names to lists of observed numerical values.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary mapping metric names to statistics and percentiles.
    """
    results = {}
    for metric_name, values in samples_dict.items():
        arr = np.array(values, dtype=np.float64)
        if arr.size == 0:
            continue
        results[metric_name] = {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "p1": float(np.percentile(arr, 1.0)),
            "p5": float(np.percentile(arr, 5.0)),
            "p50": float(np.percentile(arr, 50.0)),
            "p95": float(np.percentile(arr, 95.0)),
            "p99": float(np.percentile(arr, 99.0)),
            "max": float(np.max(arr)),
        }
    return results


# ============================================================================
# 4. CODE CONSTANTS FORMATTER FOR src/quality_agent.py
# ============================================================================

def format_threshold_constants(stats: Dict[str, Dict[str, float]]) -> str:
    """
    Generates a Python code snippet with threshold constants established from
    the computed 1st, 5th, 95th, and 99th percentiles.
    Ready to be directly copy-pasted into `src/quality_agent.py`.
    """
    blur_p1 = stats.get("blur_laplacian_var", {}).get("p1", 50.0)
    blur_p5 = stats.get("blur_laplacian_var", {}).get("p5", 100.0)

    bright_p1 = stats.get("illumination_mean_brightness", {}).get("p1", 45.0)
    bright_p5 = stats.get("illumination_mean_brightness", {}).get("p5", 65.0)
    bright_p95 = stats.get("illumination_mean_brightness", {}).get("p95", 195.0)
    bright_p99 = stats.get("illumination_mean_brightness", {}).get("p99", 220.0)

    under_p95 = stats.get("illumination_underexposed_fraction", {}).get("p95", 0.15)
    under_p99 = stats.get("illumination_underexposed_fraction", {}).get("p99", 0.30)

    over_p95 = stats.get("illumination_overexposed_fraction", {}).get("p95", 0.10)
    over_p99 = stats.get("illumination_overexposed_fraction", {}).get("p99", 0.25)

    glare_p95 = stats.get("glare_percentage", {}).get("p95", 3.0)
    glare_p99 = stats.get("glare_percentage", {}).get("p99", 8.0)

    vig_p95 = stats.get("vignette_coverage", {}).get("p95", 0.20)
    vig_p99 = stats.get("vignette_coverage", {}).get("p99", 0.45)

    hair_p95 = stats.get("hair_coverage", {}).get("p95", 0.05)
    hair_p99 = stats.get("hair_coverage", {}).get("p99", 0.12)

    code = f'''# ============================================================================
# CALIBRATED QUALITY THRESHOLDS (Module 1 Output)
# Established from dataset calibration (seed=42)
# Ready to be imported / pasted directly into src/quality_agent.py
# ============================================================================

# Blur: Variance of Laplacian (Lower values indicate blur)
# - Below FAIL: Image is severely blurred; reject/fail triage.
# - Below WARN: Image has borderline sharpness; flag warning.
BLUR_FAIL_THRESHOLD = {blur_p1:.2f}
BLUR_WARN_THRESHOLD = {blur_p5:.2f}

# Mean Brightness: Range [0.0, 255.0]
# - Two-sided check (too dark or too bright)
BRIGHTNESS_FAIL_LOW = {bright_p1:.2f}
BRIGHTNESS_WARN_LOW = {bright_p5:.2f}
BRIGHTNESS_WARN_HIGH = {bright_p95:.2f}
BRIGHTNESS_FAIL_HIGH = {bright_p99:.2f}

# Underexposed Fraction: Pixels with luminance < 30
UNDEREXPOSED_WARN_THRESHOLD = {under_p95:.4f}
UNDEREXPOSED_FAIL_THRESHOLD = {under_p99:.4f}

# Overexposed Fraction: Pixels with luminance > 225
OVEREXPOSED_WARN_THRESHOLD = {over_p95:.4f}
OVEREXPOSED_FAIL_THRESHOLD = {over_p99:.4f}

# Glare Percentage: HSV V > 230 and S < 30
GLARE_WARN_THRESHOLD_PCT = {glare_p95:.2f}
GLARE_FAIL_THRESHOLD_PCT = {glare_p99:.2f}

# Vignette Coverage: Dark pixels in corner bounding boxes
VIGNETTE_WARN_THRESHOLD = {vig_p95:.4f}
VIGNETTE_FAIL_THRESHOLD = {vig_p99:.4f}

# Hair Coverage: Morphological Black-Hat segmented hair mask
HAIR_WARN_THRESHOLD = {hair_p95:.4f}
HAIR_FAIL_THRESHOLD = {hair_p99:.4f}

# Consolidated Quality Calibration Dictionary
QUALITY_THRESHOLDS = {{
    "blur": {{
        "fail_low": BLUR_FAIL_THRESHOLD,
        "warn_low": BLUR_WARN_THRESHOLD,
    }},
    "brightness": {{
        "fail_low": BRIGHTNESS_FAIL_LOW,
        "warn_low": BRIGHTNESS_WARN_LOW,
        "warn_high": BRIGHTNESS_WARN_HIGH,
        "fail_high": BRIGHTNESS_FAIL_HIGH,
    }},
    "underexposed_fraction": {{
        "warn_high": UNDEREXPOSED_WARN_THRESHOLD,
        "fail_high": UNDEREXPOSED_FAIL_THRESHOLD,
    }},
    "overexposed_fraction": {{
        "warn_high": OVEREXPOSED_WARN_THRESHOLD,
        "fail_high": OVEREXPOSED_FAIL_THRESHOLD,
    }},
    "glare_percentage": {{
        "warn_high": GLARE_WARN_THRESHOLD_PCT,
        "fail_high": GLARE_FAIL_THRESHOLD_PCT,
    }},
    "vignette_coverage": {{
        "warn_high": VIGNETTE_WARN_THRESHOLD,
        "fail_high": VIGNETTE_FAIL_THRESHOLD,
    }},
    "hair_coverage": {{
        "warn_high": HAIR_WARN_THRESHOLD,
        "fail_high": HAIR_FAIL_THRESHOLD,
    }},
}}
'''
    return code


# ============================================================================
# 5. SYNTHETIC SAMPLE DATA GENERATOR (For verification / self-testing)
# ============================================================================

def generate_synthetic_calibration_set(
    output_dir: str, num_images: int = 500, seed: int = 42
) -> List[str]:
    """
    Generates a realistic synthetic dataset simulating clinical/dermoscopic
    images for self-testing and calibration verification when raw dataset
    is not yet mounted.

    Simulates:
      - Normal skin textures and tones
      - Optical blur variations
      - Illumination variations (under/over exposure)
      - Specular glare highlights
      - Corner vignetting
      - Fine hair artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    created_files = []

    print(f"[Self-Test] Generating {num_images} synthetic images in '{output_dir}' (seed={seed})...")

    for i in range(num_images):
        h, w = 300, 300
        # Base skin tone in BGR
        base_b = rng.randint(120, 190)
        base_g = rng.randint(base_b, min(230, base_b + 40))
        base_r = rng.randint(base_g, min(255, base_g + 40))
        
        # Base canvas with Gaussian noise
        noise = rng.normal(0, 10, (h, w, 3))
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        canvas[:, :, 0] = base_b + noise[:, :, 0]
        canvas[:, :, 1] = base_g + noise[:, :, 1]
        canvas[:, :, 2] = base_r + noise[:, :, 2]
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

        # 1. Simulate blur variation
        if rng.rand() < 0.08:
            ksize = rng.choice([7, 11, 15])
            canvas = cv2.GaussianBlur(canvas, (ksize, ksize), 0)

        # 2. Simulate specular glare
        if rng.rand() < 0.12:
            gx = rng.randint(50, w - 50)
            gy = rng.randint(50, h - 50)
            gr = rng.randint(10, 25)
            cv2.circle(canvas, (gx, gy), gr, (255, 255, 255), -1)

        # 3. Simulate dark corners / vignetting
        if rng.rand() < 0.10:
            corner_mask = np.ones((h, w), dtype=np.float32)
            cv2.circle(corner_mask, (w // 2, h // 2), int(w * 0.65), 0, -1)
            canvas[corner_mask == 1] = (canvas[corner_mask == 1] * 0.1).astype(np.uint8)

        # 4. Simulate hair artifacts
        if rng.rand() < 0.15:
            num_hairs = rng.randint(1, 5)
            for _ in range(num_hairs):
                pt1 = (rng.randint(0, w), rng.randint(0, h))
                pt2 = (rng.randint(0, w), rng.randint(0, h))
                cv2.line(canvas, pt1, pt2, (20, 20, 20), thickness=rng.choice([1, 2]))

        filename = os.path.join(output_dir, f"sample_{i:04d}.jpg")
        cv2.imwrite(filename, canvas)
        created_files.append(filename)

    print(f"[Self-Test] Completed generating {len(created_files)} images.")
    return created_files


# ============================================================================
# 6. MAIN CALIBRATION ROUTINE
# ============================================================================

def run_calibration(
    data_dir: str,
    sample_size: int = 500,
    seed: int = 42,
    output_constants_file: Optional[str] = None,
    allow_synthetic: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], str]:
    """
    Main calibration routine:
      1. Samples 500 images randomly with seed=42 from the specified directory.
      2. Computes all quality metrics (blur, illumination, glare, obstruction) across the sample.
      3. Computes 1st, 5th, 95th, and 99th percentiles to establish WARN and FAIL boundaries.
      4. Generates threshold constants formatted for src/quality_agent.py.

    Parameters
    ----------
    data_dir : str
        Path to directory containing images.
    sample_size : int, default=500
        Number of images to sample.
    seed : int, default=42
        Random seed for sampling reproducibility.
    output_constants_file : Optional[str]
        Optional file path to write generated threshold constants.
    allow_synthetic : bool, default=False
        If True and data_dir is empty or missing, generates synthetic dataset.

    Returns
    -------
    Tuple[Dict[str, Dict[str, float]], str]
        Statistics dictionary and formatted code string.
    """
    print(f"=== Quality Calibration Routine (Module 1) ===")
    print(f"Target Data Directory: {data_dir}")
    print(f"Sample Size: {sample_size}")
    print(f"Random Seed: {seed}")

    # Gather image files
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    all_files = []
    
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    all_files.append(os.path.join(root, file))

    if not all_files:
        if allow_synthetic:
            print(f"[WARN] No images found in '{data_dir}'. Generating synthetic dataset for verification...")
            synth_dir = os.path.join(data_dir, "synthetic_eval_set")
            all_files = generate_synthetic_calibration_set(synth_dir, num_images=sample_size, seed=seed)
        else:
            raise FileNotFoundError(
                f"No supported image files found in '{data_dir}'. "
                f"Please ensure images exist or run with --generate-sample-data."
            )

    print(f"Found {len(all_files)} total images.")

    # Sort files deterministically before sampling
    all_files.sort()

    # Fixed seed sampling
    rng = random.Random(seed)
    if len(all_files) >= sample_size:
        sampled_files = rng.sample(all_files, sample_size)
    else:
        print(f"[WARN] Total images ({len(all_files)}) is less than sample size ({sample_size}). Using all available images.")
        sampled_files = all_files

    print(f"Processing {len(sampled_files)} sampled images...")

    # Data collector
    metric_records: Dict[str, List[float]] = {
        "blur_laplacian_var": [],
        "illumination_mean_brightness": [],
        "illumination_underexposed_fraction": [],
        "illumination_overexposed_fraction": [],
        "glare_percentage": [],
        "vignette_coverage": [],
        "hair_coverage": [],
    }

    processed_count = 0
    failed_count = 0

    for idx, filepath in enumerate(sampled_files):
        img = cv2.imread(filepath)
        if img is None:
            failed_count += 1
            continue

        try:
            metrics = extract_all_quality_metrics(img)
            for k, v in metrics.items():
                metric_records[k].append(v)
            processed_count += 1
        except Exception as e:
            failed_count += 1
            print(f"[WARN] Error processing {filepath}: {e}")

        # Progress reporting
        if (idx + 1) % 100 == 0 or (idx + 1) == len(sampled_files):
            print(f"  Processed {idx + 1}/{len(sampled_files)} images...")

    print(f"Extraction complete: {processed_count} succeeded, {failed_count} failed.")

    if processed_count == 0:
        raise RuntimeError("No images were successfully processed during calibration.")

    # Compute percentiles
    stats = compute_metric_percentiles(metric_records)

    # Display results table
    print("\n" + "=" * 80)
    print("CALIBRATION STATISTICAL PERCENTILES (Empirical Quality Boundaries)")
    print("=" * 80)
    header = f"{'Metric Name':<38} | {'1%':>8} | {'5%':>8} | {'50%':>8} | {'95%':>8} | {'99%':>8}"
    print(header)
    print("-" * 80)
    for metric_name, p in stats.items():
        row = (
            f"{metric_name:<38} | "
            f"{p['p1']:>8.2f} | "
            f"{p['p5']:>8.2f} | "
            f"{p['p50']:>8.2f} | "
            f"{p['p95']:>8.2f} | "
            f"{p['p99']:>8.2f}"
        )
        print(row)
    print("=" * 80 + "\n")

    # Generate constants code
    constants_code = format_threshold_constants(stats)
    print("Generated Threshold Constants for src/quality_agent.py:\n")
    print(constants_code)

    if output_constants_file:
        out_dir = os.path.dirname(output_constants_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_constants_file, "w", encoding="utf-8") as f:
            f.write(constants_code)
        print(f"Saved threshold constants to '{output_constants_file}'.")

    return stats, constants_code


# ============================================================================
# 7. CLI ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Module 1: Classical OpenCV/NumPy Image Quality Metric Extractor & Calibration Routine"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/train",
        help="Path to training images directory (default: data/train)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of images to randomly sample (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to save generated Python threshold constants",
    )
    parser.add_argument(
        "--generate-sample-data",
        action="store_true",
        help="Generate synthetic training images if data-dir is empty or missing (for self-testing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run_calibration(
            data_dir=args.data_dir,
            sample_size=args.sample_size,
            seed=args.seed,
            output_constants_file=args.output_file,
            allow_synthetic=args.generate_sample_data,
        )
    except Exception as e:
        print(f"[ERROR] Calibration failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
