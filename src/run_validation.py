"""
src/run_validation.py — Validation pipeline runner for Agentic DermaGuard.

Author: Ilka Mansoori (Project Lead / ML Senior Engineer)
Integrates:
    - Data Loader: src.data (Aashi)
    - Quality Assessment: src.quality_agent (Palak)
    - Diagnostic Classifier: src.diagnosis_agent (Ilka)

Generates:
    outputs/val_results.csv (18 columns, 1,942 rows) strictly adhering to
    the competition evaluation contract defined in TEAM_PLAN_v3.md.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.common import CLASS_NAMES, OUTPUTS_DIR
from src.data import load_split
from src.diagnosis_agent import DiagnosisAgent
from src.quality_agent import assess_quality

# Exact 18-column contract
COLUMNS_18: List[str] = [
    "image_id",
    "true_label",
    "pred_class",
    "confidence",
    "prob_akiec",
    "prob_bcc",
    "prob_bkl",
    "prob_df",
    "prob_mel",
    "prob_nv",
    "prob_vasc",
    "q_status",
    "q_score",
    "q_reasons",
    "q_blur",
    "q_illumination",
    "q_glare",
    "q_obstruction",
]


def extract_subcheck_score(check_data: Dict[str, Any]) -> float:
    """
    Derive a normalized numeric quality score in [0.0, 1.0] for a subcheck.
    PASS -> 1.00, WARN -> 0.70, FAIL -> 0.20
    """
    status = check_data.get("status", "PASS")
    if status == "PASS":
        return 1.0
    elif status == "WARN":
        return 0.70
    else:
        return 0.20


def run_validation(
    split: str = "eval",
    limit: Optional[int] = None,
    output_path: Optional[Path] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Execute validation pipeline across all images in the specified split.

    Parameters
    ----------
    split : str
        Dataset split to evaluate ('eval' or 'val'). Default: 'eval'.
    limit : int, optional
        Optional limit on number of images (for smoke testing).
    output_path : Path, optional
        Custom destination for CSV. Default: outputs/val_results.csv.
    show_progress : bool
        Whether to print periodic progress updates.

    Returns
    -------
    pd.DataFrame
        DataFrame with 18 columns matching the competition contract.
    """
    if output_path is None:
        output_path = OUTPUTS_DIR / "val_results.csv"

    print(f"[*] Loading dataset split '{split}' via src.data.load_split...")
    records = load_split(split)
    total_records = len(records)
    print(f"[+] Successfully loaded {total_records:,} dataset records.")

    if limit is not None:
        records = records[:limit]
        print(f"[*] Limiting run to first {len(records)} images for smoke test.")

    print("[*] Initializing Diagnosis Agent (EfficientNet-B0 + LogisticRegression)...")
    diag_agent = DiagnosisAgent()

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    start_time = time.time()
    n_images = len(records)
    print(f"[*] Starting full pipeline across {n_images:,} images...")

    for i, (image_id, image_path, true_label) in enumerate(records, 1):
        try:
            # 1. Quality Assessment
            q_res = assess_quality(image_path)
            q_status = q_res.get("status", "FAIL")
            q_score = float(round(q_res.get("score", 0.0), 4))
            reasons_list = q_res.get("reasons", [])
            q_reasons = "; ".join(reasons_list) if reasons_list else ""

            checks = q_res.get("checks", {})
            q_blur = extract_subcheck_score(checks.get("blur", {}))
            q_illum = extract_subcheck_score(checks.get("illumination", {}))
            q_glare = extract_subcheck_score(checks.get("glare", {}))
            q_obst = extract_subcheck_score(checks.get("obstruction", {}))

            # 2. Diagnostic Inference
            d_res = diag_agent.diagnose(image_path)
            raw_probs = d_res["probabilities"]

            # Normalize probabilities ensuring exact sum = 1.0000
            prob_arr = np.array([float(raw_probs[c]) for c in CLASS_NAMES], dtype=np.float64)
            p_sum = prob_arr.sum()
            if p_sum > 0:
                prob_arr /= p_sum
            prob_arr = np.round(prob_arr, 4)
            # Adjust rounding residual to make sum exactly 1.0
            prob_arr[np.argmax(prob_arr)] += round(1.0 - prob_arr.sum(), 4)
            prob_arr = np.clip(prob_arr, 0.0, 1.0)

            # Argmax determination
            best_idx = int(np.argmax(prob_arr))
            pred_class = CLASS_NAMES[best_idx]
            confidence = float(prob_arr[best_idx])

            # Build record matching 18 columns exactly
            row = {
                "image_id": image_id,
                "true_label": true_label,
                "pred_class": pred_class,
                "confidence": confidence,
            }
            for idx, c in enumerate(CLASS_NAMES):
                row[f"prob_{c}"] = float(prob_arr[idx])

            row.update({
                "q_status": q_status,
                "q_score": q_score,
                "q_reasons": q_reasons,
                "q_blur": q_blur,
                "q_illumination": q_illum,
                "q_glare": q_glare,
                "q_obstruction": q_obst,
            })

            rows.append(row)

        except Exception as e:
            err_msg = f"Inference failed for {image_id}: {str(e)}"
            print(f"[!] ERROR: {err_msg}", file=sys.stderr)
            failures.append({"image_id": image_id, "error": str(e)})

        if show_progress and (i % 200 == 0 or i == n_images):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_images - i) / rate if rate > 0 else 0
            print(f"    Progress: {i:,}/{n_images:,} ({i/n_images*100:.1f}%) | "
                  f"Rate: {rate:.1f} img/s | ETA: {eta:.0f}s")

    elapsed_total = time.time() - start_time
    print(f"[+] Processed {len(rows):,} images in {elapsed_total:.2f}s ({len(rows)/elapsed_total:.1f} img/s).")

    if failures:
        print(f"[!] WARNING: Encountered {len(failures)} failures during validation run!", file=sys.stderr)
        for f in failures[:5]:
            print(f"    - {f['image_id']}: {f['error']}", file=sys.stderr)
        raise RuntimeError(f"Validation run failed on {len(failures)} images. CSV generation aborted.")

    # Create DataFrame in exact contract column order
    df = pd.DataFrame(rows)[COLUMNS_18]

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Successfully wrote {len(df):,} records to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Agentic DermaGuard validation pipeline.")
    parser.add_argument("--smoke", action="store_true", help="Run 5-image smoke test")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    parser.add_argument("--output", type=str, default=None, help="Custom output CSV path")
    args = parser.parse_args()

    lim = 5 if args.smoke else args.limit
    out_p = Path(args.output) if args.output else None
    run_validation(split="eval", limit=lim, output_path=out_p)
