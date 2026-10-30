"""
summarize_quality.py — Level 1 Quality Summary Tool
====================================================
Analyzes quality assessment outcomes from a validation run and prints a
clean Markdown report ready to be pasted into the Level 1 summary submission.

Usage
-----
# On real results:
    python src/summarize_quality.py --input outputs/val_results.csv
    python src/summarize_quality.py --input outputs/val_results.json

# Dry-run on 2,000 simulated images (no file needed):
    python src/summarize_quality.py --dry-run
    python src/summarize_quality.py --dry-run --seed 42 --n 2000

Output
------
Prints a Markdown-formatted quality summary table to stdout.
Optionally saves the report to --output <path>.

Author : Palak (palak-quality branch, Task P4)
Depends: numpy, csv, json (stdlib)  — no opencv needed at summary time
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Empirical distributions from Module 1 calibration (seed=42, N=500 train)
# Used exclusively by the dry-run generator.
# ---------------------------------------------------------------------------
_EMPIRICAL = {
    # blur Laplacian variance:  ~LogNormal(mu=5.2, sigma=0.9), clipped [5, 1500]
    "blur_mu": 5.2,
    "blur_sigma": 0.9,
    # illumination Y-channel mean: ~Normal(mu=128, sigma=38), clipped [10, 245]
    "bright_mu": 128.0,
    "bright_sigma": 38.0,
    # underexposed fraction: ~Beta(alpha=0.5, beta=8), scale to [0, 1]
    "underexp_alpha": 0.5,
    "underexp_beta": 8.0,
    # overexposed fraction: ~Beta(alpha=0.4, beta=9)
    "overexp_alpha": 0.4,
    "overexp_beta": 9.0,
    # glare pct: ~Exponential(lambda=1/2), clipped [0, 40]
    "glare_lambda": 2.0,
    # vignette coverage: ~Beta(alpha=1.2, beta=7)
    "vig_alpha": 1.2,
    "vig_beta": 7.0,
    # hair coverage: ~Beta(alpha=0.6, beta=12)
    "hair_alpha": 0.6,
    "hair_beta": 12.0,
}

# ---------------------------------------------------------------------------
# Thresholds — mirrored from quality_agent.py (P01/P05/P95/P99)
# ---------------------------------------------------------------------------
BLUR_FAIL_THRESHOLD         = 48.72
BLUR_WARN_THRESHOLD         = 82.15
BRIGHTNESS_FAIL_LOW         = 52.14
BRIGHTNESS_WARN_LOW         = 68.90
BRIGHTNESS_WARN_HIGH        = 192.35
BRIGHTNESS_FAIL_HIGH        = 214.80
UNDEREXPOSED_WARN_THRESHOLD = 0.1420
UNDEREXPOSED_FAIL_THRESHOLD = 0.2845
OVEREXPOSED_WARN_THRESHOLD  = 0.0890
OVEREXPOSED_FAIL_THRESHOLD  = 0.2180
GLARE_WARN_THRESHOLD_PCT    = 3.18
GLARE_FAIL_THRESHOLD_PCT    = 7.64
VIGNETTE_WARN_THRESHOLD     = 0.1810
VIGNETTE_FAIL_THRESHOLD     = 0.4230
HAIR_WARN_THRESHOLD         = 0.0430
HAIR_FAIL_THRESHOLD         = 0.1085

_DEDUCT_WARN = 0.15
_DEDUCT_FAIL = 0.35

# Anomaly gate: flag if (WARN + FAIL) / total > this fraction
FLAGGED_ANOMALY_THRESHOLD = 0.15  # 15%


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _beta_sample(alpha: float, beta: float, rng: random.Random) -> float:
    """Pure-Python Beta sample via Johnk's method (no numpy required)."""
    # Use Python's built-in random.betavariate
    return rng.betavariate(alpha, beta)


def _lognormal_sample(mu: float, sigma: float, rng: random.Random) -> float:
    return math.exp(rng.gauss(mu, sigma))


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Dry-run generator
# ---------------------------------------------------------------------------

def _evaluate_single(
    blur_var: float,
    bright: float,
    underexp: float,
    overexp: float,
    glare_pct: float,
    vignette: float,
    hair: float,
) -> dict[str, Any]:
    """
    Reproduce the assess_quality() scoring logic on raw metric values.
    Returns a result dict matching the schema contract.
    """
    checks: dict[str, dict] = {}
    reasons: list[str] = []
    n_warn = 0
    n_fail = 0

    # ---- blur ----
    if blur_var < BLUR_FAIL_THRESHOLD:
        blur_status = "FAIL"
        n_fail += 1
        reasons.append(
            f"Low sharpness (Laplacian variance {blur_var:.1f}, "
            f"below FAIL threshold {BLUR_FAIL_THRESHOLD})"
        )
    elif blur_var < BLUR_WARN_THRESHOLD:
        blur_status = "WARN"
        n_warn += 1
        reasons.append(
            f"Borderline sharpness (Laplacian variance {blur_var:.1f}, "
            f"below WARN threshold {BLUR_WARN_THRESHOLD})"
        )
    else:
        blur_status = "PASS"
    checks["blur"] = {
        "value": round(blur_var, 4),
        "status": blur_status,
        "threshold": BLUR_FAIL_THRESHOLD if blur_status == "FAIL" else BLUR_WARN_THRESHOLD,
        "passed": blur_status == "PASS",
    }

    # ---- illumination ----
    illum_status = "PASS"
    if (
        bright < BRIGHTNESS_FAIL_LOW
        or bright > BRIGHTNESS_FAIL_HIGH
        or underexp > UNDEREXPOSED_FAIL_THRESHOLD
        or overexp > OVEREXPOSED_FAIL_THRESHOLD
    ):
        illum_status = "FAIL"
        n_fail += 1
        if bright < BRIGHTNESS_FAIL_LOW:
            reasons.append(
                f"Underexposed (mean brightness {bright:.1f}, "
                f"below FAIL threshold {BRIGHTNESS_FAIL_LOW})"
            )
        if bright > BRIGHTNESS_FAIL_HIGH:
            reasons.append(
                f"Overexposed (mean brightness {bright:.1f}, "
                f"above FAIL threshold {BRIGHTNESS_FAIL_HIGH})"
            )
        if underexp > UNDEREXPOSED_FAIL_THRESHOLD:
            reasons.append(
                f"Severe underexposure (fraction {underexp:.4f}, "
                f"above FAIL threshold {UNDEREXPOSED_FAIL_THRESHOLD})"
            )
        if overexp > OVEREXPOSED_FAIL_THRESHOLD:
            reasons.append(
                f"Severe overexposure (fraction {overexp:.4f}, "
                f"above FAIL threshold {OVEREXPOSED_FAIL_THRESHOLD})"
            )
    elif (
        bright < BRIGHTNESS_WARN_LOW
        or bright > BRIGHTNESS_WARN_HIGH
        or underexp > UNDEREXPOSED_WARN_THRESHOLD
        or overexp > OVEREXPOSED_WARN_THRESHOLD
    ):
        illum_status = "WARN"
        n_warn += 1
        if bright < BRIGHTNESS_WARN_LOW:
            reasons.append(
                f"Dim image (mean brightness {bright:.1f}, "
                f"below WARN threshold {BRIGHTNESS_WARN_LOW})"
            )
        if bright > BRIGHTNESS_WARN_HIGH:
            reasons.append(
                f"Bright image (mean brightness {bright:.1f}, "
                f"above WARN threshold {BRIGHTNESS_WARN_HIGH})"
            )
        if underexp > UNDEREXPOSED_WARN_THRESHOLD:
            reasons.append(
                f"Underexposure warning (fraction {underexp:.4f}, "
                f"above WARN threshold {UNDEREXPOSED_WARN_THRESHOLD})"
            )
        if overexp > OVEREXPOSED_WARN_THRESHOLD:
            reasons.append(
                f"Overexposure warning (fraction {overexp:.4f}, "
                f"above WARN threshold {OVEREXPOSED_WARN_THRESHOLD})"
            )
    checks["illumination"] = {
        "value": round(bright, 4),
        "status": illum_status,
        "passed": illum_status == "PASS",
        "details": {
            "underexposed_fraction": round(underexp, 4),
            "overexposed_fraction": round(overexp, 4),
        },
    }

    # ---- glare ----
    if glare_pct > GLARE_FAIL_THRESHOLD_PCT:
        glare_status = "FAIL"
        n_fail += 1
        reasons.append(
            f"Glare detected ({glare_pct:.2f}% affected pixels, "
            f"above FAIL threshold {GLARE_FAIL_THRESHOLD_PCT}%)"
        )
    elif glare_pct > GLARE_WARN_THRESHOLD_PCT:
        glare_status = "WARN"
        n_warn += 1
        reasons.append(
            f"Mild glare ({glare_pct:.2f}% affected pixels, "
            f"above WARN threshold {GLARE_WARN_THRESHOLD_PCT}%)"
        )
    else:
        glare_status = "PASS"
    checks["glare"] = {
        "value": round(glare_pct, 4),
        "status": glare_status,
        "threshold": GLARE_FAIL_THRESHOLD_PCT if glare_status == "FAIL" else GLARE_WARN_THRESHOLD_PCT,
        "passed": glare_status == "PASS",
    }

    # ---- obstruction ----
    obs_value = max(vignette, hair)
    if vignette > VIGNETTE_FAIL_THRESHOLD or hair > HAIR_FAIL_THRESHOLD:
        obs_status = "FAIL"
        n_fail += 1
        if vignette > VIGNETTE_FAIL_THRESHOLD:
            reasons.append(
                f"Severe vignetting (coverage {vignette:.4f}, "
                f"above FAIL threshold {VIGNETTE_FAIL_THRESHOLD})"
            )
        if hair > HAIR_FAIL_THRESHOLD:
            reasons.append(
                f"Hair obstruction (Black-Hat coverage {hair:.4f}, "
                f"above FAIL threshold {HAIR_FAIL_THRESHOLD})"
            )
    elif vignette > VIGNETTE_WARN_THRESHOLD or hair > HAIR_WARN_THRESHOLD:
        obs_status = "WARN"
        n_warn += 1
        if vignette > VIGNETTE_WARN_THRESHOLD:
            reasons.append(
                f"Vignette warning (coverage {vignette:.4f}, "
                f"above WARN threshold {VIGNETTE_WARN_THRESHOLD})"
            )
        if hair > HAIR_WARN_THRESHOLD:
            reasons.append(
                f"Hair warning (Black-Hat coverage {hair:.4f}, "
                f"above WARN threshold {HAIR_WARN_THRESHOLD})"
            )
    else:
        obs_status = "PASS"
    checks["obstruction"] = {
        "value": round(obs_value, 4),
        "status": obs_status,
        "passed": obs_status == "PASS",
        "details": {
            "vignette_coverage": round(vignette, 4),
            "hair_coverage": round(hair, 4),
        },
    }

    # ---- overall score & status ----
    raw_score = 1.0 - n_fail * _DEDUCT_FAIL - n_warn * _DEDUCT_WARN
    score = round(max(0.0, min(1.0, raw_score)), 2)
    if n_fail > 0:
        status = "FAIL"
    elif n_warn > 0:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "status": status,
        "score": score,
        "checks": checks,
        "reasons": reasons,
    }


def generate_dry_run(n: int = 2000, seed: int = 42) -> list[dict[str, Any]]:
    """
    Simulate `n` quality evaluations using calibrated empirical distributions.
    Returns a list of result dicts matching the assess_quality() schema.
    """
    rng = random.Random(seed)
    results = []

    for _ in range(n):
        blur_var  = _clip(_lognormal_sample(_EMPIRICAL["blur_mu"],
                                            _EMPIRICAL["blur_sigma"], rng), 5.0, 1500.0)
        bright    = _clip(rng.gauss(_EMPIRICAL["bright_mu"],
                                    _EMPIRICAL["bright_sigma"]), 10.0, 245.0)
        underexp  = _beta_sample(_EMPIRICAL["underexp_alpha"],
                                  _EMPIRICAL["underexp_beta"], rng)
        overexp   = _beta_sample(_EMPIRICAL["overexp_alpha"],
                                  _EMPIRICAL["overexp_beta"], rng)
        glare_pct = _clip(rng.expovariate(1.0 / _EMPIRICAL["glare_lambda"]),
                          0.0, 40.0)
        vignette  = _beta_sample(_EMPIRICAL["vig_alpha"],
                                  _EMPIRICAL["vig_beta"], rng)
        hair      = _beta_sample(_EMPIRICAL["hair_alpha"],
                                  _EMPIRICAL["hair_beta"], rng)

        result = _evaluate_single(blur_var, bright, underexp, overexp,
                                   glare_pct, vignette, hair)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Loaders for real results files
# ---------------------------------------------------------------------------

def load_json(path: str) -> list[dict[str, Any]]:
    """Load a JSON file containing a list of assess_quality() result dicts."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Support {image_path: result_dict, ...} format
        return list(data.values())
    return data


def load_csv(path: str) -> list[dict[str, Any]]:
    """
    Load a CSV with at minimum columns: status, score, reasons,
    blur_status, illumination_status, glare_status, obstruction_status.
    """
    results = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reasons_raw = row.get("reasons", "")
            reasons = [r.strip() for r in reasons_raw.split("|") if r.strip()]
            results.append({
                "status": row.get("status", "FAIL"),
                "score":  float(row.get("score", 0.0)),
                "checks": {
                    "blur":         {"status": row.get("blur_status", "FAIL"),
                                     "passed": row.get("blur_status", "FAIL") == "PASS"},
                    "illumination": {"status": row.get("illumination_status", "FAIL"),
                                     "passed": row.get("illumination_status", "FAIL") == "PASS"},
                    "glare":        {"status": row.get("glare_status", "FAIL"),
                                     "passed": row.get("glare_status", "FAIL") == "PASS"},
                    "obstruction":  {"status": row.get("obstruction_status", "FAIL"),
                                     "passed": row.get("obstruction_status", "FAIL") == "PASS"},
                },
                "reasons": reasons,
            })
    return results


# ---------------------------------------------------------------------------
# Analysis engine
# ---------------------------------------------------------------------------

def analyze(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all summary statistics from a list of assess_quality() results."""
    total = len(results)
    if total == 0:
        raise ValueError("No results to analyze.")

    status_counts: Counter = Counter()
    check_status: dict[str, Counter] = {
        "blur": Counter(),
        "illumination": Counter(),
        "glare": Counter(),
        "obstruction": Counter(),
    }
    reason_counter: Counter = Counter()
    score_sum = 0.0

    for r in results:
        status_counts[r["status"]] += 1
        score_sum += float(r.get("score", 0.0))
        for check_name, check_data in r.get("checks", {}).items():
            if check_name in check_status:
                check_status[check_name][check_data.get("status", "FAIL")] += 1

        # Normalize reasons to category labels for frequency counting
        for reason in r.get("reasons", []):
            reason_lower = reason.lower()
            if "sharpness" in reason_lower or "laplacian" in reason_lower:
                cat = "blur"
            elif (
                "underexpos" in reason_lower
                or "overexpos" in reason_lower
                or "brightness" in reason_lower
                or "dim image" in reason_lower
                or "bright image" in reason_lower
            ):
                cat = "illumination"
            elif "glare" in reason_lower:
                cat = "glare"
            elif "vignett" in reason_lower or "hair" in reason_lower or "obstruction" in reason_lower:
                cat = "obstruction"
            else:
                cat = "other"
            reason_counter[cat] += 1

    n_pass = status_counts.get("PASS", 0)
    n_warn = status_counts.get("WARN", 0)
    n_fail = status_counts.get("FAIL", 0)
    n_flagged = n_warn + n_fail
    flagged_pct = 100.0 * n_flagged / total
    anomaly = flagged_pct > (FLAGGED_ANOMALY_THRESHOLD * 100)
    mean_score = score_sum / total

    return {
        "total": total,
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "n_flagged": n_flagged,
        "pct_pass": 100.0 * n_pass / total,
        "pct_warn": 100.0 * n_warn / total,
        "pct_fail": 100.0 * n_fail / total,
        "pct_flagged": flagged_pct,
        "mean_score": mean_score,
        "check_status": {k: dict(v) for k, v in check_status.items()},
        "reason_counter": dict(reason_counter),
        "anomaly": anomaly,
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def render_markdown(stats: dict[str, Any], source_label: str = "dry-run") -> str:
    """Return a complete Markdown quality summary report string."""
    total = stats["total"]
    lines: list[str] = []

    def pct(n: int) -> str:
        return f"{100.0 * n / total:.1f}%"

    lines.append("# Level 1 Quality Summary Report")
    lines.append(f"\n**Source:** {source_label}  ")
    lines.append(f"**Total images evaluated:** {total:,}  ")
    lines.append(f"**Mean quality score:** {stats['mean_score']:.3f}  \n")

    # ── Overall Status Distribution ──────────────────────────────────────
    lines.append("## Overall Status Distribution\n")
    lines.append("| Status | Count | Percentage |")
    lines.append("|--------|------:|-----------:|")
    lines.append(f"| ✅ PASS | {stats['n_pass']:,} | {stats['pct_pass']:.1f}% |")
    lines.append(f"| ⚠️ WARN | {stats['n_warn']:,} | {stats['pct_warn']:.1f}% |")
    lines.append(f"| ❌ FAIL | {stats['n_fail']:,} | {stats['pct_fail']:.1f}% |")
    lines.append(f"| **Total Flagged (WARN+FAIL)** | **{stats['n_flagged']:,}** | **{stats['pct_flagged']:.1f}%** |")

    # Anomaly warning
    if stats["anomaly"]:
        lines.append(
            f"\n> ⚠️ **ANOMALY FLAG:** Flagged images ({stats['pct_flagged']:.1f}%) exceed the "
            f"{FLAGGED_ANOMALY_THRESHOLD * 100:.0f}% threshold. "
            "Consider reviewing calibration thresholds — they may be too strict for this acquisition site."
        )
    else:
        lines.append(
            f"\n> ✅ Flagged rate ({stats['pct_flagged']:.1f}%) is within the acceptable "
            f"{FLAGGED_ANOMALY_THRESHOLD * 100:.0f}% limit. Thresholds appear well-calibrated."
        )

    # ── Per-Check Breakdown ──────────────────────────────────────────────
    lines.append("\n## Per-Check Status Breakdown\n")
    lines.append("| Check | PASS | WARN | FAIL | PASS% | WARN% | FAIL% |")
    lines.append("|-------|-----:|-----:|-----:|------:|------:|------:|")

    check_order = ["blur", "illumination", "glare", "obstruction"]
    check_labels = {
        "blur": "🔍 Blur / Sharpness",
        "illumination": "💡 Illumination",
        "glare": "✨ Glare",
        "obstruction": "🚧 Obstruction",
    }
    for check in check_order:
        cs = stats["check_status"].get(check, {})
        cp = cs.get("PASS", 0)
        cw = cs.get("WARN", 0)
        cf = cs.get("FAIL", 0)
        ct = cp + cw + cf
        if ct == 0:
            ct = 1  # avoid div-by-zero
        lines.append(
            f"| {check_labels[check]} "
            f"| {cp:,} | {cw:,} | {cf:,} "
            f"| {100*cp/ct:.1f}% | {100*cw/ct:.1f}% | {100*cf/ct:.1f}% |"
        )

    # ── Reason Frequency ────────────────────────────────────────────────
    lines.append("\n## Flagged Reason Frequency (by Category)\n")
    lines.append("| Reason Category | Occurrences | % of All Flags |")
    lines.append("|-----------------|------------:|---------------:|")
    rc = stats["reason_counter"]
    total_reasons = sum(rc.values()) or 1
    category_labels = {
        "blur": "Low sharpness / blur",
        "illumination": "Illumination (under / overexposure)",
        "glare": "Glare / specular reflection",
        "obstruction": "Obstruction (vignette / hair)",
        "other": "Other / unclassified",
    }
    for cat in ["blur", "illumination", "glare", "obstruction", "other"]:
        count = rc.get(cat, 0)
        if count == 0:
            continue
        lines.append(
            f"| {category_labels[cat]} | {count:,} | {100*count/total_reasons:.1f}% |"
        )

    # ── Level 2 Triage Projection ────────────────────────────────────────
    lines.append("\n## Level 2 Triage Projection\n")
    lines.append("| L1 Status | Count | Action |")
    lines.append("|-----------|------:|--------|")
    lines.append(f"| PASS | {stats['n_pass']:,} | → Standard Diagnosis Agent triage |")
    lines.append(f"| WARN | {stats['n_warn']:,} | → Raised confidence threshold; REFER if below |")
    lines.append(f"| FAIL | {stats['n_fail']:,} | → RE_IMAGE (return to acquisition, skip diagnosis) |")

    # ── Calibration Health ───────────────────────────────────────────────
    lines.append("\n## Calibration Health Check\n")
    lines.append("| Metric | Value | Verdict |")
    lines.append("|--------|------:|---------|")
    lines.append(
        f"| Mean quality score | {stats['mean_score']:.3f} "
        f"| {'✅ Healthy (≥ 0.70)' if stats['mean_score'] >= 0.70 else '⚠️ Low — review thresholds'} |"
    )
    lines.append(
        f"| Flagged rate (WARN+FAIL) | {stats['pct_flagged']:.1f}% "
        f"| {'✅ Within 15% gate' if not stats['anomaly'] else '❌ Exceeds 15% gate — recalibrate'} |"
    )
    lines.append(
        f"| FAIL rate | {stats['pct_fail']:.1f}% "
        f"| {'✅ ≤ 5%' if stats['pct_fail'] <= 5.0 else '⚠️ > 5% — investigate acquisition pipeline'} |"
    )

    lines.append("\n---")
    lines.append(
        "_Generated by `src/summarize_quality.py` (palak-quality, Task P4). "
        "Thresholds from Module 1 calibration (seed=42, N=500 train images)._"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Summarize Level 1 quality assessment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run on 2,000 simulated images
  python src/summarize_quality.py --dry-run

  # Real CSV results
  python src/summarize_quality.py --input outputs/val_results.csv

  # Real JSON results, save report
  python src/summarize_quality.py --input outputs/val_results.json --output outputs/quality_summary.md
""",
    )
    p.add_argument(
        "--input", "-i",
        metavar="FILE",
        help="Path to val_results.csv or val_results.json produced by the quality agent.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and analyze 2,000 simulated results (no real file needed).",
    )
    p.add_argument(
        "--n",
        type=int,
        default=2000,
        metavar="N",
        help="Number of images to simulate in dry-run mode (default: 2000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dry-run reproducibility (default: 42).",
    )
    p.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Optional path to save the Markdown report (prints to stdout regardless).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.dry_run and not args.input:
        parser.error("Provide --input <file> or --dry-run.")

    # ---- Load results ----
    if args.dry_run:
        print(
            f"[summarize_quality] Dry-run mode: generating {args.n:,} simulated "
            f"evaluations (seed={args.seed}) ...",
            file=sys.stderr,
        )
        results = generate_dry_run(n=args.n, seed=args.seed)
        source_label = f"dry-run ({args.n:,} simulated images, seed={args.seed})"
    else:
        path = args.input
        if not os.path.isfile(path):
            sys.exit(f"[summarize_quality] ERROR: File not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        print(f"[summarize_quality] Loading results from {path} ...", file=sys.stderr)
        if ext == ".json":
            results = load_json(path)
        elif ext == ".csv":
            results = load_csv(path)
        else:
            sys.exit(f"[summarize_quality] ERROR: Unsupported file extension '{ext}'. Use .json or .csv.")
        source_label = f"file: {path} ({len(results):,} images)"

    # ---- Analyze ----
    print("[summarize_quality] Computing statistics ...", file=sys.stderr)
    stats = analyze(results)

    # ---- Render ----
    report = render_markdown(stats, source_label=source_label)
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[summarize_quality] Report saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
