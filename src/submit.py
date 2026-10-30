"""Submission Integration and Hidden-Test Evaluation Runner.

Module: src/submit.py
Author: Aashi (Submission Integration Owner)
Project: Agentic DermaGuard Healthcare Hackathon

Responsibilities:
- Generate and validate the final hidden-test evaluation CSV.
- Enforce the exact 3-column official format:
    image_id,predicted_label,decision
- Reject missing or duplicate image IDs.
- Ensure predicted_label is one of the 7 canonical classes.
- Ensure decision is one of: AUTO-TRIAGE, REFER, RE-IMAGE.
- Save output locally with strict data validation.
- Guard external portal submission behind explicit authorization and environment credentials.
  (Never submits externally without explicit approval from Technical Lead Ilka).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, List, Optional, Union

# Ensure project root is available
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CLASS_LABELS, validate_image_id

# Canonical 3-column schema specified for official submission
SUBMISSION_COLUMNS: list[str] = ["image_id", "predicted_label", "decision"]

# Allowed decision categories based on Level 2 Triage Contract
VALID_DECISIONS: tuple[str, ...] = ("AUTO-TRIAGE", "REFER", "RE-IMAGE")

# Canonical 7 classes from project contract
VALID_PREDICTED_LABELS: tuple[str, ...] = tuple(CLASS_LABELS)


def normalize_decision(decision: Any) -> str:
    """Normalize and validate a clinical decision string."""
    if not isinstance(decision, str):
        raise ValueError(f"Decision must be a string, got {type(decision).__name__}.")
    d = decision.strip().upper().replace("_", "-")
    if d not in VALID_DECISIONS:
        raise ValueError(
            f"Invalid decision '{decision}'. Expected one of: {list(VALID_DECISIONS)}"
        )
    return d


def validate_submission_record(
    image_id: Any,
    predicted_label: Any,
    decision: Any,
) -> dict[str, str]:
    """Validate a single submission row record.

    Args:
        image_id: Unique image identifier.
        predicted_label: One of the 7 canonical diagnostic labels.
        decision: One of 'AUTO-TRIAGE', 'REFER', 'RE-IMAGE'.

    Returns:
        Sanitized record dictionary matching SUBMISSION_COLUMNS.

    Raises:
        ValueError: If any field fails security or domain validation.
    """
    clean_id = validate_image_id(image_id)

    if not isinstance(predicted_label, str):
        raise ValueError(
            f"predicted_label must be a string, got {type(predicted_label).__name__} for '{image_id}'."
        )
    clean_label = predicted_label.strip().lower()
    if clean_label not in VALID_PREDICTED_LABELS:
        raise ValueError(
            f"Invalid predicted_label '{predicted_label}' for '{image_id}'. "
            f"Expected one of canonical classes: {list(VALID_PREDICTED_LABELS)}"
        )

    clean_decision = normalize_decision(decision)

    return {
        "image_id": clean_id,
        "predicted_label": clean_label,
        "decision": clean_decision,
    }


def validate_submission_file(csv_path: Union[str, Path]) -> list[dict[str, str]]:
    """Validate a submission CSV file against the official 3-column contract.

    Verifies:
    1. File exists on disk.
    2. Header contains exactly 'image_id', 'predicted_label', 'decision' in order.
    3. No extra or missing columns.
    4. Dataset contains at least 1 record.
    5. Zero duplicate image IDs.
    6. Every predicted_label belongs to CLASS_LABELS.
    7. Every decision is AUTO-TRIAGE, REFER, or RE-IMAGE.

    Args:
        csv_path: Path to candidate submission CSV file.

    Returns:
        List of validated record dictionaries.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If schema, duplicates, or value constraints are violated.
    """
    path = Path(csv_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Submission CSV file does not exist at: {path}")

    records: list[dict[str, str]] = []
    seen_ids: list[str] = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Submission file '{path.name}' is empty.")

        clean_header = [col.strip() for col in header]
        if clean_header != SUBMISSION_COLUMNS:
            raise ValueError(
                f"Invalid submission CSV header in '{path.name}': {clean_header}. "
                f"Expected exactly: {SUBMISSION_COLUMNS}"
            )

        for row_idx, row in enumerate(reader, start=2):
            if not row or all(c.strip() == "" for c in row):
                continue  # skip trailing empty lines

            if len(row) != 3:
                raise ValueError(
                    f"Row {row_idx} in '{path.name}' has {len(row)} columns, expected 3: {row}"
                )

            raw_id, raw_label, raw_dec = row
            try:
                rec = validate_submission_record(raw_id, raw_label, raw_dec)
            except ValueError as e:
                raise ValueError(f"Validation error at row {row_idx} in '{path.name}': {e}") from e

            records.append(rec)
            seen_ids.append(rec["image_id"])

    if not records:
        raise ValueError(f"Submission file '{path.name}' contains 0 valid data rows.")

    # Duplicate check
    counts = Counter(seen_ids)
    duplicates = [img_id for img_id, c in counts.items() if c > 1]
    if duplicates:
        sample_dup = duplicates[:5]
        raise ValueError(
            f"Found {len(duplicates)} duplicate image_id(s) in '{path.name}'. "
            f"Sample duplicates: {sample_dup}"
        )

    return records


def write_submission_csv(
    records: list[dict[str, str]],
    output_path: Union[str, Path] = "outputs/submission.csv",
) -> Path:
    """Safely serialize validated records to the official submission CSV format.

    Args:
        records: List of record dictionaries with keys 'image_id', 'predicted_label', 'decision'.
        output_path: Target CSV file path.

    Returns:
        Path to the verified output file.
    """
    if not records:
        raise ValueError("Cannot write an empty submission records list.")

    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Validate before writing
    clean_records = []
    seen = set()
    for idx, r in enumerate(records):
        clean = validate_submission_record(
            r.get("image_id"),
            r.get("predicted_label"),
            r.get("decision"),
        )
        img_id = clean["image_id"]
        if img_id in seen:
            raise ValueError(f"Duplicate image_id '{img_id}' encountered at record index {idx}.")
        seen.add(img_id)
        clean_records.append(clean)

    with open(out_p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        for r in clean_records:
            writer.writerow(r)

    # Re-verify written file
    validate_submission_file(out_p)
    return out_p


def submit_to_portal(
    csv_path: Union[str, Path],
    confirm: bool = False,
    endpoint_url: Optional[str] = None,
    api_token: Optional[str] = None,
) -> dict[str, Any]:
    """Client for portal submission with explicit authorization safeguards.

    Security & Compliance Rules:
    - Default behavior is OFFLINE / DRY-RUN.
    - Requires explicit `confirm=True` parameter to proceed.
    - Requires `PORTAL_API_URL` and `PORTAL_API_KEY` environment variables.
    - Never commits credentials or sends unapproved network requests.

    Args:
        csv_path: Path to the validated submission CSV.
        confirm: Explicit flag requiring user confirmation before external transmission.
        endpoint_url: Optional override URL (defaults to $PORTAL_API_URL).
        api_token: Optional override token (defaults to $PORTAL_API_KEY).

    Returns:
        Status dictionary describing the action taken or refusal reason.
    """
    validated_records = validate_submission_file(csv_path)

    if not confirm:
        return {
            "status": "REFUSED",
            "authorized": False,
            "message": (
                "Submission aborted: Explicit authorization flag (confirm=True) was not provided. "
                "Per hackathon governance rules, submit externally only with explicit approval "
                "from Technical Lead Ilka."
            ),
            "records_ready": len(validated_records),
            "csv_path": str(Path(csv_path).resolve()),
        }

    url = endpoint_url or os.getenv("PORTAL_API_URL")
    token = api_token or os.getenv("PORTAL_API_KEY")

    if not url:
        return {
            "status": "BLOCKED",
            "authorized": True,
            "message": (
                "Missing PORTAL_API_URL. The organizer HTTP submission endpoint is unconfirmed/UNKNOWN. "
                "Output CSV has been saved locally for offline evaluation."
            ),
            "records_ready": len(validated_records),
            "csv_path": str(Path(csv_path).resolve()),
        }

    # If endpoint exists in future, network client would execute here
    return {
        "status": "PENDING_PORTAL_SPECIFICATION",
        "authorized": True,
        "endpoint": url,
        "records_count": len(validated_records),
        "message": "Organizer endpoint schema unconfirmed; offline CSV preserved.",
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build command-line parser for submission runner."""
    parser = argparse.ArgumentParser(
        description="Agentic DermaGuard — Hidden-Test Submission Runner (src/submit.py)"
    )
    parser.add_argument(
        "--validate-only",
        type=str,
        default=None,
        help="Validate an existing submission CSV without running inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/submission.csv",
        help="Output path for submission CSV (default: outputs/submission.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode without external transmission",
    )
    return parser


def main() -> None:
    """CLI entrypoint for submission validation and generation."""
    parser = _build_cli_parser()
    args = parser.parse_args()

    print("=" * 70)
    print("  Agentic DermaGuard — Submission Integration (`src/submit.py`)")
    print("=" * 70)

    if args.validate_only:
        print(f"\n[*] Validating submission file: {args.validate_only}")
        try:
            records = validate_submission_file(args.validate_only)
            print(f"[+] VALIDATION SUCCESS: {len(records):,} valid rows conforming to schema:")
            print(f"    Columns : {SUBMISSION_COLUMNS}")
            print(f"    Classes : {VALID_PREDICTED_LABELS}")
            print(f"    Decisions: {VALID_DECISIONS}")
        except Exception as e:
            print(f"[!] VALIDATION ERROR: {e}")
            sys.exit(1)
        return

    print("\n[*] Submission runner ready.")
    print("    Format: image_id,predicted_label,decision")
    print("    Ready for hidden test evaluation when test images are supplied.")


if __name__ == "__main__":
    main()
