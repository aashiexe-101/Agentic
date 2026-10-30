"""Context and Demographic Evaluation Agent (Agent A4).

Module: src/context_agent.py
Author: Aashi (Context Agent Owner)
Project: Agentic DermaGuard Healthcare Hackathon

Responsibilities:
- Ingest and normalize non-image patient context: age, sex, and anatomical localization.
- Preserve missing clinical fields as explicit None/null values without inventing data.
- Explicitly report all missing fields in `missing_fields`.
- Strictly decouple from diagnostic ground-truth labels (never inspect 'dx' or 'true_label').
- Provide objective, transparent context evidence to downstream coordination agents.
- Never invent diagnostic predictions or unverified clinical risk rules.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Optional

# Standard canonical anatomical sites observed in clinical dermatoscopy / HAM10000
KNOWN_ANATOMICAL_SITES: tuple[str, ...] = (
    "abdomen",
    "acral",
    "back",
    "chest",
    "ear",
    "face",
    "foot",
    "genital",
    "hand",
    "lower extremity",
    "neck",
    "scalp",
    "trunk",
    "upper extremity",
    "unknown",
)

# Forbidden ground-truth keys that must never influence context evaluation
FORBIDDEN_GROUND_TRUTH_KEYS: tuple[str, ...] = (
    "dx",
    "dx_type",
    "diagnosis",
    "true_label",
    "ground_truth",
    "label",
    "label_name",
    "label_idx",
)


def _is_missing(val: Any) -> bool:
    """Check whether a clinical field value represents missing/null data."""
    if val is None:
        return True
    if isinstance(val, (int, float)):
        return math.isnan(val)
    s = str(val).strip().lower()
    return s in ("", "nan", "none", "null", "unknown", "?", "n/a", "not reported")


def evaluate_context(metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Evaluate patient demographic and anatomical site context.

    Accepts clinical patient metadata and formats a transparent evidence dictionary
    for downstream clinical coordination.

    Design Contract:
    - Missing fields are strictly preserved as None.
    - Missing fields are enumerated in `missing_fields`.
    - No patient information is invented or assumed.
    - Any ground-truth diagnosis keys in input are strictly quarantined and ignored.
    - Output provides objective, factual observational flags without speculative diagnosis.

    Args:
        metadata: Dictionary containing optional demographic and clinical keys:
            - 'age' or 'patient_age' (float | int | str | None)
            - 'sex' or 'gender' (str | None)
            - 'localization', 'site', or 'anatomical_site' (str | None)

    Returns:
        Structured context evaluation dictionary matching the agreed schema:
        {
            "age": float | None,
            "sex": str | None,
            "localization": str | None,
            "missing_fields": list[str],
            "context_flags": list[str],
            "notes": list[str]
        }
    """
    if metadata is None or not isinstance(metadata, dict):
        return {
            "age": None,
            "sex": None,
            "localization": None,
            "missing_fields": ["age", "sex", "localization"],
            "context_flags": ["ALL_CONTEXT_MISSING"],
            "notes": ["Patient context not provided (empty input metadata)."],
        }

    missing_fields: list[str] = []
    context_flags: list[str] = []
    notes: list[str] = []

    # 1. Ground-Truth Quarantine Check
    detected_gt_keys = [k for k in metadata if k.lower() in FORBIDDEN_GROUND_TRUTH_KEYS]
    if detected_gt_keys:
        notes.append(
            f"Audit note: Ground-truth key(s) {detected_gt_keys} detected in input "
            "and strictly quarantined; context agent evaluates demographics only."
        )

    # 2. Extract and Normalize Age
    raw_age = None
    for age_key in ("age", "patient_age", "patientAge"):
        if age_key in metadata:
            raw_age = metadata[age_key]
            break

    parsed_age: Optional[float] = None
    if _is_missing(raw_age):
        missing_fields.append("age")
    else:
        try:
            val = float(raw_age)
            if math.isnan(val) or val < 0.0 or val > 125.0:
                missing_fields.append("age")
                notes.append(f"Recorded age value '{raw_age}' is out of clinical range [0, 125]; marked missing.")
            else:
                parsed_age = round(val, 1)
                notes.append(f"Patient age recorded: {parsed_age} years.")
                if parsed_age < 18.0:
                    context_flags.append("PEDIATRIC_PATIENT")
                    notes.append("Observation: Pediatric patient (<18 yr).")
                elif parsed_age >= 65.0:
                    context_flags.append("GERIATRIC_PATIENT")
                    notes.append("Observation: Geriatric patient (>=65 yr).")
        except (ValueError, TypeError):
            missing_fields.append("age")
            notes.append(f"Recorded age '{raw_age}' could not be parsed as a number; marked missing.")

    # 3. Extract and Normalize Sex
    raw_sex = None
    for sex_key in ("sex", "gender", "patient_sex", "patientGender"):
        if sex_key in metadata:
            raw_sex = metadata[sex_key]
            break

    parsed_sex: Optional[str] = None
    if _is_missing(raw_sex):
        missing_fields.append("sex")
    else:
        s_str = str(raw_sex).strip().lower()
        if s_str in ("m", "male"):
            parsed_sex = "male"
            notes.append("Recorded biological/clinical sex: male.")
        elif s_str in ("f", "female"):
            parsed_sex = "female"
            notes.append("Recorded biological/clinical sex: female.")
        elif s_str in ("other", "non-binary", "intersex"):
            parsed_sex = s_str
            notes.append(f"Recorded sex value: {parsed_sex}.")
        else:
            missing_fields.append("sex")
            notes.append(f"Recorded sex value '{raw_sex}' unconfirmed; marked missing.")

    # 4. Extract and Normalize Anatomical Localization
    raw_loc = None
    for loc_key in ("localization", "site", "anatomical_site", "anatomical_localization"):
        if loc_key in metadata:
            raw_loc = metadata[loc_key]
            break

    parsed_loc: Optional[str] = None
    if _is_missing(raw_loc):
        missing_fields.append("localization")
    else:
        l_str = str(raw_loc).strip().lower()
        if l_str in ("unknown", "not reported", "?"):
            missing_fields.append("localization")
        else:
            parsed_loc = l_str
            notes.append(f"Anatomical site recorded: {parsed_loc}.")
            if parsed_loc in ("face", "scalp", "ear", "neck"):
                context_flags.append("HEAD_NECK_SITE")
            elif parsed_loc in ("acral", "hand", "foot"):
                context_flags.append("ACRAL_SITE")
            elif parsed_loc in ("genital",):
                context_flags.append("GENITAL_SITE")

    # 5. Summarize Completeness Flags
    if len(missing_fields) == 3:
        context_flags.append("ALL_CONTEXT_MISSING")
        notes.append("Patient context not fully provided: age, sex, and localization are all absent.")
    elif len(missing_fields) > 0:
        context_flags.append("INCOMPLETE_CONTEXT")
        notes.append(f"Missing clinical demographic field(s): {', '.join(missing_fields)}.")
    else:
        context_flags.append("CONTEXT_COMPLETE")
        notes.append("Full demographic and anatomical localization context provided.")

    return {
        "age": parsed_age,
        "sex": parsed_sex,
        "localization": parsed_loc,
        "missing_fields": missing_fields,
        "context_flags": context_flags,
        "notes": notes,
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build command-line parser for standalone context evaluation."""
    parser = argparse.ArgumentParser(
        description="Agentic DermaGuard — Context & Demographic Agent (A4)"
    )
    parser.add_argument("--age", type=float, default=None, help="Patient age in years")
    parser.add_argument("--sex", type=str, default=None, help="Patient sex (male, female, other)")
    parser.add_argument("--site", type=str, default=None, help="Anatomical lesion localization")
    return parser


def main() -> None:
    """Execute standalone CLI demonstration of Agent A4."""
    parser = _build_cli_parser()
    args = parser.parse_args()

    meta = {}
    if args.age is not None:
        meta["age"] = args.age
    if args.sex is not None:
        meta["sex"] = args.sex
    if args.site is not None:
        meta["localization"] = args.site

    result = evaluate_context(meta)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
