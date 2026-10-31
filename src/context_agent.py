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
    if isinstance(val, bool):
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

    # 2. Extract and Normalize Age (supporting aliases, fallback, and conflict detection)
    age_candidates: list[tuple[str, float]] = []
    age_invalid_notes: list[str] = []

    for age_key in ("age", "patient_age", "patientAge"):
        if age_key in metadata:
            raw_age = metadata[age_key]
            if isinstance(raw_age, bool):
                age_invalid_notes.append(
                    f"Recorded age alias '{age_key}' has boolean value '{raw_age}', which is invalid; ignored."
                )
                continue
            if _is_missing(raw_age):
                continue
            try:
                val = float(raw_age)
                if math.isnan(val) or math.isinf(val) or val < 0.0 or val > 125.0:
                    age_invalid_notes.append(
                        f"Recorded age value '{raw_age}' is out of clinical range [0, 125]; marked missing."
                    )
                else:
                    age_candidates.append((age_key, round(val, 1)))
            except (ValueError, TypeError):
                age_invalid_notes.append(
                    f"Recorded age '{raw_age}' could not be parsed as a number; marked missing."
                )

    parsed_age: Optional[float] = None
    distinct_ages = {v for _, v in age_candidates}
    if len(distinct_ages) > 1:
        conflict_desc = ", ".join(f"{k}={v}" for k, v in age_candidates)
        notes.append(
            f"Conflict / discrepancy detected across multiple age aliases ({conflict_desc}); "
            "safely marked age as missing due to conflicting values."
        )
        missing_fields.append("age")
    elif len(distinct_ages) == 1:
        parsed_age = distinct_ages.pop()
        notes.append(f"Patient age recorded: {parsed_age} years.")
        if parsed_age < 18.0:
            context_flags.append("PEDIATRIC_PATIENT")
            notes.append("Observation: Pediatric patient (<18 yr).")
        elif parsed_age >= 65.0:
            context_flags.append("GERIATRIC_PATIENT")
            notes.append("Observation: Geriatric patient (>=65 yr).")
    else:
        missing_fields.append("age")
        for inv_note in age_invalid_notes:
            notes.append(inv_note)

    # 3. Extract and Normalize Sex (supporting aliases, fallback, and conflict detection)
    sex_candidates: list[tuple[str, str]] = []
    sex_unconfirmed_notes: list[str] = []

    for sex_key in ("sex", "gender", "patient_sex", "patientGender"):
        if sex_key in metadata:
            raw_sex = metadata[sex_key]
            if isinstance(raw_sex, bool) or _is_missing(raw_sex):
                continue
            s_str = str(raw_sex).strip().lower()
            if s_str in ("m", "male"):
                sex_candidates.append((sex_key, "male"))
            elif s_str in ("f", "female"):
                sex_candidates.append((sex_key, "female"))
            elif s_str in ("other", "non-binary", "intersex"):
                sex_candidates.append((sex_key, s_str))
            else:
                sex_unconfirmed_notes.append(
                    f"Recorded sex value '{raw_sex}' unconfirmed; marked missing."
                )

    parsed_sex: Optional[str] = None
    distinct_sexes = {v for _, v in sex_candidates}
    if len(distinct_sexes) > 1:
        conflict_desc = ", ".join(f"{k}={v}" for k, v in sex_candidates)
        notes.append(
            f"Conflict / discrepancy detected across multiple sex aliases ({conflict_desc}); "
            "safely marked sex as missing due to conflicting values."
        )
        missing_fields.append("sex")
    elif len(distinct_sexes) == 1:
        parsed_sex = distinct_sexes.pop()
        if parsed_sex in ("male", "female"):
            notes.append(f"Recorded biological/clinical sex: {parsed_sex}.")
        else:
            notes.append(f"Recorded sex value: {parsed_sex}.")
    else:
        missing_fields.append("sex")
        for u_note in sex_unconfirmed_notes:
            notes.append(u_note)

    # 4. Extract and Normalize Anatomical Localization (supporting aliases, fallback, and conflict detection)
    loc_candidates: list[tuple[str, str]] = []

    for loc_key in ("localization", "site", "anatomical_site", "anatomical_localization"):
        if loc_key in metadata:
            raw_loc = metadata[loc_key]
            if isinstance(raw_loc, bool) or _is_missing(raw_loc):
                continue
            l_str = str(raw_loc).strip().lower()
            if l_str in ("unknown", "not reported", "?"):
                continue
            loc_candidates.append((loc_key, l_str))

    parsed_loc: Optional[str] = None
    distinct_locs = {v for _, v in loc_candidates}
    if len(distinct_locs) > 1:
        conflict_desc = ", ".join(f"{k}={v}" for k, v in loc_candidates)
        notes.append(
            f"Conflict / discrepancy detected across multiple localization aliases ({conflict_desc}); "
            "safely marked localization as missing due to conflicting values."
        )
        missing_fields.append("localization")
    elif len(distinct_locs) == 1:
        parsed_loc = distinct_locs.pop()
        notes.append(f"Anatomical site recorded: {parsed_loc}.")
        if parsed_loc in ("face", "scalp", "ear", "neck"):
            context_flags.append("HEAD_NECK_SITE")
        elif parsed_loc in ("acral", "hand", "foot"):
            context_flags.append("ACRAL_SITE")
        elif parsed_loc in ("genital",):
            context_flags.append("GENITAL_SITE")
    else:
        missing_fields.append("localization")

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
