"""
src/coordinator.py — Agent A6: Multi-Agent Triage Coordinator & Submission Engine.

Key Responsibilities:
1. Orchestrate and synthesize outputs from Agent A1 (Image Quality), Agent A2 (Diagnosis),
   Agent A3 (Lesion Evidence), Agent A4 (Patient Context), and Agent A5 (Safety Auditor).
2. Apply authoritative, auditable clinical triage policy to assign one of three decisions:
   - AUTO-TRIAGE: Low-risk benign lesions with confirmed high image quality and zero safety vetoes.
   - RE-IMAGE: Severely degraded images that fail the optical quality gate.
   - REFER: Malignant, pre-malignant, high-risk, conflicting, or uncertain lesions.
3. Generate and strictly validate the official 3-column hidden-test submission CSV:
   `image_id,predicted_label,decision`

Policy Version: v2.1.0-clinical-safety
Author: Utsavi (Senior Developer — Evaluation & Clinical Safety Track)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

try:
    from src.common import CLASS_NAMES
except ImportError:
    CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

from src.safety_agent import MALIGNANT_CLASSES, BENIGN_CLASSES, audit_safety

POLICY_VERSION: str = "v2.1.0-clinical-safety"

ALLOWED_DECISIONS: List[str] = ["AUTO-TRIAGE", "RE-IMAGE", "REFER"]
SUBMISSION_COLUMNS: List[str] = ["image_id", "predicted_label", "decision"]


def coordinate_triage(agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesize multi-agent outputs and produce the final clinical triage decision.

    Parameters
    ----------
    agent_outputs : dict
        Collection of evidence from all upstream agents (A1 through A5).
        If A5 (Safety Auditor) output is not pre-computed, it will be automatically
        invoked via `audit_safety(agent_outputs)`.

    Returns
    -------
    dict
        Strictly conforms to the Day 2 coordinator contract:
        {
            "predicted_label": str,
            "decision": "AUTO-TRIAGE" | "RE-IMAGE" | "REFER",
            "requires_review": bool,
            "rationale": List[str],
            "agent_evidence": dict,
            "policy_version": "v2.1.0-clinical-safety"
        }
    """
    rationale: List[str] = []

    # --------------------------------------------------------------------------
    # 1. Extract agent evidence
    # --------------------------------------------------------------------------
    a1 = (
        agent_outputs.get("a1_quality")
        or agent_outputs.get("A1")
        or agent_outputs.get("quality")
        or {}
    )
    a2 = (
        agent_outputs.get("a2_diagnosis")
        or agent_outputs.get("A2")
        or agent_outputs.get("diagnosis")
        or {}
    )
    a3 = (
        agent_outputs.get("a3_lesion")
        or agent_outputs.get("A3")
        or agent_outputs.get("lesion_evidence")
        or {}
    )
    a4 = (
        agent_outputs.get("a4_context")
        or agent_outputs.get("A4")
        or agent_outputs.get("patient_context")
        or {}
    )
    a5 = (
        agent_outputs.get("a5_safety")
        or agent_outputs.get("A5")
        or agent_outputs.get("safety")
        or {}
    )

    # If Safety Auditor was not explicitly run, run it now to guarantee safety evaluation
    if not a5:
        a5 = audit_safety(agent_outputs)

    # --------------------------------------------------------------------------
    # 2. Extract predicted diagnosis label
    # --------------------------------------------------------------------------
    predicted_label = a2.get("pred_class") or a2.get("predicted_label") or "nv"
    predicted_label = str(predicted_label).lower().strip()
    if predicted_label not in CLASS_NAMES:
        # Fallback to nv if unrecognized, but note it
        rationale.append(f"Model prediction '{predicted_label}' unmapped; defaulting to canonical nv.")
        predicted_label = "nv"

    # --------------------------------------------------------------------------
    # 3. Apply Decision Policy Hierarchy
    # --------------------------------------------------------------------------
    decision: str
    requires_review: bool

    q_status = (a1.get("overall_status") or a1.get("status") or "").upper().strip()

    # RULE 1: Severe Image Degradation -> RE-IMAGE
    if q_status == "FAIL":
        decision = "RE-IMAGE"
        requires_review = True
        q_reasons = a1.get("reasons", [])
        reasons_str = f" ({', '.join(q_reasons)})" if q_reasons else ""
        rationale.append(
            f"Image failed clinical quality threshold (FAIL){reasons_str}. "
            "Optical artifacts obscure diagnostic microstructure. Re-image requested before clinical evaluation."
        )

    # RULE 2: Malignant or Pre-malignant Lesion -> REFER
    elif predicted_label in MALIGNANT_CLASSES:
        decision = "REFER"
        requires_review = True
        rationale.append(
            f"High-risk diagnosis predicted: '{predicted_label}' ({'Melanoma' if predicted_label == 'mel' else 'Non-melanoma skin cancer'}). "
            "Immediate specialist clinical dermatological referral and biopsy evaluation recommended."
        )

    # RULE 3: Safety Auditor Veto -> REFER
    elif a5.get("veto_auto_triage", False):
        decision = "REFER"
        requires_review = True
        veto_reasons = a5.get("reasons", [])
        conflicts = a5.get("conflicts", [])
        rationale.append("Safety Auditor vetoed automatic disposition.")
        if veto_reasons:
            rationale.append(f"Safety findings: {'; '.join(veto_reasons)}")
        if conflicts:
            rationale.append(f"Agent conflicts detected: {'; '.join(conflicts)}")

    # RULE 4: Image Quality Warning -> REFER with human review
    elif q_status == "WARN":
        decision = "REFER"
        requires_review = True
        rationale.append(
            "Image quality has borderline warnings (WARN). Requires clinical review to confirm diagnostic clarity."
        )

    # RULE 5: Safe Low-Risk AUTO-TRIAGE
    elif (
        predicted_label in BENIGN_CLASSES
        and q_status == "PASS"
        and not a5.get("safety_flag", False)
        and not a5.get("requires_human_review", False)
    ):
        decision = "AUTO-TRIAGE"
        requires_review = False
        confidence = a2.get("confidence")
        conf_str = f" with confidence {float(confidence):.2f}" if confidence is not None else ""
        rationale.append(
            f"Benign lesion ({predicted_label}){conf_str} on verified high-quality dermoscopy (PASS). "
            "Zero high-risk morphological signs, zero clinical safety flags, and no melanoma history. "
            "Safe for standard routine monitoring without immediate specialist intervention."
        )

    # RULE 6: Fallback for all other cases (e.g. df, vasc, uncertainty, missing metadata) -> REFER
    else:
        decision = "REFER"
        requires_review = True
        rationale.append(
            f"Lesion classified as '{predicted_label}'. Precautionary specialist review recommended "
            "due to non-routine class or incomplete low-risk clearance criteria."
        )

    # Document missing patient metadata if reported by A4/A5
    missing_ev = a5.get("missing_evidence", [])
    if missing_ev:
        rationale.append(f"Missing clinical metadata noted: {'; '.join(missing_ev)}")

    # Build structured evidence payload
    agent_evidence = {
        "a1_quality_status": q_status or "UNKNOWN",
        "a2_confidence": a2.get("confidence"),
        "a3_morphology_flags": [k for k, v in a3.items() if bool(v)] if isinstance(a3, dict) else [],
        "a4_missing_fields": [m for m in missing_ev if "patient context" in m],
        "a5_risk_level": a5.get("risk_level", "UNKNOWN"),
    }

    return {
        "predicted_label": predicted_label,
        "decision": decision,
        "requires_review": requires_review,
        "rationale": rationale,
        "agent_evidence": agent_evidence,
        "policy_version": POLICY_VERSION,
    }


# ==============================================================================
# SUBMISSION CSV GENERATION & VALIDATION (TASK 5)
# ==============================================================================

def generate_submission_csv(
    image_ids: List[str],
    predictions: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Generate the official 3-column hidden-test submission CSV.

    Contract: `image_id,predicted_label,decision`

    Parameters
    ----------
    image_ids : List[str]
        Ordered list of image identifiers.
    predictions : List[Dict[str, Any]]
        List of coordinator outputs matching `image_ids`.
    output_path : Optional[Union[str, Path]]
        Destination path to write CSV. If None, DataFrame is returned without saving.

    Returns
    -------
    pd.DataFrame
        Validated 3-column DataFrame.
    """
    if len(image_ids) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(image_ids)} image_ids vs {len(predictions)} predictions."
        )

    rows = []
    for img_id, pred in zip(image_ids, predictions):
        rows.append({
            "image_id": str(img_id).strip(),
            "predicted_label": str(pred.get("predicted_label", "")).strip().lower(),
            "decision": str(pred.get("decision", "")).strip().upper(),
        })

    df = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    validate_submission_csv(df)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)

    return df


def validate_submission_csv(
    df_or_path: Union[pd.DataFrame, str, Path],
    expected_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Strictly validate an official 3-column submission CSV.

    Checks:
    1. Exact 3 columns: `["image_id", "predicted_label", "decision"]`.
    2. Column order exact.
    3. No NaN, None, or empty string values.
    4. No duplicate image_ids.
    5. Exactly 1 row per image.
    6. All predicted_label values belong to the 7 canonical classes.
    7. All decision values belong to {"AUTO-TRIAGE", "RE-IMAGE", "REFER"}.
    8. Row count matches expected_rows (if specified).

    Raises
    ------
    ValueError
        If any submission contract requirement is violated.
    """
    if isinstance(df_or_path, (str, Path)):
        p = Path(df_or_path)
        if not p.is_file():
            raise FileNotFoundError(f"Submission CSV not found at {p}")
        df = pd.read_csv(p)
    elif isinstance(df_or_path, pd.DataFrame):
        df = df_or_path
    else:
        raise TypeError(f"Expected DataFrame, str, or Path; got {type(df_or_path).__name__}")

    # Check columns
    actual_cols = list(df.columns)
    if actual_cols != SUBMISSION_COLUMNS:
        raise ValueError(
            f"Submission CSV columns must be exactly {SUBMISSION_COLUMNS}, got {actual_cols}"
        )

    # Check non-empty
    if len(df) == 0:
        raise ValueError("Submission CSV is empty (0 rows).")

    # Check expected rows if specified
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(
            f"Expected exactly {expected_rows} rows, but submission CSV has {len(df)} rows."
        )

    # Check duplicates
    if df["image_id"].duplicated().any():
        dups = df.loc[df["image_id"].duplicated(), "image_id"].tolist()
        raise ValueError(f"Submission CSV contains duplicate image_ids: {dups[:5]}")

    # Check nulls / blank strings
    for col in SUBMISSION_COLUMNS:
        if df[col].isna().any():
            raise ValueError(f"Submission CSV column '{col}' contains NaN values.")
        blanks = df[col].astype(str).str.strip() == ""
        if blanks.any():
            raise ValueError(f"Submission CSV column '{col}' contains blank/whitespace values.")

    # Check valid labels
    invalid_labels = set(df["predicted_label"]) - set(CLASS_NAMES)
    if invalid_labels:
        raise ValueError(
            f"Invalid predicted_label values found: {invalid_labels}. Allowed: {CLASS_NAMES}"
        )

    # Check valid decisions
    invalid_decisions = set(df["decision"]) - set(ALLOWED_DECISIONS)
    if invalid_decisions:
        raise ValueError(
            f"Invalid decision values found: {invalid_decisions}. Allowed: {ALLOWED_DECISIONS}"
        )

    return df
