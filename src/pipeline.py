"""
src/pipeline.py — End-to-End Multi-Agent Diagnostic & Triage Pipeline.

Orchestrates the complete 6-agent hierarchy for Agentic DermaGuard:
  Agent A1: Image Quality Assessor (src/quality_agent.py)
  Agent A2: Diagnosis Classifier (src/diagnosis_agent.py)
  Agent A3: Lesion Morphology Analyst (src/lesion_agent.py)
  Agent A4: Clinical Context Evaluator (src/context_agent.py)
  Agent A5: Clinical Safety Auditor (src/safety_agent.py)
  Agent A6: Triage Coordinator (src/coordinator.py)

Key Principles:
1. Strict Error Isolation: Upstream agent failures are trapped and converted into
   safe fallbacks (e.g. A1 FAIL triggers RE-IMAGE; unhandled errors trigger REFER).
2. Deterministic Clinical Policy: Decisions are bound to {"AUTO-TRIAGE", "REFER", "RE-IMAGE"}.
3. 3-Column Submission Engine: Produces and validates official hackathon submission CSVs.

Author: Palak (Integration Owner & Agent A6 Coordinator)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Core agent imports
from src.quality_agent import assess_quality
from src.diagnosis_agent import predict as predict_diagnosis
from src.lesion_agent import analyze_lesion
from src.context_agent import evaluate_context
from src.safety_agent import audit_safety
from src.coordinator import (
    ALLOWED_DECISIONS,
    CLASS_NAMES,
    POLICY_VERSION,
    coordinate_triage,
)

logger = logging.getLogger("DermaGuard_Pipeline")

SUBMISSION_COLUMNS: List[str] = ["image_id", "prediction", "decision"]


def run_pipeline(
    image_input: Union[str, Path, Any],
    metadata: Optional[Dict[str, Any]] = None,
    image_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the sequential end-to-end multi-agent clinical triage pipeline.

    Parameters
    ----------
    image_input : Union[str, Path, np.ndarray, Image.Image]
        Path to dermoscopy image file or loaded image representation.
    metadata : Optional[Dict[str, Any]]
        Patient clinical demographic metadata (age, sex, localization).
    image_id : Optional[str]
        Unique image identifier (e.g., 'ISIC_0024306'). Inferred from path if omitted.

    Returns
    -------
    Dict[str, Any]
        Complete synthesized case disposition:
        {
            "image_id": str,
            "prediction": str,
            "predicted_label": str,
            "decision": "AUTO-TRIAGE" | "REFER" | "RE-IMAGE",
            "requires_review": bool,
            "rationale": List[str],
            "agent_evidence": dict,
            "trace": dict,
            "policy_version": "1.0",
            "raw_agent_outputs": dict
        }
    """
    # Infer image_id if path given
    if image_id is None:
        if isinstance(image_input, (str, Path)):
            image_id = Path(image_input).stem
        else:
            image_id = "unknown_image"

    agent_outputs: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # STEP 1: Agent A1 (Image Quality Gate)
    # -------------------------------------------------------------------------
    try:
        a1_result = assess_quality(image_input)
    except Exception as exc:
        logger.warning(f"[{image_id}] A1 Quality Agent raised exception: {exc}")
        a1_result = {
            "overall_status": "WARN",
            "score": 0.0,
            "blur_check": {"passed": False},
            "illumination_check": {"passed": False},
            "glare_check": {"passed": False},
            "obstruction_check": {"passed": False},
            "reasons": [f"A1 quality evaluation exception: {exc}"],
        }
    agent_outputs["a1_quality"] = a1_result

    # -------------------------------------------------------------------------
    # STEP 2: Agent A2 (Diagnosis Classifier)
    # -------------------------------------------------------------------------
    try:
        a2_result = predict_diagnosis(image_input)
    except Exception as exc:
        logger.warning(f"[{image_id}] A2 Diagnosis Agent raised exception: {exc}")
        a2_result = {
            "predicted_class": "unknown",
            "confidence": 0.0,
            "probabilities": {},
            "error": str(exc),
        }
    agent_outputs["a2_diagnosis"] = a2_result

    # -------------------------------------------------------------------------
    # STEP 3: Agent A3 (Lesion Morphology & Evidence)
    # -------------------------------------------------------------------------
    try:
        a3_result = analyze_lesion(image_input)
    except Exception as exc:
        logger.warning(f"[{image_id}] A3 Lesion Agent raised exception: {exc}")
        a3_result = {
            "reliability": "LOW",
            "reliable": False,
            "suspicion_score": 0.0,
            "features": [],
            "flags": {},
            "limitations": [f"A3 lesion analysis exception: {exc}"],
            "error": str(exc),
        }
    agent_outputs["a3_lesion"] = a3_result

    # -------------------------------------------------------------------------
    # STEP 4: Agent A4 (Clinical Context Evaluator)
    # -------------------------------------------------------------------------
    try:
        a4_result = evaluate_context(metadata)
    except Exception as exc:
        logger.warning(f"[{image_id}] A4 Context Agent raised exception: {exc}")
        a4_result = {
            "age": None,
            "sex": None,
            "localization": None,
            "missing_fields": ["age", "sex", "localization"],
            "context_flags": ["CONTEXT_PROCESSING_ERROR"],
            "notes": [f"A4 context evaluation exception: {exc}"],
        }
    agent_outputs["a4_context"] = a4_result

    # -------------------------------------------------------------------------
    # STEP 5: Agent A5 (Clinical Safety Auditor)
    # -------------------------------------------------------------------------
    try:
        a5_result = audit_safety(agent_outputs)
    except Exception as exc:
        logger.warning(f"[{image_id}] A5 Safety Auditor raised exception: {exc}")
        a5_result = {
            "safety_flag": True,
            "veto_auto_triage": True,
            "requires_human_review": True,
            "risk_level": "CRITICAL",
            "reasons": [f"A5 safety auditor execution failure: {exc}"],
            "conflicts": [],
            "missing_evidence": [],
        }
    agent_outputs["a5_safety"] = a5_result

    # -------------------------------------------------------------------------
    # STEP 6: Agent A6 (Coordinator & Triage Decision Engine)
    # -------------------------------------------------------------------------
    coord_result = coordinate_triage(agent_outputs)
    agent_outputs["a6_coordinator"] = coord_result

    pred_label = coord_result.get("predicted_label", "unknown")
    decision = coord_result.get("decision", "REFER")

    return {
        "image_id": image_id,
        "prediction": pred_label,
        "predicted_label": pred_label,
        "decision": decision,
        "requires_review": coord_result.get("requires_review", True),
        "rationale": coord_result.get("rationale", []),
        "agent_evidence": coord_result.get("agent_evidence", {}),
        "trace": coord_result.get("trace", coord_result.get("agent_evidence", {})),
        "policy_version": coord_result.get("policy_version", POLICY_VERSION),
        "raw_agent_outputs": agent_outputs,
    }


def generate_submission_csv(
    results: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    column_name: str = "prediction",
) -> pd.DataFrame:
    """
    Generate and strictly validate the official 3-column submission CSV.

    Columns: `image_id,prediction,decision` (or `image_id,predicted_label,decision`).

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of pipeline or coordinator result dictionaries.
    output_path : Optional[Union[str, Path]]
        Destination path to write CSV. If None, returns DataFrame.
    column_name : str
        Name for the prediction column ('prediction' or 'predicted_label').

    Returns
    -------
    pd.DataFrame
        Validated 3-column submission DataFrame.
    """
    if not results:
        raise ValueError("Results list is empty; cannot generate submission CSV.")

    target_pred_col = column_name.strip()
    columns = ["image_id", target_pred_col, "decision"]

    rows = []
    seen_ids = set()

    for i, res in enumerate(results):
        img_id = str(res.get("image_id", "")).strip()
        if not img_id:
            raise ValueError(f"Missing or blank image_id at record index {i}.")

        if img_id in seen_ids:
            raise ValueError(f"Duplicate image_id '{img_id}' encountered at record index {i}.")
        seen_ids.add(img_id)

        # Retrieve prediction
        pred = (
            res.get("prediction")
            or res.get("predicted_label")
            or res.get("pred_class")
            or ""
        )
        pred_clean = str(pred).strip().lower()

        # Retrieve decision
        dec = str(res.get("decision", "")).strip().upper()

        # Validate decision enum
        if dec not in ALLOWED_DECISIONS:
            raise ValueError(
                f"Invalid decision '{dec}' for image '{img_id}'. Allowed decisions: {ALLOWED_DECISIONS}"
            )

        # If decision is AUTO-TRIAGE, label MUST belong to canonical classes
        if dec == "AUTO-TRIAGE" and pred_clean not in CLASS_NAMES:
            raise ValueError(
                f"AUTO-TRIAGE decision cannot have invalid/unknown prediction '{pred_clean}' for '{img_id}'."
            )

        # For REFER / RE-IMAGE, if prediction is unknown, map to safe default 'nv' if required by strict schema
        if pred_clean not in CLASS_NAMES:
            if pred_clean == "unknown" or not pred_clean:
                # In submission CSVs, label column must be one of 7 classes
                pred_clean = "nv"
            else:
                raise ValueError(
                    f"Invalid class '{pred_clean}' for image '{img_id}'. Allowed: {CLASS_NAMES}"
                )

        rows.append({
            "image_id": img_id,
            target_pred_col: pred_clean,
            "decision": dec,
        })

    df = pd.DataFrame(rows, columns=columns)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        logger.info(f"Saved {len(df)} submission rows to {out_p}")

    return df
