"""
src/safety_agent.py — Agent A5: Clinical Safety Auditor for Agentic DermaGuard.

Key Responsibilities:
1. Inspect outputs from Agent A1 (Image Quality), Agent A2 (Diagnosis),
   Agent A3 (Lesion Evidence), and Agent A4 (Patient Context).
2. Screen for malignancy, clinical red flags, and diagnostic uncertainty.
3. Detect conflicting evidence between agents (e.g. benign model prediction
   vs atypical clinical dermoscopy features).
4. Identify missing evidence and incomplete clinical metadata.
5. Exercise independent clinical veto authority to prevent unsafe automatic
   low-risk triage dispositions.

Author: Utsavi (Senior Developer — Evaluation & Clinical Safety Track)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

try:
    from src.common import CLASS_NAMES
except ImportError:
    CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Malignant / pre-malignant categories that must NEVER receive automatic low-risk discharge
MALIGNANT_CLASSES: Set[str] = {"mel", "bcc", "akiec"}

# Benign categories eligible for safe monitoring if all safety criteria are satisfied
BENIGN_CLASSES: Set[str] = {"nv", "bkl"}

# High-risk dermoscopic and clinical morphology features
HIGH_RISK_MORPHOLOGY_KEYS: Set[str] = {
    "ulceration",
    "bleeding",
    "rapid_evolution",
    "rapid_change",
    "blue_white_veil",
    "atypical_network",
    "atypical_pigment_network",
    "irregular_streaks",
}


def audit_safety(agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Audit multi-agent evidence and determine whether safety vetoes apply.

    Parameters
    ----------
    agent_outputs : dict
        Dictionary containing evidence from upstream agents. Supported keys:
        - "a1_quality" / "A1" / "quality": output from quality_agent.py
        - "a2_diagnosis" / "A2" / "diagnosis": output from diagnosis pipeline
        - "a3_lesion" / "A3" / "lesion_evidence": clinical dermoscopic signs
        - "a4_context" / "A4" / "patient_context": metadata (age, sex, history)

    Returns
    -------
    dict
        Structured clinical safety audit containing:
        - "safety_flag": bool (True if any safety concern was identified)
        - "veto_auto_triage": bool (True if AUTO-TRIAGE is strictly forbidden)
        - "requires_human_review": bool (True if a human clinician must review)
        - "reasons": List[str] (auditable list of safety findings)
        - "conflicts": List[str] (detected conflicts between agent observations)
        - "missing_evidence": List[str] (unsupplied or incomplete fields)
        - "risk_level": "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
    """
    reasons: List[str] = []
    conflicts: List[str] = []
    missing_evidence: List[str] = []

    safety_flag: bool = False
    veto_auto_triage: bool = False
    requires_human_review: bool = False

    # --------------------------------------------------------------------------
    # 1. Normalize agent inputs
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

    # --------------------------------------------------------------------------
    # 2. Check for missing critical agents
    # --------------------------------------------------------------------------
    if not a1:
        missing_evidence.append("A1 (Image Quality Agent) output is missing.")
        safety_flag = True
        veto_auto_triage = True
        requires_human_review = True

    if not a2:
        missing_evidence.append("A2 (Diagnosis Agent) output is missing.")
        safety_flag = True
        veto_auto_triage = True
        requires_human_review = True

    # --------------------------------------------------------------------------
    # 3. Audit Agent A1 (Image Quality)
    # --------------------------------------------------------------------------
    q_status = a1.get("overall_status") or a1.get("status")
    if q_status == "FAIL":
        safety_flag = True
        veto_auto_triage = True
        requires_human_review = True
        q_reasons = a1.get("reasons", [])
        reasons.append(
            f"Image quality gate failed (FAIL). Reasons: {', '.join(q_reasons) if q_reasons else 'unspecified artifacts'}."
        )
    elif q_status == "WARN":
        safety_flag = True
        requires_human_review = True
        reasons.append("Image quality has warnings (WARN); requires human verification.")
    elif q_status is None and a1:
        missing_evidence.append("A1 quality output lacks overall_status.")
        veto_auto_triage = True

    # --------------------------------------------------------------------------
    # 4. Audit Agent A2 (Diagnostic Model)
    # --------------------------------------------------------------------------
    pred_class = a2.get("pred_class") or a2.get("predicted_label")
    confidence = a2.get("confidence")
    probabilities = a2.get("probabilities") or {}

    if pred_class:
        pred_class = str(pred_class).lower().strip()

    if pred_class in MALIGNANT_CLASSES:
        safety_flag = True
        veto_auto_triage = True
        requires_human_review = True
        reasons.append(
            f"Malignant or pre-malignant lesion predicted: '{pred_class}'. Specialist referral required."
        )

    # Inspect probabilities if available
    prob_mel = probabilities.get("prob_mel") or probabilities.get("mel")
    if prob_mel is not None:
        try:
            prob_mel = float(prob_mel)
            # If melanoma probability is substantial (> 0.20) or secondary highest
            if prob_mel >= 0.20 and pred_class != "mel":
                safety_flag = True
                veto_auto_triage = True
                requires_human_review = True
                reasons.append(
                    f"Substantial secondary melanoma probability ({prob_mel:.3f}) despite '{pred_class}' prediction."
                )
        except (ValueError, TypeError):
            pass

    # Inspect diagnostic uncertainty
    if confidence is not None:
        try:
            conf_val = float(confidence)
            if conf_val < 0.60:
                safety_flag = True
                veto_auto_triage = True
                requires_human_review = True
                reasons.append(
                    f"Low diagnostic confidence ({conf_val:.3f} < 0.600); automated disposition unsafe."
                )
        except (ValueError, TypeError):
            pass

    # --------------------------------------------------------------------------
    # 5. Audit Agent A3 (Lesion Evidence) & Cross-Agent Conflicts
    # --------------------------------------------------------------------------
    detected_red_flags: List[str] = []
    if isinstance(a3, dict):
        for k, v in a3.items():
            k_norm = str(k).lower().strip()
            if k_norm in HIGH_RISK_MORPHOLOGY_KEYS and bool(v):
                detected_red_flags.append(k_norm)

    if detected_red_flags:
        safety_flag = True
        veto_auto_triage = True
        requires_human_review = True
        reasons.append(
            f"High-risk lesion morphological features detected: {', '.join(detected_red_flags)}."
        )
        # Check conflict with A2
        if pred_class in BENIGN_CLASSES:
            conflicts.append(
                f"Diagnostic conflict: A2 model predicted benign '{pred_class}', "
                f"but clinical feature inspection revealed high-risk signs ({', '.join(detected_red_flags)})."
            )

    # --------------------------------------------------------------------------
    # 6. Audit Agent A4 (Patient Context)
    # --------------------------------------------------------------------------
    if isinstance(a4, dict):
        # High-risk personal or family history
        hist_personal = a4.get("personal_history_melanoma") or a4.get("history_melanoma")
        hist_family = a4.get("family_history_melanoma")
        immuno = a4.get("immunosuppressed")

        if hist_personal:
            safety_flag = True
            veto_auto_triage = True
            requires_human_review = True
            reasons.append("Patient has personal history of melanoma.")

        if hist_family:
            safety_flag = True
            veto_auto_triage = True
            requires_human_review = True
            reasons.append("Patient has family history of melanoma.")

        if immuno:
            safety_flag = True
            veto_auto_triage = True
            requires_human_review = True
            reasons.append("Patient is immunocompromised.")

        # Check for missing metadata fields
        for field in ("age", "sex", "localization"):
            val = a4.get(field)
            if val is None or str(val).strip().lower() in {"", "unknown", "nan", "none"}:
                missing_evidence.append(f"Missing patient context field: '{field}'.")

    # --------------------------------------------------------------------------
    # 7. Synthesize Overall Risk Level
    # --------------------------------------------------------------------------
    if pred_class == "mel" or "ulceration" in detected_red_flags or (a4.get("personal_history_melanoma") and pred_class in MALIGNANT_CLASSES):
        risk_level = "CRITICAL"
    elif pred_class in MALIGNANT_CLASSES or veto_auto_triage:
        risk_level = "HIGH"
    elif safety_flag or requires_human_review:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "safety_flag": safety_flag,
        "veto_auto_triage": veto_auto_triage,
        "requires_human_review": requires_human_review,
        "reasons": reasons,
        "conflicts": conflicts,
        "missing_evidence": missing_evidence,
        "risk_level": risk_level,
    }
