"""Agentic DermaGuard — Clinical Dermatoscopy Screening & Multi-Agent Triage Interface.

File: app.py
Author: Aashi (Frontend & Integration Owner)
Project: Agentic DermaGuard Healthcare Hackathon

Architecture:
- Frontend: Streamlit interactive clinical dashboard.
- Backend Integration:
    A1: Image Quality Agent (`src.quality_agent.assess_quality`)
    A2: 7-Class Diagnosis Agent (`src.diagnosis_agent.diagnose` — fallback: NOT AVAILABLE)
    A3: Lesion Morphology Agent (fallback: NOT AVAILABLE)
    A4: Demographic Context Agent (`src.context_agent.evaluate_context`)
    A5: Safety & Regulatory Audit (fallback: NOT AVAILABLE)
    A6: Coordinator & Clinical Triage (fallback: quality-driven RE-IMAGE / PENDING)

Compliance Guarantee:
- No hardcoded fake predictions or probabilities.
- Unavailable agents are transparently marked 'NOT AVAILABLE'.
- Real functions are executed dynamically with full audit evidence.
- Full structured screening report exportable as JSON.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

# Ensure project root is available
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Core data & agent imports
from src.context_agent import evaluate_context
from src.data import CLASS_LABELS, PROJECT_ROOT

# Conditional imports for optional agent modules
try:
    from src.quality_agent import assess_quality
    QUALITY_AGENT_AVAILABLE = True
except (ImportError, Exception):
    QUALITY_AGENT_AVAILABLE = False
    assess_quality = None  # type: ignore[assignment]

# Check if Diagnosis Agent (A2) has an actual implementation
DIAGNOSIS_AGENT_AVAILABLE = False
diagnose_fn = None
try:
    import src.diagnosis_agent as diag_mod
    if hasattr(diag_mod, "diagnose") and callable(diag_mod.diagnose):
        diagnose_fn = diag_mod.diagnose
        DIAGNOSIS_AGENT_AVAILABLE = True
except (ImportError, Exception):
    DIAGNOSIS_AGENT_AVAILABLE = False

# Third-party imaging and frontend libraries
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None
    STREAMLIT_AVAILABLE = False


# -----------------------------------------------------------------------------
# 1. Core Screening Pipeline Engine (Usable from CLI, Streamlit, and Tests)
# -----------------------------------------------------------------------------

def run_screening_pipeline(
    image_input: Union[str, Path, io.BytesIO, bytes],
    context_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Execute the multi-agent screening and triage pipeline on an input image.

    Coordinates all 6 agents (A1-A6), dynamically executing implemented agents
    and cleanly recording 'NOT AVAILABLE' for pending teammate modules.

    Args:
        image_input: Image path, file-like BytesIO, or raw image bytes.
        context_metadata: Dictionary of demographic fields (age, sex, localization).

    Returns:
        Structured clinical evaluation record containing inputs, agent evidence,
        and final coordinator triage decision.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 1. Load and Inspect Image Dimensions
    image_bytes: Optional[bytes] = None
    temp_image_path: Optional[Path] = None
    dimensions: tuple[int, int] = (0, 0)
    image_format: str = "UNKNOWN"

    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            raise FileNotFoundError(f"Input image file not found: {p}")
        temp_image_path = p
        if Image is not None:
            with Image.open(p) as img:
                dimensions = img.size
                image_format = img.format or "JPEG"
    elif isinstance(image_input, (io.BytesIO, bytes)):
        raw_b = image_input.getvalue() if isinstance(image_input, io.BytesIO) else image_input
        image_bytes = raw_b
        if Image is not None:
            with Image.open(io.BytesIO(raw_b)) as img:
                dimensions = img.size
                image_format = img.format or "JPEG"

    # 2. Agent A1: Image Quality Assessment
    quality_result: dict[str, Any]
    if QUALITY_AGENT_AVAILABLE and assess_quality is not None:
        try:
            # If path exists, pass path; otherwise pass bytes or PIL image
            target = str(temp_image_path) if temp_image_path else (image_bytes or image_input)
            quality_result = assess_quality(target)
        except Exception as e:
            quality_result = {
                "status": "ERROR",
                "score": 0.0,
                "reasons": [f"Quality agent execution failed: {e}"],
                "checks": {},
            }
    else:
        quality_result = {
            "status": "NOT AVAILABLE",
            "score": None,
            "reasons": ["Quality agent module not available in active environment."],
            "checks": {},
        }

    # 3. Agent A4: Demographic Context Evaluation
    context_result = evaluate_context(context_metadata or {})

    # 4. Agent A2: 7-Class Diagnosis Agent
    diagnosis_result: dict[str, Any]
    if DIAGNOSIS_AGENT_AVAILABLE and diagnose_fn is not None and temp_image_path is not None:
        try:
            raw_diag = diagnose_fn(str(temp_image_path))
            diagnosis_result = {
                "status": "AVAILABLE",
                "predicted_class": raw_diag.get("class"),
                "confidence": raw_diag.get("confidence"),
                "probabilities": raw_diag.get("probabilities", {}),
            }
        except Exception as e:
            diagnosis_result = {
                "status": "ERROR",
                "error": f"Diagnosis agent execution error: {e}",
                "predicted_class": None,
                "confidence": None,
                "probabilities": {},
            }
    else:
        diagnosis_result = {
            "status": "NOT AVAILABLE",
            "predicted_class": None,
            "confidence": None,
            "probabilities": {},
            "note": "A2 Diagnosis model weights not yet trained or available in repository.",
        }

    # 5. Agent A3: Lesion Morphology
    morphology_result = {
        "status": "NOT AVAILABLE",
        "note": "A3 Lesion Morphology Agent not implemented in current repository.",
    }

    # 6. Agent A5: Safety & Regulatory Audit
    safety_result = {
        "status": "NOT AVAILABLE",
        "note": "A5 Safety Audit Agent not implemented in current repository.",
    }

    # 7. Agent A6: Coordinator & Final Triage Decision
    # Clinical governance:
    # - If Quality is FAIL -> RE-IMAGE (Hard reject; severe degradation)
    # - If Quality is WARN and diagnosis unavailable -> REFER (Quality borderline)
    # - If Quality is PASS and diagnosis unavailable -> PENDING DIAGNOSIS
    q_status = quality_result.get("status")
    final_triage: dict[str, Any]

    if q_status == "FAIL":
        final_triage = {
            "decision": "RE-IMAGE",
            "reasoning": (
                "Severe image quality defects detected by Agent A1. "
                "Clinical diagnosis would be unreliable. Re-acquisition required."
            ),
            "reasons": quality_result.get("reasons", []),
            "action_required": "Request clinician or technician re-capture following quality guidelines.",
        }
    elif q_status == "WARN":
        final_triage = {
            "decision": "REFER",
            "reasoning": (
                "Borderline image quality detected by Agent A1. "
                "Autonomous triage barred; requires human expert dermatological review."
            ),
            "reasons": quality_result.get("reasons", []),
            "action_required": "Refer patient to qualified dermatologist.",
        }
    elif q_status == "PASS" and diagnosis_result["status"] == "NOT AVAILABLE":
        final_triage = {
            "decision": "PENDING DIAGNOSIS",
            "reasoning": (
                "Image quality successfully validated (PASS). "
                "Awaiting Technical Lead (Ilka) diagnosis model training to generate class probabilities."
            ),
            "reasons": [],
            "action_required": "Stand by for production model deployment.",
        }
    elif diagnosis_result["status"] == "AVAILABLE":
        pred_cls = diagnosis_result.get("predicted_class")
        conf = diagnosis_result.get("confidence", 0.0) or 0.0
        if pred_cls in ("mel", "bcc", "akiec") or conf < 0.85:
            decision = "REFER"
            reasoning = f"Suspected high-risk lesion ({pred_cls}) or low confidence ({conf:.2f})."
        else:
            decision = "AUTO-TRIAGE"
            reasoning = f"High confidence ({conf:.2f}) benign prediction ({pred_cls})."

        final_triage = {
            "decision": decision,
            "reasoning": reasoning,
            "reasons": [],
            "action_required": "Standard clinical decision support workflow.",
        }
    else:
        final_triage = {
            "decision": "INCOMPLETE_PIPELINE",
            "reasoning": "Quality check could not be completed.",
            "reasons": quality_result.get("reasons", []),
            "action_required": "Verify input image validity.",
        }

    return {
        "timestamp": timestamp,
        "image_metadata": {
            "dimensions": dimensions,
            "format": image_format,
        },
        "agents": {
            "A1_quality": quality_result,
            "A2_diagnosis": diagnosis_result,
            "A3_morphology": morphology_result,
            "A4_context": context_result,
            "A5_safety": safety_result,
            "A6_coordinator": final_triage,
        },
        "final_decision": final_triage["decision"],
    }


# -----------------------------------------------------------------------------
# 2. Streamlit Application Interface
# -----------------------------------------------------------------------------

def render_streamlit_app() -> None:
    """Render the interactive Streamlit clinical dashboard."""
    if not STREAMLIT_AVAILABLE or st is None:
        print("[!] Streamlit is not installed in the active environment.")
        print("    Install it via: pip install streamlit")
        return

    st.set_page_config(
        page_title="Agentic DermaGuard — Screening & Triage",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔬 Agentic DermaGuard")
    st.caption(
        "Autonomous Multi-Agent Skin Lesion Screening & Clinical Triage | "
        "JAI 2026 International Agentic AI Hackathon"
    )

    # Sidebar: Patient Context & Sample Selection
    st.sidebar.header("📋 Patient Demographics & Site")
    st.sidebar.markdown(
        "Demographic context is evaluated by **Agent A4** to inform the clinical coordinator."
    )

    has_age = st.sidebar.checkbox("Patient age available", value=True)
    patient_age: Optional[float] = None
    if has_age:
        patient_age = float(st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=52))

    patient_sex_input = st.sidebar.selectbox(
        "Biological Sex",
        options=["female", "male", "other", "unknown / not provided"],
        index=0,
    )
    patient_sex: Optional[str] = None if "unknown" in patient_sex_input else patient_sex_input

    anatomical_sites = [
        "back",
        "lower extremity",
        "trunk",
        "upper extremity",
        "abdomen",
        "face",
        "chest",
        "neck",
        "scalp",
        "hand",
        "foot",
        "acral",
        "genital",
        "ear",
        "unknown / not provided",
    ]
    site_input = st.sidebar.selectbox("Anatomical Site / Localization", options=anatomical_sites, index=0)
    patient_site: Optional[str] = None if "unknown" in site_input else site_input

    context_dict = {
        "age": patient_age,
        "sex": patient_sex,
        "localization": patient_site,
    }

    # Sidebar: Image Input Method
    st.sidebar.markdown("---")
    st.sidebar.header("📷 Dermatoscopic Image Input")
    input_method = st.sidebar.radio("Select Source", ["Upload Image File", "Select Validation Sample"])

    uploaded_file = None
    selected_sample_path: Optional[Path] = None

    if input_method == "Upload Image File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Dermatoscopic Image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clinical dermoscopic photograph (JPG/PNG).",
        )
    else:
        val_images_dir = PROJECT_ROOT / "data" / "val" / "images"
        if val_images_dir.is_dir():
            sample_files = sorted([p.name for p in val_images_dir.glob("*.jpg")])[:15]
            if sample_files:
                chosen_name = st.sidebar.selectbox("Choose Sample Image", sample_files)
                selected_sample_path = val_images_dir / chosen_name
            else:
                st.sidebar.warning("No sample images found in data/val/images.")
        else:
            st.sidebar.info("Validation dataset not found locally. Please use file upload.")

    # Main Dashboard Columns
    col_img, col_agents = st.columns([1, 1.4], gap="large")

    image_source = uploaded_file or selected_sample_path

    if image_source is None:
        with col_img:
            st.info("👈 Please upload an image or select a validation sample from the sidebar.")
        with col_agents:
            st.subheader("System Architecture")
            st.markdown(
                """
                | Agent | Role | Status |
                |:---|:---|:---:|
                | **A1: Image Quality** | Sharpness, illumination, glare, obstruction audit | **ONLINE** |
                | **A2: Diagnosis** | 7-class lesion probability distribution | *Awaiting Model* |
                | **A3: Morphology** | Asymmetry, border, pigment structure evidence | *Planned* |
                | **A4: Context** | Demographic normalization & site auditing | **ONLINE** |
                | **A5: Safety** | Uncertainty calibration & regulatory guards | *Planned* |
                | **A6: Coordinator** | Unified clinical triage (`AUTO-TRIAGE` / `REFER` / `RE-IMAGE`) | **ONLINE** |
                """
            )
        return

    # Process and Display Image
    with col_img:
        st.subheader("Dermatoscopic Acquisition")
        try:
            if uploaded_file is not None:
                img_bytes = uploaded_file.getvalue()
                display_img = Image.open(io.BytesIO(img_bytes))
                pipeline_input = img_bytes
            else:
                assert selected_sample_path is not None
                display_img = Image.open(selected_sample_path)
                pipeline_input = selected_sample_path

            st.image(display_img, use_container_width=True, caption="Dermatoscopic Clinical View")
            st.markdown(
                f"**Dimensions:** `{display_img.size[0]} × {display_img.size[1]} px` | "
                f"**Mode:** `{display_img.mode}` | "
                f"**Format:** `{display_img.format or 'JPEG'}`"
            )
        except Exception as e:
            st.error(f"Error reading image: {e}")
            return

    # Execute Multi-Agent Screening
    with st.spinner("Executing Multi-Agent Screening Pipeline..."):
        try:
            results = run_screening_pipeline(pipeline_input, context_metadata=context_dict)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return

    # Render Agent Evidence & Triage in Right Column
    with col_agents:
        st.subheader("Clinical Screening & Triage Evidence")

        # Top Badge: Final Triage Decision
        decision = results["final_decision"]
        if decision == "RE-IMAGE":
            st.error(f"🚨 **FINAL DECISION: {decision}**")
        elif decision == "REFER":
            st.warning(f"⚠️ **FINAL DECISION: {decision}**")
        elif decision == "AUTO-TRIAGE":
            st.success(f"✅ **FINAL DECISION: {decision}**")
        else:
            st.info(f"ℹ️ **FINAL DECISION: {decision}**")

        coord = results["agents"]["A6_coordinator"]
        st.markdown(f"**Clinical Reasoning:** {coord['reasoning']}")
        if coord.get("action_required"):
            st.markdown(f"**Action Required:** `{coord['action_required']}`")

        st.markdown("---")

        # Agent Tabs
        t1, t2, t3, t4 = st.tabs(["Agent A1: Quality", "Agent A4: Context", "Agent A2: Diagnosis", "Full Audit Report"])

        # Tab 1: Agent A1 Quality
        with t1:
            q = results["agents"]["A1_quality"]
            status = q.get("status")
            score = q.get("score")

            if status == "PASS":
                st.success(f"Quality Status: **PASS** (Score: {score:.2f} / 1.00)")
            elif status == "WARN":
                st.warning(f"Quality Status: **WARN** (Score: {score:.2f} / 1.00)")
            else:
                st.error(f"Quality Status: **{status}** (Score: {score if score is not None else 0.0:.2f} / 1.00)")

            if q.get("reasons"):
                st.markdown("**Identified Quality Defect(s):**")
                for r in q["reasons"]:
                    st.markdown(f"- ⚠️ {r}")

            checks = q.get("checks", {})
            if checks:
                st.markdown("**Metric Measurements:**")
                c1, c2 = st.columns(2)
                with c1:
                    if "blur" in checks:
                        st.metric("Blur (Laplacian Var)", f"{checks['blur'].get('value', 0):.1f}", delta=checks['blur'].get('status'))
                    if "illumination" in checks:
                        st.metric("Mean Luminance", f"{checks['illumination'].get('mean_luminance', 0):.1f}", delta=checks['illumination'].get('status'))
                with c2:
                    if "glare" in checks:
                        st.metric("Specular Glare %", f"{checks['glare'].get('glare_percentage', 0):.2f}%", delta=checks['glare'].get('status'))
                    if "obstruction" in checks:
                        st.metric("Obstruction", checks["obstruction"].get("status", "PASS"))

        # Tab 2: Agent A4 Context
        with t2:
            ctx = results["agents"]["A4_context"]
            st.markdown(f"**Recorded Age:** `{ctx.get('age') or 'Missing'}`")
            st.markdown(f"**Recorded Sex:** `{ctx.get('sex') or 'Missing'}`")
            st.markdown(f"**Anatomical Site:** `{ctx.get('localization') or 'Missing'}`")

            if ctx.get("missing_fields"):
                st.info(f"Missing context fields: `{', '.join(ctx['missing_fields'])}`")
            if ctx.get("context_flags"):
                st.markdown(f"**Context Flags:** `{', '.join(ctx['context_flags'])}`")

            st.markdown("**Clinical Notes:**")
            for n in ctx.get("notes", []):
                st.markdown(f"- {n}")

        # Tab 3: Agent A2 Diagnosis
        with t3:
            diag = results["agents"]["A2_diagnosis"]
            if diag["status"] == "NOT AVAILABLE":
                st.info("ℹ️ **Model Status: NOT AVAILABLE**")
                st.markdown(
                    "The production 7-class diagnosis model (EfficientNet-B0 + Logistic Regression) "
                    "has not yet been trained or committed to the repository by Technical Lead Ilka.\n\n"
                    "Per hackathon requirements, mock/fake diagnosis outputs are **strictly barred**."
                )
            elif diag["status"] == "AVAILABLE":
                st.success(f"Predicted Diagnosis: **{diag['predicted_class'].upper()}**")
                st.metric("Top Confidence", f"{diag['confidence']:.2%}")
                st.bar_chart(diag["probabilities"])
            else:
                st.error(diag.get("error", "Diagnosis Error"))

        # Tab 4: Full Audit Report & JSON Export
        with t4:
            st.json(results)
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="📥 Download Clinical Screening Report (JSON)",
                data=json_str,
                file_name=f"screening_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )


# -----------------------------------------------------------------------------
# 3. CLI Runner Fallback
# -----------------------------------------------------------------------------

def _build_cli_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Agentic DermaGuard — Screening Pipeline CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to input dermatoscopic image")
    parser.add_argument("--age", type=float, default=None, help="Patient age")
    parser.add_argument("--sex", type=str, default=None, help="Patient biological sex")
    parser.add_argument("--site", type=str, default=None, help="Anatomical site")
    parser.add_argument("--output-report", type=str, default=None, help="Path to save JSON report")
    return parser


def main() -> None:
    """Entrypoint supporting both CLI execution and Streamlit launch."""
    # Check if executed via 'streamlit run app.py'
    is_streamlit_run = any("streamlit" in arg for arg in sys.argv) or os.getenv("STREAMLIT_RUN") == "1"

    if STREAMLIT_AVAILABLE and (is_streamlit_run or "streamlit" in sys.modules):
        try:
            render_streamlit_app()
            return
        except Exception as e:
            # If not in active streamlit context, proceed to CLI
            pass

    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.image:
        print("=" * 70)
        print("  Agentic DermaGuard — Clinical Screening Pipeline (CLI Mode)")
        print("=" * 70)
        meta = {}
        if args.age is not None:
            meta["age"] = args.age
        if args.sex is not None:
            meta["sex"] = args.sex
        if args.site is not None:
            meta["localization"] = args.site

        res = run_screening_pipeline(args.image, context_metadata=meta)
        out_json = json.dumps(res, indent=2)
        print(out_json)

        if args.output_report:
            out_p = Path(args.output_report).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(out_json, encoding="utf-8")
            print(f"\n[+] Saved report to: {out_p}")
    else:
        print("Agentic DermaGuard — Streamlit Application Entrypoint")
        print("To launch the interactive dashboard, run:")
        print("    streamlit run app.py")


if __name__ == "__main__":
    main()
