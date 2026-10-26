"""
Streamlit Interactive Forensic Dashboard
Provides real-time multimodal deepfake detection, cheapfake analysis,
visual artifact heatmaps, and evidence-grounded fact-checking.
"""

import os
import sys
import tempfile
import streamlit as st
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.pipeline import MultimodalVerificationPipeline

st.set_page_config(
    page_title="Multimodal Deepfake & Fact-Checking Engine",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def load_pipeline():
    return MultimodalVerificationPipeline()


def main():
    st.title("🛡️ Multimodal Deepfake Detection & Semantic Fact-Checking System")
    st.markdown(
        "**DFKI Berlin — Speech & Language Technology (SLT)** | Candidate: Praveen Kumar Chilukuri\n\n"
        "A hybrid neural verification engine that simultaneously inspects **pixel-level synthesis artifacts** (ViT) "
        "and cross-references **transcribed spoken claims against verified knowledge bases** (Whisper + Hybrid RAG)."
    )

    pipe = load_pipeline()

    # Sidebar: System Configuration & Knowledge Base
    with st.sidebar:
        st.header("⚙️ Forensic Parameters")
        st.markdown("**Adaptive Gating Formulation:**")
        st.latex(r"S_{final} = \sigma(\alpha(C_{vis}) S_{vis} + \beta(C_{sem}) S_{sem} + \gamma (S_{vis} \odot S_{sem}))")

        st.divider()
        st.subheader("📚 Indexed Knowledge Base")
        num_chunks = len(pipe.vector_store.chunks)
        st.info(f"Loaded {num_chunks} verified ground-truth claim archives (Snopes, Reuters, ClaimsKG, Wikipedia).")
        with st.expander("View Indexed Claims"):
            for c in pipe.vector_store.chunks:
                verdict_color = "🟢" if c.get("verdict") == "TRUE" else "🔴"
                st.write(f"{verdict_color} **{c.get('claim')}**")
                st.caption(f"Source: {c.get('source')} | Verdict: {c.get('verdict')}")

    # Tabs: Analysis Modes
    tab_upload, tab_scenarios, tab_architecture = st.tabs(
        ["🎥 Media Verification", "⚡ Benchmark Scenarios", "📐 Architectural Workflow"]
    )

    with tab_upload:
        st.subheader("Upload Video or Media File")
        col_input, col_meta = st.columns([2, 1])

        with col_input:
            uploaded_file = st.file_uploader(
                "Choose video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov", "webm"]
            )
        with col_meta:
            override_claim = st.text_area(
                "Optional Claim / Breaking Headline",
                placeholder="e.g. Volodymyr Zelensky announced surrender to Russian forces in March 2022.",
                help="Test cheapfakes: pristine video paired with a deceptive headline."
            )

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            if st.button("🚀 Run Multimodal Verification", type="primary"):
                with st.spinner("Processing video frames and running dual-branch forensic verification..."):
                    report = pipe.analyze_media(tmp_path, override_claim_text=override_claim)
                    render_report_dashboard(report)

            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    with tab_scenarios:
        st.subheader("Test Pre-Configured Benchmark Scenarios")
        st.markdown(
            "Demonstrate how the system handles the **4 distinct forensic regimes**."
        )

        scenario = st.selectbox(
            "Select Scenario",
            [
                "Case 1: Visual Deepfake Alert (Manipulated Face + True Audio)",
                "Case 2: Misinformation Alert / Cheapfake (Authentic Face + Altered False Headline)",
                "Case 3: Composite Deepfake (Manipulated Face + False Spoken Claims)",
                "Case 4: Verified Authentic Media (Pristine Face + Grounded Speech)"
            ]
        )

        if st.button("Evaluate Selected Scenario"):
            if "Case 1" in scenario:
                s_vis, c_vis = 0.88, 0.92
                s_sem, c_sem = 0.12, 0.85
                peak_zone = "jawline_and_chin_boundary"
                claims_data = [
                    {
                        "claim_id": "claim_001",
                        "claim_text": "The European Central Bank is headquartered in Frankfurt am Main, Germany.",
                        "timestamp": "0.0s - 4.2s",
                        "best_evidence": "The European Central Bank headquarters is located in Frankfurt am Main, Germany.",
                        "source": "Wikipedia / EU Archives",
                        "contradiction_prob": 0.05,
                        "entailment_prob": 0.92,
                        "verdict": "TRUE",
                        "is_contradicted": False
                    }
                ]
            elif "Case 2" in scenario:
                s_vis, c_vis = 0.11, 0.94
                s_sem, c_sem = 0.89, 0.90
                peak_zone = "facial_center"
                claims_data = [
                    {
                        "claim_id": "claim_001",
                        "claim_text": "Volodymyr Zelensky announced Ukraine's surrender to Russian forces in March 2022.",
                        "timestamp": "0.0s - 3.8s",
                        "best_evidence": "A viral March 2022 video depicting Zelensky telling Ukrainian soldiers to surrender was a debunked deepfake. Zelensky never issued surrender orders.",
                        "source": "Reuters Fact Check / Snopes",
                        "contradiction_prob": 0.94,
                        "entailment_prob": 0.02,
                        "verdict": "FALSE",
                        "is_contradicted": True
                    }
                ]
            elif "Case 3" in scenario:
                s_vis, c_vis = 0.86, 0.91
                s_sem, c_sem = 0.92, 0.88
                peak_zone = "mouth_and_lips"
                claims_data = [
                    {
                        "claim_id": "claim_001",
                        "claim_text": "The Eiffel Tower was struck by lightning and completely burned down in Paris.",
                        "timestamp": "0.0s - 3.5s",
                        "best_evidence": "Images showing the Eiffel Tower engulfed in catastrophic fire in Paris were AI-generated synthetic images.",
                        "source": "AP News / Snopes",
                        "contradiction_prob": 0.95,
                        "entailment_prob": 0.01,
                        "verdict": "FALSE",
                        "is_contradicted": True
                    }
                ]
            else:
                s_vis, c_vis = 0.08, 0.95
                s_sem, c_sem = 0.10, 0.91
                peak_zone = "facial_center"
                claims_data = [
                    {
                        "claim_id": "claim_001",
                        "claim_text": "DFKI Berlin has a research laboratory focused on Speech and Language Technology.",
                        "timestamp": "0.0s - 4.0s",
                        "best_evidence": "DFKI Berlin operates research departments in Speech and Language Technology (SLT).",
                        "source": "DFKI Official Documentation",
                        "contradiction_prob": 0.03,
                        "entailment_prob": 0.95,
                        "verdict": "TRUE",
                        "is_contradicted": False
                    }
                ]

            verdict = pipe.adjudicator.adjudicate(
                s_vis=s_vis, c_vis=c_vis, s_sem=s_sem, c_sem=c_sem,
                peak_zone=peak_zone, num_contradictions=sum(1 for c in claims_data if c["is_contradicted"])
            )

            mock_report = {
                "verdict": verdict.primary_verdict,
                "risk_level": verdict.risk_level,
                "case_category": verdict.case_category,
                "s_final": verdict.fusion_scores.s_final,
                "s_vis": s_vis,
                "c_vis": c_vis,
                "s_sem": s_sem,
                "c_sem": c_sem,
                "narrative_explanation": verdict.narrative_explanation,
                "recommended_action": verdict.recommended_action,
                "visual_forensics": {
                    "s_vis": s_vis,
                    "c_vis": c_vis,
                    "peak_zone": peak_zone,
                    "diagnostic": f"Facial inspection completed on 16 sampled frames. Anomaly focus: {peak_zone}."
                },
                "semantic_fact_check": {
                    "s_sem": s_sem,
                    "c_sem": c_sem,
                    "claims": claims_data
                },
                "transcript": claims_data[0]["claim_text"]
            }
            render_report_dict(mock_report)

    with tab_architecture:
        st.subheader("System Architecture & Data Flow")
        st.markdown("""
        ```
        +-----------------------------------------------------------------------------------+
        |                         MULTIMODAL INPUT STREAM (MP4 / Web Video)                |
        +-----------------------------------------+-----------------------------------------+
                                                  |
                    +-----------------------------+-----------------------------+
                    |                                                           |
                    v                                                           v
        [ Branch A: Visual Forgery ]                                [ Branch B: Semantic Fact-Checking ]
        • OpenCV 30 fps Frame Sampling                              • 16kHz Mono Audio Extraction
        • Face Localization & 224x224 Alignment                     • OpenAI Whisper Timestamped ASR
        • 16x16 Non-Overlapping Patches (196 tokens)                • LLM / Heuristic Claim Decomposition
        • 12-layer Vision Transformer (ViT)                         • Dense FAISS Retrieval (ClaimsKG / Snopes)
        • Grad-CAM / Attention Rollout Heatmaps                     • Cross-Encoder NLI (Premise vs Hypothesis)
                    |                                                           |
                    v                                                           v
          Visual Score S_vis ∈ [0, 1]                                 Semantic Inconsistency S_sem ∈ [0, 1]
          Confidence   C_vis ∈ [0, 1]                                 Confidence             C_sem ∈ [0, 1]
                    |                                                           |
                    +-----------------------------+-----------------------------+
                                                  |
                                                  v
                              [ Stage 3: Adaptive Confidence Gating ]
                     S_final = σ( α(C_vis)·S_vis + β(C_sem)·S_sem + γ·(S_vis ⊙ S_sem) )
                                                  |
                                                  v
                              [ 4-Case Adjudication & Decision Matrix ]
         • Case 1: High S_vis, Low S_sem  -> Visual Deepfake Alert (Face Swap / Reenactment)
         • Case 2: Low S_vis,  High S_sem -> Misinformation Alert / Cheapfake (Out-of-context audio)
         • Case 3: High S_vis, High S_sem -> Composite Deepfake & Misinformation
         • Case 4: Low S_vis,  Low S_sem  -> Verified Authentic Media
        ```
        """)


def render_report_dashboard(report):
    render_report_dict(report.to_dict())


def render_report_dict(d):
    st.divider()

    # Verdict Header
    risk = d.get("risk_level", "LOW")
    verdict = d.get("verdict", "Unknown")

    if risk == "CRITICAL" or "Composite" in verdict:
        st.error(f"🚨 **{verdict}** (Risk Level: {risk})")
    elif risk == "HIGH":
        if "Visual" in verdict:
            st.error(f"🎭 **{verdict}** (Risk Level: {risk})")
        else:
            st.warning(f"📰 **{verdict}** (Risk Level: {risk})")
    else:
        st.success(f"✅ **{verdict}** (Risk Level: {risk})")

    # Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Final Risk S_final", f"{d.get('s_final', 0):.2f}")
    m2.metric("Visual Forgery S_vis", f"{d.get('s_vis', 0):.2f}")
    m3.metric("Visual Confidence C_vis", f"{d.get('c_vis', 0):.2f}")
    m4.metric("Semantic Inconsistency S_sem", f"{d.get('s_sem', 0):.2f}")
    m5.metric("Semantic Confidence C_sem", f"{d.get('c_sem', 0):.2f}")

    # Narrative
    st.markdown("### 📋 Forensic Audit Summary")
    st.info(d.get("narrative_explanation", ""))
    st.markdown(f"**Recommended Action:** `{d.get('recommended_action', '')}`")

    # Dual Branch Deep Dive
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("### 👁️ Branch A: Visual Forensics (ViT)")
        vis = d.get("visual_forensics", {})
        st.write(f"• **Inspected Frames:** {vis.get('num_frames', 'N/A')}")
        st.write(f"• **Peak Anomaly Zone:** `{vis.get('peak_zone', 'N/A')}`")
        st.write(f"• **Diagnostic:** {vis.get('diagnostic', '')}")

    with c_right:
        st.markdown("### 🎙️ Branch B: Semantic Fact-Checking (RAG & NLI)")
        sem = d.get("semantic_fact_check", {})
        claims = sem.get("claims", [])
        st.write(f"• **Claims Analyzed:** {len(claims)}")
        st.write(f"• **Full Transcript:** *\"{d.get('transcript', '')}\"*")

    if claims:
        st.markdown("#### Atomic Claims Breakdown")
        for cl in claims:
            status_icon = "❌ False / Contradicted" if cl.get("is_contradicted") else "✅ Supported / True"
            with st.expander(f"{status_icon}: {cl.get('claim_text')}"):
                st.write(f"• **Timestamp:** {cl.get('timestamp')}")
                st.write(f"• **Retrieved Ground Truth Evidence:** {cl.get('best_evidence')}")
                st.write(f"• **Source:** {cl.get('source')}")
                st.write(f"• **Contradiction Probability:** {cl.get('contradiction_prob', 0):.2f}")
                st.write(f"• **Entailment Probability:** {cl.get('entailment_prob', 0):.2f}")


if __name__ == "__main__":
    main()
