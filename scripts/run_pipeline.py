"""
Command Line Interface (CLI) Entry Point
Runs multimodal deepfake forensics and semantic fact-checking on media files.
"""

import argparse
import json
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import MultimodalVerificationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Deepfake Detection & Semantic Fact-Checking System (DFKI Berlin)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to target MP4/AVI/MOV video file"
    )
    parser.add_argument(
        "--claim",
        type=str,
        default=None,
        help="Optional breaking headline / claim to test cheapfake recontextualization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to export verification report JSON"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run self-contained synthetic benchmark scenario demonstration"
    )

    args = parser.parse_args()

    print("=" * 80)
    print(" MULTIMODAL DEEPFAKE FORENSICS & SEMANTIC FACT-CHECKING SYSTEM")
    print(" DFKI Berlin — Speech & Language Technology | Praveen Kumar Chilukuri")
    print("=" * 80)

    pipe = MultimodalVerificationPipeline(config_path=args.config)

    if args.demo or not args.video:
        print("\n[INFO] Running Case 2 (Cheapfake) Benchmark Demonstration...")
        # Evaluate representative cheapfake scenario
        verdict = pipe.adjudicator.adjudicate(
            s_vis=0.12,
            c_vis=0.92,
            s_sem=0.88,
            c_sem=0.90,
            peak_zone="facial_center",
            peak_timestamp=1.5,
            num_contradictions=1
        )
        print(f"\n>> PRIMARY VERDICT       : {verdict.primary_verdict}")
        print(f">> CASE CATEGORY         : {verdict.case_category}")
        print(f">> RISK LEVEL            : {verdict.risk_level}")
        print(f">> FINAL RISK SCORE S_fin: {verdict.fusion_scores.s_final:.4f}")
        print(f">> VISUAL FORGERY S_vis  : {verdict.fusion_scores.s_vis:.4f} (Conf: {verdict.fusion_scores.c_vis:.4f})")
        print(f">> SEMANTIC INCONS S_sem : {verdict.fusion_scores.s_sem:.4f} (Conf: {verdict.fusion_scores.c_sem:.4f})")
        print(f"\n[NARRATIVE REPORT]\n{verdict.narrative_explanation}")
        print(f"\n[RECOMMENDED ACTION]\n{verdict.recommended_action}")
        print("=" * 80)
        return

    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found at: {args.video}")
        sys.exit(1)

    print(f"\n[+] Analyzing media file: {args.video}")
    if args.claim:
        print(f"[+] Attached claim text : \"{args.claim}\"")

    report = pipe.analyze_media(args.video, override_claim_text=args.claim)

    print(f"\n>> PRIMARY VERDICT       : {report.verdict}")
    print(f">> CASE CATEGORY         : {report.case_category}")
    print(f">> RISK LEVEL            : {report.risk_level}")
    print(f">> FINAL RISK SCORE S_fin: {report.s_final:.4f}")
    print(f">> VISUAL FORGERY S_vis  : {report.s_vis:.4f} (Conf: {report.c_vis:.4f})")
    print(f">> SEMANTIC INCONS S_sem : {report.s_sem:.4f} (Conf: {report.c_sem:.4f})")
    print(f"\n[NARRATIVE REPORT]\n{report.narrative_explanation}")
    print(f"\n[RECOMMENDED ACTION]\n{report.recommended_action}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\n[+] Report exported successfully to: {args.output_json}")

    print("=" * 80)


if __name__ == "__main__":
    main()
