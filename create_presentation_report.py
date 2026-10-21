"""
COGAIT-VISION: Presentation Report Document Generator
Generates a comprehensive, professional PDF presentation document covering:
- Project Overview & Clinical Motivation
- How to Use the Prototype (Step 1 -> Step 2 -> Step 3)
- Algorithmic Foundations (MediaPipe, Kinematics, DTC, Tri-Model ML Ensemble)
- Commands to Run & Deployment Instructions
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_presentation_document(output_filename: str = "presentation_report_document.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0f2942")     # Deep Clinical Navy
    ACCENT = colors.HexColor("#2563eb")      # Medical Blue
    SECONDARY = colors.HexColor("#475569")   # Slate
    SUCCESS = colors.HexColor("#059669")     # Medical Green
    WARNING = colors.HexColor("#dc2626")     # Alert Red
    LIGHT_BG = colors.HexColor("#f8fafc")    # Off-white / light slate
    CARD_BORDER = colors.HexColor("#cbd5e1") # Border grey

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=ACCENT,
    )
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=SECONDARY,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=ACCENT,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
    )
    body_bold = ParagraphStyle(
        "BodyBold_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
    )
    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=12,
        firstLineIndent=-8,
    )

    elements = []

    # =========================================================================
    # HEADER & DOCUMENT COVER BANNER
    # =========================================================================
    elements.append(Paragraph("<b>COGAIT-VISION</b> | PRESENTATION REPORT DOCUMENT", title_style))
    elements.append(Spacer(1, 3))
    elements.append(Paragraph("Clinical AI Gait & Markerless TUG Screening System for Early Alzheimer's Detection", subtitle_style))
    elements.append(Spacer(1, 4))
    elements.append(HRFlowable(width="100%", thickness=2, color=ACCENT, spaceAfter=8))

    meta_table_data = [
        [
            Paragraph("<b>Project Investigators:</b><br/>B P S Kruthi • Ch. Karthika • A. Laxmi Sahithi", meta_style),
            Paragraph("<b>Project Guide:</b><br/>Ms. B K Aishwarya, Assistant Professor", meta_style),
            Paragraph("<b>SDG Alignment & Domain:</b><br/>UN SDG 3 (Good Health) & SDG 9 (Innovation)", meta_style),
            Paragraph(f"<b>System Status:</b><br/>Operational (Port 8501)", meta_style),
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[150, 130, 150, 102])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
        ('BOX', (0, 0), (-1, -1), 1, CARD_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10))

    # =========================================================================
    # 1. EXECUTIVE SUMMARY & CLINICAL MOTIVATION
    # =========================================================================
    elements.append(Paragraph("1. Executive Summary & Clinical Rationale", h1_style))
    elements.append(Paragraph(
        "Alzheimer's Disease (AD) and Mild Cognitive Impairment (MCI) represent major global healthcare challenges. "
        "Historically, cognitive decline has been diagnosed through pen-and-paper assessments (e.g., MMSE, MoCA) or expensive "
        "imaging (PET/MRI). However, emerging clinical neuroscience shows that subtle motor-cognitive changes appear years "
        "before clinical dementia manifests. Under <b>dual-task cognitive loading</b> (e.g., serial subtraction while walking), "
        "prefrontal attentional networks compete with motor planning circuits, causing executive motor deceleration. "
        "<b>COGAIT-VISION</b> provides a 100% markerless, camera-based Timed Up and Go (TUG) screening solution that quantifies "
        "this cognitive-motor interference using standard RGB webcams, achieving hospital-grade diagnostic screening with zero wearable sensors.",
        body_style
    ))
    elements.append(Spacer(1, 10))

    # =========================================================================
    # 2. STEP-BY-STEP PROTOTYPE USER MANUAL ("HOW TO USE")
    # =========================================================================
    elements.append(Paragraph("2. Prototype User Manual: Step-by-Step Clinical Workflow", h1_style))
    
    workflow_steps = [
        [
            Paragraph("<b>Workflow Phase</b>", body_bold),
            Paragraph("<b>User Action & Protocol</b>", body_bold),
            Paragraph("<b>Automated System Response</b>", body_bold),
        ],
        [
            Paragraph("<b>Step 1: Camera Scan<br/>(TUG Test)</b>", body_style),
            Paragraph("1. Grant webcam privacy authorization checkbox.<br/>"
                      "2. Enter Patient Full Name & ID.<br/>"
                      "3. Click <b>'🟢 Turn Camera ON & Start TUG Scan'</b>.<br/>"
                      "4. Perform standard TUG: Sit -> Stand -> Walk 3m -> Turn 180° -> Walk back -> Sit down.", body_style),
            Paragraph("• Markerless MediaPipe tracks 33 3D skeletal joints at 30 FPS.<br/>"
                      "• Real-time HUD displays knee flexion, cadence, and stopwatch.<br/>"
                      "• Auto-completes upon sitting down in Phase 6.<br/>"
                      "• <b>Instantly auto-redirects to Step 2!</b>", body_style),
        ],
        [
            Paragraph("<b>Step 2: Medical<br/>Questionnaire</b>", body_style),
            Paragraph("1. Review connected live gait biomarker banner at the top.<br/>"
                      "2. Verify patient profile demographics.<br/>"
                      "3. Check relevant cognitive symptom checklist boxes.<br/>"
                      "4. Check any orthopedic confounders (surgery, cane).<br/>"
                      "5. Click <b>'👉 Generate Diagnostic Report'</b>.", body_style),
            Paragraph("• Telemetry automatically binds live duration, speed, cadence, and DTC.<br/>"
                      "• Evaluates multi-modal risk scoring.<br/>"
                      "• Calculates cognitive concern vs. orthopedic confounder scores.<br/>"
                      "• Automatically transitions directly to Step 3.", body_style),
        ],
        [
            Paragraph("<b>Step 3: Diagnostic<br/>Screening Report</b>", body_style),
            Paragraph("1. Inspect executive clinical verdict card.<br/>"
                      "2. Review 4 core biomarker metric cards.<br/>"
                      "3. Analyze quantitative spatio-temporal comparison matrix.<br/>"
                      "4. Inspect Tri-Model AI breakdown (SVM, RF, XGBoost).<br/>"
                      "5. Click <b>'📥 Download Official Clinical PDF'</b>.", body_style),
            Paragraph("• Tri-model supervised ensemble consensus verdict.<br/>"
                      "• Full clinical advice & normative reference comparison.<br/>"
                      "• Generates downloadable hospital-grade PDF report with Patient Name and verified parameters.", body_style),
        ],
    ]
    wf_table = Table(workflow_steps, colWidths=[90, 220, 222])
    wf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ('GRID', (0, 0), (-1, -1), 0.5, CARD_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    elements.append(wf_table)
    elements.append(Spacer(1, 12))

    # =========================================================================
    # 3. ALGORITHMIC FOUNDATIONS & MATHEMATICAL FORMULATIONS
    # =========================================================================
    elements.append(Paragraph("3. Algorithmic Architecture & Mathematical Foundations", h1_style))

    elements.append(Paragraph("A. Markerless Skeletal Pose Estimation (MediaPipe 33-Keypoint Pipeline)", h2_style))
    elements.append(Paragraph(
        "The system utilizes the MediaPipe BlazePose convolutional architecture running locally in RAM. "
        "Keypoints tracked include mid-shoulders (11, 12), mid-hips (23, 24), knees (25, 26), ankles (27, 28), and feet (31, 32). "
        "No frames are saved to disk or transmitted over the internet, guaranteeing complete HIPAA/GDPR local data privacy.",
        body_style
    ))

    elements.append(Paragraph("B. Dynamic Phase Segmentation & Physical TUG Duration", h2_style))
    elements.append(Paragraph(
        "To avoid capturing dead camera idle time, the vertical hip trajectory <i>y<sub>hip</sub>(t)</i> is filtered using edge-padded "
        "convolution to prevent boundary collapse: "
        "<font name='Courier'>padded = np.pad(y, pad_width, mode='edge')</font>. "
        "Movement onset (stand initiation <i>t<sub>start</sub></i>) occurs when hip height decreases from the seated plateau. "
        "Movement completion (sit completed <i>t<sub>end</sub></i>) occurs when hip height returns to the chair baseline and settles. "
        "Total physical TUG duration is strictly: "
        "<b><i>T<sub>TUG</sub> = (t<sub>end</sub> - t<sub>start</sub>) &times; &Delta;t</i></b>, ensuring a true ~8–9s physical trial is never misread as the full buffer.",
        body_style
    ))

    elements.append(Paragraph("C. Biomechanical Formulations & Joint Angles", h2_style))
    elements.append(Paragraph(
        "• <b>Knee Flexion Angle:</b> Computed using the 2D dot product between thigh vector <i>BA</i> (hip to knee) and shank vector <i>BC</i> (ankle to knee):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>&theta;<sub>knee</sub> = arccos[(BA &bull; BC) / (||BA|| ||BC||)]</b><br/>"
        "• <b>Postural Trunk Sway:</b> Evaluated during linear locomotion (excluding the 180° turn window) from the shoulder-to-hip coronal vector:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>&theta;<sub>trunk</sub> = arctan2(&Delta;x<sub>sh-hip</sub>, -&Delta;y<sub>sh-hip</sub>)</b> &nbsp;&nbsp;|&nbsp;&nbsp; Sway = std(&theta;<sub>trunk</sub>)<br/>"
        "• <b>Dual-Task Cost (DTC):</b> Quantifies cognitive-motor interference under simultaneous counting/subtraction:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>DTC (%) = [(T<sub>Dual-Task</sub> - T<sub>Single-Task</sub>) / T<sub>Single-Task</sub>] &times; 100%</b> &nbsp;&nbsp;(Clinical cutoff: &gt;22% indicates high alert)",
        body_style
    ))

    elements.append(Paragraph("D. Tri-Model Supervised Machine Learning Ensemble", h2_style))
    elements.append(Paragraph(
        "Inference is executed by an ensemble of three complementary, mathematically independent classifiers trained on quantitative gait features:<br/>"
        "1. <b>Support Vector Machine (SVM):</b> Radial Basis Function (RBF) kernel mapping non-linear biomechanical feature spaces.<br/>"
        "2. <b>Random Forest Classifier:</b> Ensemble of 100 decorrelated decision trees evaluating feature interaction thresholds.<br/>"
        "3. <b>XGBoost (Extreme Gradient Boosting):</b> Gradient boosted tree optimization maximizing discriminative precision.<br/>"
        "• <b>Consensus Consensus Probability:</b> <i>P<sub>consensus</sub> = (P<sub>SVM</sub> + P<sub>RF</sub> + P<sub>XGB</sub>) / 3</i>. "
        "Healthy normal walks yield <i>P &lt; 1%</i> (Consensus: Healthy Normal); pathological gaits yield <i>P &gt; 75%</i> (Consensus: Impaired).",
        body_style
    ))
    elements.append(Spacer(1, 10))

    # =========================================================================
    # 4. COMMANDS TO RUN & DEPLOYMENT GUIDE
    # =========================================================================
    elements.append(Paragraph("4. System Execution & Operational Commands", h1_style))
    elements.append(Paragraph(
        "The project provides multiple execution modalities for developers, evaluators, and clinicians:",
        body_style
    ))

    cmd_data = [
        [
            Paragraph("<b>Execution Mode</b>", body_bold),
            Paragraph("<b>Command / Method</b>", body_bold),
            Paragraph("<b>Description & Purpose</b>", body_bold),
        ],
        [
            Paragraph("<b>Standard Terminal<br/>(Streamlit Run)</b>", body_style),
            Paragraph("<font name='Courier'>python -m streamlit run app.py</font>", code_style),
            Paragraph("Launches the full interactive web application on local host port <b>8501</b>.", body_style),
        ],
        [
            Paragraph("<b>One-Click Batch<br/>(Windows Launcher)</b>", body_style),
            Paragraph("<font name='Courier'>run_app.bat</font>", code_style),
            Paragraph("Double-clickable Windows batch script that auto-detects Python and boots the Streamlit server.", body_style),
        ],
        [
            Paragraph("<b>VS Code 1-Click<br/>(Launch Config)</b>", body_style),
            Paragraph("Press <b>F5</b> or run config:<br/><i>'Run COGAIT-VISION App'</i>", body_style),
            Paragraph("Pre-configured in <font name='Courier'>.vscode/launch.json</font> with integrated Streamlit debugging.", body_style),
        ],
        [
            Paragraph("<b>Video Evaluation<br/>Batch Processor</b>", body_style),
            Paragraph("<font name='Courier'>python src/process_user_videos.py</font>", code_style),
            Paragraph("Runs headless MediaPipe tracking on video files, produces annotated MP4s and CSV metrics.", body_style),
        ],
        [
            Paragraph("<b>Model Re-Training<br/>Script</b>", body_style),
            Paragraph("<font name='Courier'>python src/train_models.py</font>", code_style),
            Paragraph("Trains and serializes SVM, Random Forest, and XGBoost models into the <font name='Courier'>models/</font> directory.", body_style),
        ],
    ]
    cmd_table = Table(cmd_data, colWidths=[105, 200, 227])
    cmd_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ('GRID', (0, 0), (-1, -1), 0.5, CARD_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    elements.append(cmd_table)
    elements.append(Spacer(1, 10))

    # =========================================================================
    # 5. CLINICAL BIOMARKER NORMATIVE REFERENCE MATRIX
    # =========================================================================
    elements.append(Paragraph("5. Quantitative Clinical Biomarker Reference Matrix", h1_style))
    
    biomarker_data = [
        [
            Paragraph("<b>Biomarker Metric</b>", body_bold),
            Paragraph("<b>Healthy Normal Adult</b>", body_bold),
            Paragraph("<b>Alzheimer's / At-Risk Gait</b>", body_bold),
            Paragraph("<b>Clinical Neurological Significance</b>", body_bold),
        ],
        [
            Paragraph("<b>Dual-Task Cost (DTC)</b>", body_style),
            Paragraph("<font color='#059669'><b>7.0% – 13.5%</b> (Normal)</font>", body_style),
            Paragraph("<font color='#dc2626'><b>&gt; 22.0%</b> (High Alert)</font>", body_style),
            Paragraph("Primary hallmark of prefrontal cognitive-motor competition under dual loading.", body_style),
        ],
        [
            Paragraph("<b>Total TUG Duration</b>", body_style),
            Paragraph("<font color='#059669'><b>7.0s – 10.0s</b></font>", body_style),
            Paragraph("<font color='#dc2626'><b>&gt; 13.5s</b> (Fall Risk)</font>", body_style),
            Paragraph("Validated clinical mobility cutoff reflecting motor initiation and postural speed.", body_style),
        ],
        [
            Paragraph("<b>Mean Walking Velocity</b>", body_style),
            Paragraph("<font color='#059669'><b>0.90 – 1.30 m/s</b></font>", body_style),
            Paragraph("<font color='#dc2626'><b>&lt; 0.72 m/s</b> (Bradykinesia)</font>", body_style),
            Paragraph("Executive motor planning speed across the 3-meter walkway.", body_style),
        ],
        [
            Paragraph("<b>Step Cadence</b>", body_style),
            Paragraph("<font color='#059669'><b>98 – 115 steps/min</b></font>", body_style),
            Paragraph("<font color='#dc2626'><b>&lt; 75 steps/min</b> (Shuffling)</font>", body_style),
            Paragraph("Central pattern generator rhythmic coordination.", body_style),
        ],
        [
            Paragraph("<b>Axial Trunk Sway Angle</b>", body_style),
            Paragraph("<font color='#059669'><b>2.0° – 4.0°</b> (Stable)</font>", body_style),
            Paragraph("<font color='#dc2626'><b>&gt; 6.0°</b> (Postural Instability)</font>", body_style),
            Paragraph("Static and dynamic equilibrium control in the coronal plane.", body_style),
        ],
        [
            Paragraph("<b>180° Turning Duration</b>", body_style),
            Paragraph("<font color='#059669'><b>1.8s – 2.5s</b></font>", body_style),
            Paragraph("<font color='#dc2626'><b>&gt; 3.2s</b> (Hesitant Pivot)</font>", body_style),
            Paragraph("Complex vestibular, multi-step turn reorganization.", body_style),
        ],
    ]
    bm_table = Table(biomarker_data, colWidths=[120, 110, 115, 187])
    bm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ('GRID', (0, 0), (-1, -1), 0.5, CARD_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    elements.append(bm_table)
    elements.append(Spacer(1, 14))

    # =========================================================================
    # FOOTER SUMMARY BOX
    # =========================================================================
    footer_text = (
        "<b>Clinical Disclaimer:</b> COGAIT-VISION is an automated assistive AI screening prototype designed to identify "
        "digital gait biomarkers associated with early neuro-motor decline. It does not replace comprehensive medical consultation. "
        "All positive screening alerts recommend follow-up clinical assessment, MMSE/MoCA evaluation, and neurologist review."
    )
    footer_table = Table([[Paragraph(footer_text, meta_style)]], colWidths=[532])
    footer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(footer_table)

    # Build the document
    doc.build(elements)
    print(f"Document successfully created at: {output_filename}")


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "presentation_report_document.pdf"
    generate_presentation_document(out_file)
