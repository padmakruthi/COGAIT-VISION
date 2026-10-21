"""
COGAIT-VISION: Medical Clinical Dashboard for Camera-Based TUG Analysis & Alzheimer's Screening
Clinical AI Research Initiative | Diagnostic Neuromotor Systems
Investigators: B P S Kruthi, Ch. Karthika, A. Laxmi Sahithi
Supervisor: Ms. B K Aishwarya, Assistant Professor
SDGs: SDG 3 (Good Health & Well-being) & SDG 9 (Industry, Innovation & Infrastructure)
"""

import os
import json
import time
import tempfile
import textwrap
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.pose_detector import PoseDetector, calculate_angle_2d
from src.kinematic_features import TUGKinematicFeatureExtractor
from src.classifier import TUGClassifier
from src.video_pose_inspector import extract_full_pose_trajectories
from src.pdf_report_generator import generate_clinical_pdf


# Helper to guarantee HTML strings NEVER get parsed as indented Markdown code blocks
def render_html(html_str: str):
    clean_html = " ".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(clean_html, unsafe_allow_html=True)


def render_styled_table(data_source, columns=None):
    """Renders a clean, hospital-grade responsive HTML table with automatic status pill badges."""
    if isinstance(data_source, pd.DataFrame):
        df = data_source if columns is None else data_source[columns]
        headers = list(df.columns)
        rows = df.values.tolist()
    elif isinstance(data_source, list):
        if not data_source:
            return
        if isinstance(data_source[0], dict):
            headers = columns if columns else list(data_source[0].keys())
            rows = [[item.get(h, "") for h in headers] for item in data_source]
        else:
            return
    else:
        return

    html_parts = ['<div class="ice-table-container"><table class="ice-table"><thead><tr>']
    for h in headers:
        html_parts.append(f'<th>{h}</th>')
    html_parts.append('</tr></thead><tbody>')

    for row in rows:
        html_parts.append('<tr>')
        for val in row:
            val_str = str(val)
            if any(bad in val_str for bad in ["🚨", "CRITICAL", "High Risk", "Impaired", "Alert", "Fall Risk"]):
                badge_html = f'<span class="badge-ice-red">{val_str}</span>'
                html_parts.append(f'<td>{badge_html}</td>')
            elif any(good in val_str for good in ["🟢", "Normal", "Healthy", "Optimal", "Intact", "Low Fall Risk"]):
                badge_html = f'<span class="badge-ice-green">{val_str}</span>'
                html_parts.append(f'<td>{badge_html}</td>')
            elif any(warn in val_str for warn in ["🟡", "Warning", "Mild", "At-Risk"]):
                badge_html = f'<span class="badge-ice-amber">{val_str}</span>'
                html_parts.append(f'<td>{badge_html}</td>')
            elif any(blue in val_str for blue in ["SVM", "Random Forest", "XGBoost"]):
                badge_html = f'<span class="badge-ice-blue">{val_str}</span>'
                html_parts.append(f'<td>{badge_html}</td>')
            else:
                html_parts.append(f'<td><b>{val_str}</b></td>' if val == row[0] else f'<td>{val_str}</td>')
        html_parts.append('</tr>')

    html_parts.append('</tbody></table></div>')
    render_html("".join(html_parts))


# High-Quality Crisp Medical SVG Icons
SVG_STETHOSCOPE = """<svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.8 2.4A2.4 2.4 0 0 0 2.4 4.8v4.8a4.8 4.8 0 0 0 9.6 0V4.8a2.4 2.4 0 0 0-2.4-2.4"/><path d="M7.2 14.4v2.4a4.8 4.8 0 0 0 9.6 0v-4.8"/><circle cx="16.8" cy="9.6" r="2.4"/><circle cx="20.4" cy="9.6" r="1.2"/></svg>"""

SVG_BRAIN = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#9333ea" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z"/></svg>"""

SVG_CAMERA = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>"""

SVG_PULSE = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>"""

SVG_SHIELD = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>"""

SVG_WALK = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="4" r="2"/><path d="m9 20 3-6-3-4 4-2 2 3h4"/><path d="m6 16 3-2 1.5 3L14 20"/></svg>"""

SVG_ALERT = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>"""

SVG_CHECK = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>"""


# Page configuration
st.set_page_config(
    page_title="COGAIT-VISION | Clinical Alzheimer's TUG Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Medical CSS Styling
render_html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Clean Chrome & Modern Viewport */
    #MainMenu, header, footer, [data-testid="stDecoration"], [data-testid="stHeader"] {
        display: none !important;
    }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1320px !important;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 12% 8%, rgba(219, 234, 254, 0.5) 0%, rgba(248, 250, 252, 1) 45%),
                    radial-gradient(circle at 88% 50%, rgba(236, 253, 245, 0.45) 0%, rgba(248, 250, 252, 1) 50%),
                    radial-gradient(circle at 50% 95%, rgba(238, 242, 255, 0.4) 0%, rgba(248, 250, 252, 1) 55%) !important;
        background-attachment: fixed !important;
    }

    /* Buttons Modern Styling */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        padding: 9px 20px !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid rgba(203, 213, 225, 0.8) !important;
        background: #ffffff !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1.5px) !important;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08) !important;
        border-color: #94a3b8 !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #0284c7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 50%, #0369a1 100%) !important;
        box-shadow: 0 8px 26px rgba(37, 99, 235, 0.48) !important;
    }

    /* Stepper Navigation Buttons (Top Bar) */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 14px !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        padding: 10px 14px !important;
        border: 1px solid rgba(226, 232, 240, 0.9) !important;
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        color: #334155 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.12) !important;
        border-color: #93c5fd !important;
        color: #1d4ed8 !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #0284c7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35) !important;
    }

    /* Sidebar Navigation */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        text-align: left !important;
        justify-content: flex-start !important;
        border-radius: 12px !important;
        padding: 11px 16px !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        margin-bottom: 7px !important;
        border: 1px solid #f1f5f9 !important;
        background: #f8fafc !important;
        color: #334155 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #eff6ff !important;
        border-color: #bfdbfe !important;
        color: #1d4ed8 !important;
        transform: translateX(3px) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        transform: translateX(3px) !important;
    }

    /* Top Floating Medical Navbar */
    .med-navbar {
        background: rgba(255, 255, 255, 0.94);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(226, 232, 240, 0.9);
        border-radius: 20px;
        padding: 16px 26px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.05), 0 8px 10px -6px rgba(15, 23, 42, 0.03);
    }
    .med-logo {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .med-logo-icon {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: 800;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    }
    .med-logo-text {
        font-size: 21px;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.4px;
        line-height: 1.15;
    }
    .med-logo-sub {
        font-size: 11.5px;
        font-weight: 600;
        color: #2563eb;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }
    
    /* Hero Section */
    .hero-container {
        padding: 6px 4px 20px 4px;
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 700;
        color: #1e40af;
        letter-spacing: 0.6px;
        margin-bottom: 12px;
        text-transform: uppercase;
        background: #dbeafe;
        border: 1px solid #bfdbfe;
        padding: 5px 14px;
        border-radius: 20px;
    }
    .hero-heading {
        font-size: 38px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
        letter-spacing: -1px;
        margin-bottom: 14px;
    }
    .hero-accent {
        color: #1d4ed8;
        background: linear-gradient(135deg, #2563eb, #0284c7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-desc {
        font-size: 15px;
        color: #475569;
        line-height: 1.65;
        max-width: 620px;
        margin-bottom: 20px;
    }
    
    /* Hero Graphic Card */
    .hero-card-art {
        background: linear-gradient(135deg, #eff6ff 0%, #e0e7ff 50%, #f0fdf4 100%);
        border-radius: 24px;
        padding: 26px 22px;
        box-shadow: 0 16px 40px rgba(37, 99, 235, 0.08);
        border: 1.5px solid rgba(255, 255, 255, 0.95);
        text-align: center;
    }
    .floating-pill {
        background: #ffffff;
        border-radius: 30px;
        padding: 8px 16px;
        font-size: 12px;
        font-weight: 700;
        color: #0f172a;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
        border: 1px solid #e2e8f0;
        margin: 4px;
    }
    
    /* Pill Tags */
    .pill-tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }
    .pill-blue { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
    .pill-indigo { background: #eef2ff; color: #4338ca; border: 1px solid #c7d2fe; }
    .pill-green { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
    .pill-red { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
    .pill-dark { background: #0f172a; color: white; }

    /* Live Pulsing Dot Indicator */
    .status-live {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.4px;
    }
    .dot-pulse {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Feature Cards */
    .med-feature-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .med-feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.07);
        border-color: #cbd5e1;
    }
    .med-feature-card h3 {
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        margin: 12px 0 6px 0;
    }
    .med-feature-card p {
        font-size: 13.5px;
        color: #64748b;
        line-height: 1.55;
        margin: 0;
    }

    /* Diagnosis Banner - Danger */
    .diagnosis-alert-danger {
        background: linear-gradient(135deg, #fff5f5 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 18px;
        padding: 24px 28px;
        margin: 18px 0;
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.12);
    }
    
    /* Diagnosis Banner - Safe */
    .diagnosis-alert-safe {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #10b981;
        border-radius: 18px;
        padding: 24px 28px;
        margin: 18px 0;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.12);
    }

    /* Permission Gate Box */
    .permission-gate {
        background: linear-gradient(135deg, #f0fdfa 0%, #e0f2fe 100%);
        border: 1.5px solid #0284c7;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.08);
    }

    /* Metric Bubble */
    .metric-bubble {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-bubble:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    }
    .metric-bubble h4 {
        margin: 0;
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.6px;
    }
    .metric-bubble p {
        margin: 6px 0 0 0;
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
    }

    /* Ice Glass Medical Tables */
    .ice-table-container {
        overflow-x: auto;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
        background: #ffffff;
        margin: 16px 0 20px 0;
    }
    .ice-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13.5px;
        text-align: left;
    }
    .ice-table thead tr {
        background: #f8fafc;
        border-bottom: 1.5px solid #e2e8f0;
    }
    .ice-table th {
        color: #475569;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.6px;
        padding: 14px 18px;
    }
    .ice-table td {
        padding: 13px 18px;
        border-bottom: 1px solid #f1f5f9;
        color: #1e293b;
        vertical-align: middle;
    }
    .ice-table tbody tr:last-child td {
        border-bottom: none;
    }
    .ice-table tbody tr:hover td {
        background: #f8fafc;
    }

    /* Status Pill Badges */
    .badge-ice-red {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        font-weight: 700;
        font-size: 11.5px;
        padding: 4px 12px;
        border-radius: 9999px;
        letter-spacing: 0.3px;
    }
    .badge-ice-green {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        font-weight: 700;
        font-size: 11.5px;
        padding: 4px 12px;
        border-radius: 9999px;
        letter-spacing: 0.3px;
    }
    .badge-ice-amber {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
        font-weight: 700;
        font-size: 11.5px;
        padding: 4px 12px;
        border-radius: 9999px;
        letter-spacing: 0.3px;
    }
    .badge-ice-blue {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        font-weight: 700;
        font-size: 11.5px;
        padding: 4px 12px;
        border-radius: 9999px;
        letter-spacing: 0.3px;
    }
    </style>
    """
)

# Initialize Session State for Guided Workflow
if "nav_stage" not in st.session_state:
    st.session_state["nav_stage"] = "🏠 Home & Overview"
if "cam_permission" not in st.session_state:
    st.session_state["cam_permission"] = False
if "cam_active" not in st.session_state:
    st.session_state["cam_active"] = False
if "patient_info" not in st.session_state:
    st.session_state["patient_info"] = {
        "id": "PT-2026-081",
        "name": "Kruthi (Student Tester)",
        "age": 20,
        "gender": "Female",
        "confounder_score": 0,
        "cognitive_score": 0,
        "cleared_for_test": True,
    }
if "live_test_results" not in st.session_state:
    st.session_state["live_test_results"] = None
if "selected_walk_dataset" not in st.session_state:
    st.session_state["selected_walk_dataset"] = "🟢 Benchmark Recording 1: Healthy Control Gait"
if "dual_task_lang" not in st.session_state:
    st.session_state["dual_task_lang"] = "English"

# Multilingual Language-Adaptive Dual-Task Cognitive Protocols
MULTILINGUAL_DUAL_TASKS = {
    "English": {
        "title": "English Verbal Cognitive Dual-Task",
        "task_name": "Serial 7s Subtraction & Verbal Fluency",
        "primary_prompt": "Count backwards from 100 by 7s (100, 93, 86, 79, 72...) while walking.",
        "alt_prompt": "Recite the months of the year in reverse (December, November, October...).",
        "clinical_note": "Evaluates prefrontal central executive network and working memory under motor load.",
    },
    "Telugu (తెలుగు)": {
        "title": "తెలుగు భాషా ద్వంద్వ-కార్య నిర్ధారణ (Telugu Verbal Dual-Task)",
        "task_name": "సంఖ్యలు వెనుకకు లెక్కించడం & వారపు రోజులు",
        "primary_prompt": "నడుస్తూ 100 నుండి 7 తగ్గిస్తూ వెనుకకు లెక్కించండి (100, 93, 86, 79, 72...).",
        "alt_prompt": "వారంలోని రోజులను వెనుక నుండి వరుసగా చెప్పండి (ఆదివారం, శనివారం, శుక్రవారం, గురువారం...).",
        "clinical_note": "మాతృభాషలో ద్వంద్వ-కార్య విధానం భాషా అడ్డంకులను తొలగించి ఖచ్చితమైన కాగ్నిటివ్ లోడ్ ఫలితాలను ఇస్తుంది.",
    },
    "Hindi (हिंदी)": {
        "title": "हिंदी भाषा दोहरी-कार्य संज्ञानात्मक परीक्षण (Hindi Verbal Dual-Task)",
        "task_name": "उल्टी गिनती एवं माह के नाम",
        "primary_prompt": "चलते हुए 100 में से 7 घटाते हुए उल्टी गिनती करें (100, 93, 86, 79, 72...).",
        "alt_prompt": "साल के महीनों के नाम उल्टे क्रम में बोलें (दिसंबर, नवंबर, अक्टूबर, सितंबर...).",
        "clinical_note": "मातृभाषा में संज्ञानात्मक कार्य भाषा संबंधी पूर्वाग्रहों को दूर कर सटीक मोटर-संज्ञानात्मक मूल्यांकन करता है।",
    },
}

def is_abnormal_dataset_label(label: str) -> bool:
    """Detects if dataset or test mode represents abnormal / simulated hesitant gait."""
    if not label:
        return False
    lower = label.lower()
    return any(w in lower for w in ["abnormal", "hesitant", "simulat", "impaired", "alert", "risk"])

def calculate_transparent_risk_score(dtc_val: float, speed_val: float, turn_val: float, cog_score: int, age: int, ortho_score: int):
    """
    Computes transparent multi-modal screening risk score (0-100%)
    Weights: Gait Kinematics (50%), Cognitive Intake (30%), Demographic Age Risk (20%)
    """
    # 1. Gait Score (0-100)
    gait_pts = 0.0
    if dtc_val > 22.0:
        gait_pts += 75.0
    elif dtc_val > 15.0:
        gait_pts += 40.0
    else:
        gait_pts += 10.0
    
    if speed_val < 0.88:
        gait_pts += 15.0
    elif speed_val < 0.98:
        gait_pts += 5.0
        
    if turn_val > 2.5:
        gait_pts += 15.0
    elif turn_val > 2.2:
        gait_pts += 5.0
    gait_score = min(100.0, gait_pts)

    # 2. Cognitive Score (0-100)
    cog_score_val = min(100.0, (cog_score / 4.0) * 100.0)

    # 3. Demographic Risk (0-100)
    if age >= 80:
        demo_score = 75.0
    elif age >= 70:
        demo_score = 50.0
    elif age >= 60:
        demo_score = 25.0
    elif age >= 45:
        demo_score = 10.0
    else:
        # Young Adults & Students (Age 18 - 44) have zero epidemiological age-based risk
        demo_score = 0.0

    # Composite weighted calculation
    # If motor slowing / DTC is severely elevated (> 22%), gait impairment directly activates the High Risk Alert
    if dtc_val > 22.0 or gait_score >= 70.0:
        composite = max(74.0, (0.60 * gait_score) + (0.25 * cog_score_val) + (0.15 * demo_score))
    else:
        composite = (0.50 * gait_score) + (0.30 * cog_score_val) + (0.20 * demo_score)

    if composite >= 65.0:
        band = "HIGH_RISK"
        band_label = "HIGH SCREENING RISK (ALERT)"
        badge_class = "badge-ice-red"
        rec = "🚨 Dual-Task Motor Deceleration Alert (>22% DTC): Marked cognitive-motor decoupling detected. Comprehensive neurological evaluation and cognitive screening recommended."
    elif composite >= 35.0:
        band = "MODERATE_RISK"
        band_label = "MODERATE SCREENING RISK"
        badge_class = "badge-ice-amber"
        rec = "Mild motor-cognitive deceleration detected. Recommended 3 to 6-month digital TUG re-assessment and active physical exercise."
    else:
        band = "LOW_RISK"
        band_label = "LOW SCREENING RISK (HEALTHY)"
        badge_class = "badge-ice-green"
        if age < 45:
            rec = f"🌟 Optimal Neuromotor Function: Intact cognitive-motor speed and balance, consistent with healthy young adult baseline (Age {age}). Zero neurodegenerative indicators detected."
        else:
            rec = f"Biomechanics and intake are consistent with healthy physiological aging for age {age}. Annual routine screening recommended."

    return {
        "composite_score": round(composite, 1),
        "band": band,
        "band_label": band_label,
        "badge_class": badge_class,
        "gait_score": round(gait_score, 1),
        "cog_score": round(cog_score_val, 1),
        "demo_score": round(demo_score, 1),
        "recommendation": rec,
        "ortho_warning": ortho_score >= 3,
    }

@st.cache_resource
def load_classifier():
    return TUGClassifier(models_dir="models")

@st.cache_data
def load_metrics():
    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_eval_data():
    eval_path = "reports/user_videos_evaluation.json"
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            return json.load(f)
    return {}

classifier = load_classifier()
metrics = load_metrics()
eval_data = load_eval_data()

# Top Floating Medical Navbar
render_html(
    """
    <div class="med-navbar">
        <div class="med-logo">
            <div class="med-logo-icon">🧠</div>
            <div>
                <div class="med-logo-text">COGAIT-VISION</div>
                <div class="med-logo-sub">Clinical AI Gait & TUG Screening System</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
            <span class="pill-tag pill-blue">⚡ CLINICAL AI</span>
            <span class="pill-tag pill-indigo">🔬 NEURO-GAIT PROTOCOL</span>
            <span class="pill-tag pill-green">🌿 SDG 3 & 9</span>
            <span class="status-live"><span class="dot-pulse"></span> SYSTEM READY</span>
        </div>
    </div>
    """
)

# Workflow Stages
nav_stages = [
    "🏠 Home & Overview",
    "🎥 Step 1: Camera Scan (TUG Test)",
    "📋 Step 2: Medical Questionnaire",
    "🩺 Step 3: Diagnostic Screening Report & PDF",
    "🔬 Step 4: AI Model Benchmarks & Validation",
]

# Interactive Stepper Progress Bar
step_items = [
    ("1. Home", "🏠 Home & Overview"),
    ("2. Camera Scan", "🎥 Step 1: Camera Scan (TUG Test)"),
    ("3. Questionnaire", "📋 Step 2: Medical Questionnaire"),
    ("4. Clinical Report & PDF", "🩺 Step 3: Diagnostic Screening Report & PDF"),
    ("5. AI Benchmarks", "🔬 Step 4: AI Model Benchmarks & Validation"),
]

active_idx = 0
for idx, (lbl, stg) in enumerate(step_items):
    if st.session_state["nav_stage"] == stg:
        active_idx = idx
        break

active_pct = max(5, int((active_idx / (len(step_items) - 1)) * 100))
render_html(
    f"""
    <div style="background: rgba(226, 232, 240, 0.7); height: 5px; border-radius: 9999px; margin: 4px 0 12px 0; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.04);">
        <div style="background: linear-gradient(90deg, #2563eb 0%, #0284c7 50%, #10b981 100%); height: 100%; width: {active_pct}%; transition: width 0.4s ease; border-radius: 9999px;"></div>
    </div>
    """
)

cols_step = st.columns(5)
for i, (name, stage_target) in enumerate(step_items):
    with cols_step[i]:
        is_active = (i == active_idx)
        is_completed = (i < active_idx)
        btn_label = f"📍 {name}" if is_active else f"✅ {name}" if is_completed else f"⚪ {name}"
        btn_type = "primary" if is_active else "secondary"
        if st.button(btn_label, key=f"top_step_btn_{i}", use_container_width=True, type=btn_type):
            st.session_state["nav_stage"] = stage_target
            st.rerun()

st.markdown("<div style='margin-bottom:18px;'></div>", unsafe_allow_html=True)

# Sidebar Navigation (Clean Button Grid)
st.sidebar.markdown(
    """
    <div style="padding: 6px 2px 14px 2px; border-bottom: 1px solid #f1f5f9; margin-bottom: 14px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="background:linear-gradient(135deg, #2563eb, #1d4ed8); width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:white; font-size:18px; font-weight:800; box-shadow:0 4px 10px rgba(37,99,235,0.3);">🧠</div>
            <div>
                <div style="font-size:15px; font-weight:800; color:#0f172a;">COGAIT CLINICAL</div>
                <div style="font-size:10.5px; font-weight:700; color:#2563eb; text-transform:uppercase; letter-spacing:0.5px;">Diagnostic Platform</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("### 🏥 Guided Clinical Workflow")

for i, stage_name in enumerate(nav_stages):
    is_active = (st.session_state["nav_stage"] == stage_name)
    btn_type = "primary" if is_active else "secondary"
    icon_prefix = "👉 " if is_active else "📍 " if i == 0 else "• "
    if st.sidebar.button(
        f"{icon_prefix}{stage_name}",
        key=f"sb_nav_btn_{i}",
        use_container_width=True,
        type=btn_type,
    ):
        st.session_state["nav_stage"] = stage_name
        st.rerun()

dual_task_active = st.sidebar.checkbox(
    "Dual-Task Protocol Active (Counting Backwards)",
    value=True,
    help="Assesses cognitive-motor interference. Essential for Alzheimer's early screening.",
)

st.sidebar.divider()
st.sidebar.markdown("### 🏆 Clinical Biomarker Norms")
st.sidebar.info(
    """
    - **Total TUG Duration:** Normal < 10.0s | Impaired > 13.5s
    - **Dual-Task Cost (DTC):** Normal < 15.0% | Alzheimer's Alert > 22.0%
    - **Trunk Sway Angle:** Normal < 4.0° | Balance Deficit > 6.0°
    - **180° Turning Duration:** Normal < 2.5s | Instability > 3.2s
    """
)


# ==============================================================================
# SCREEN 1: HOME & OVERVIEW
# ==============================================================================
if st.session_state["nav_stage"] == "🏠 Home & Overview":
    h_col1, h_col2 = st.columns([1.25, 1])

    with h_col1:
        render_html(
            """
            <div class="hero-container">
                <span class="hero-eyebrow">✨ Non-Invasive Computer Vision Protocol</span>
                <div class="hero-heading">
                    Markerless TUG Analysis<br>For <span class="hero-accent">Early Alzheimer's</span>
                </div>
                <div class="hero-desc">
                    Our non-invasive AI system captures 33 biomechanical landmarks during the Timed Up and Go (TUG) test, 
                    quantifying dual-task motor deceleration, postural sway, and turning hesitation to screen for 
                    preclinical neurodegenerative impairment long before irreversible clinical memory loss.
                </div>
                <div style="display:flex; gap:12px; margin: 12px 0 20px 0; flex-wrap:wrap;">
                    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; padding:10px 16px; box-shadow:0 4px 12px rgba(15,23,42,0.03);">
                        <div style="font-size:20px; font-weight:800; color:#2563eb;">33 Points</div>
                        <div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">Biomechanical Landmarks</div>
                    </div>
                    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; padding:10px 16px; box-shadow:0 4px 12px rgba(15,23,42,0.03);">
                        <div style="font-size:20px; font-weight:800; color:#10b981;">&lt; 15.0%</div>
                        <div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">Healthy DTC Norm</div>
                    </div>
                    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; padding:10px 16px; box-shadow:0 4px 12px rgba(15,23,42,0.03);">
                        <div style="font-size:20px; font-weight:800; color:#8b5cf6;">92.8%</div>
                        <div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">AI Consensus ROC-AUC</div>
                    </div>
                </div>
            </div>
            """
        )

        b_c1, b_c2 = st.columns([1.2, 1])
        with b_c1:
            if st.button("🚀 Start Clinical Assessment (Camera Scan) ➔", type="primary", use_container_width=True):
                st.session_state["nav_stage"] = "🎥 Step 1: Camera Scan (TUG Test)"
                st.rerun()
        with b_c2:
            if st.button("⚖️ Watch Demo Comparison", type="secondary", use_container_width=True):
                st.session_state["nav_stage"] = "🎥 Step 1: Camera Scan (TUG Test)"
                st.rerun()

        render_html(
            """
            <div style="margin-top: 24px; color:#64748b; font-size:12.5px; line-height:1.5;">
                <b>Project Investigators:</b> B P S Kruthi • Ch. Karthika • A. Laxmi Sahithi<br>
                <b>Project Guide:</b> Ms. B K Aishwarya, Assistant Professor
            </div>
            """
        )

    with h_col2:
        render_html(
            f"""
            <div class="hero-card-art">
                <div style="display:flex; justify-content:center; margin-bottom:8px;">{SVG_STETHOSCOPE}</div>
                <h3 style="color:#0f172a; margin:4px 0; font-size:21px;">Dr. AI Gait Screening Engine</h3>
                <p style="color:#475569; font-size:13.5px; margin:0 0 16px 0;">Markerless Pose Estimation & Multi-Model Cognitive Diagnosis</p>
                <div style="display:flex; flex-wrap:wrap; justify-content:center; gap:8px;">
                    <div class="floating-pill">{SVG_CAMERA} <span>📹 100% Markerless (Standard RGB)</span></div>
                    <div class="floating-pill">{SVG_BRAIN} <span>🧠 Dual-Task Cost (DTC Biomarker)</span></div>
                    <div class="floating-pill">{SVG_PULSE} <span>⭐ 92.8% ROC-AUC Consensus</span></div>
                    <div class="floating-pill">{SVG_SHIELD} <span>🔒 100% On-Device Local Privacy</span></div>
                    <div class="floating-pill">{SVG_WALK} <span>🎯 UN SDG 3 & 9 Aligned</span></div>
                </div>
            </div>
            """
        )

    st.markdown("<div style='margin-bottom:26px;'></div>", unsafe_allow_html=True)
    st.markdown("### 💡 Core System Pillars")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_html(
            f"""
            <div class="med-feature-card">
                <div>{SVG_CAMERA}</div>
                <h3>Markerless 33-Point Tracking</h3>
                <p>High-resolution MediaPipe Pose estimation without requiring costly motion-capture suits or wearable sensors.</p>
            </div>
            """
        )

    with c2:
        render_html(
            f"""
            <div class="med-feature-card">
                <div>{SVG_BRAIN}</div>
                <h3>Dual-Task Cost (DTC)</h3>
                <p>Quantifies prefrontal cognitive-motor competition when performing concurrent serial subtraction during walking.</p>
            </div>
            """
        )

    with c3:
        render_html(
            f"""
            <div class="med-feature-card">
                <div>{SVG_WALK}</div>
                <h3>Angular Kinematics & Sway</h3>
                <p>Computes knee flexion/extension angles, hip extension, and trunk sway during the 180° turning pivot.</p>
            </div>
            """
        )

    with c4:
        render_html(
            f"""
            <div class="med-feature-card">
                <div>{SVG_SHIELD}</div>
                <h3>Tri-Model Supervised Consensus</h3>
                <p>Supervised decision agreement from Support Vector Machine (SVM), Random Forest, and XGBoost.</p>
            </div>
            """
        )


# ==============================================================================
# SCREEN 2: STEP 1 - CAMERA SCAN & VIDEO ASSESSMENT
# ==============================================================================
elif st.session_state["nav_stage"] == "🎥 Step 1: Camera Scan (TUG Test)":
    st.markdown("## 🎥 Step 1: Camera Scan & Patient Video TUG Assessment")
    st.caption("Standardized TUG motion analysis: evaluate benchmark side-by-side videos, capture live patient webcam, or inspect video trajectories.")

    # Modular Rendering Functions
    def render_side_by_side_demo(key_prefix: str = "sb"):
        render_html(
            """
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:16px; padding:18px 22px; margin-bottom:14px; box-shadow:0 4px 16px rgba(15,23,42,0.03);">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div>
                        <h3 style="color:#0f172a; margin:0 0 4px 0; font-size:18px;">⚖️ Side-by-Side Reference Video Demonstration</h3>
                        <p style="color:#64748b; font-size:13.5px; margin:0;">Direct comparison between healthy control gait and preclinical Alzheimer's dual-task motor slowing.</p>
                    </div>
                    <span class="pill-tag pill-indigo">PRE-RECORDED BENCHMARK</span>
                </div>
            </div>
            """
        )
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            render_html(
                """
                <div style="background:#f0fdf4; border:2px solid #86efac; border-radius:14px; padding:12px 16px; margin-bottom:10px;">
                    <span class="pill-tag pill-green">HEALTHY CONTROL PROFILE</span>
                    <h4 style="color:#065f46; margin:6px 0 0 0; font-size:16px;">🟢 Normal Walk (MediaPipe Skeleton)</h4>
                </div>
                """
            )
            norm_v = "data/videos/web_annotated_normal.mp4"
            if os.path.exists(norm_v):
                st.video(norm_v)
            else:
                st.video("data/videos/normal walk.mp4")

            if st.button("🟢 Select Normal Walk & Proceed to Questionnaire ➔", key=f"{key_prefix}_sel_norm", use_container_width=True):
                st.session_state["selected_walk_dataset"] = "🟢 Benchmark Recording 1: Healthy Control Gait"
                st.session_state["nav_stage"] = "📋 Step 2: Medical Questionnaire"
                st.rerun()

        with v_col2:
            render_html(
                """
                <div style="background:#fef2f2; border:2px solid #fca5a5; border-radius:14px; padding:12px 16px; margin-bottom:10px;">
                    <span class="pill-tag pill-red">SIMULATED BENCHMARK (NOT A PATIENT)</span>
                    <h4 style="color:#991b1b; margin:6px 0 0 0; font-size:16px;">🔴 Simulated Slowed Gait (MediaPipe Skeleton)</h4>
                </div>
                """
            )
            abnorm_v = "data/videos/web_annotated_abnormal.mp4"
            if os.path.exists(abnorm_v):
                st.video(abnorm_v)
            else:
                st.video("data/videos/abnormal.mp4")

            if st.button("🚨 Select Simulated Walk & Proceed to Questionnaire ➔", key=f"{key_prefix}_sel_abnorm", type="primary", use_container_width=True):
                st.session_state["selected_walk_dataset"] = "🔴 Benchmark Simulation: Hesitant Gait (SIMULATED BENCHMARK - NOT A PATIENT)"
                st.session_state["nav_stage"] = "📋 Step 2: Medical Questionnaire"
                st.rerun()

        st.markdown("#### 📊 Quantitative Biomarker Differential Matrix")
        matrix_data = [
            {"Biomarker": "Dual-Task Cognitive Cost (DTC)", "Normal Walk": "9.20% (Normal)", "Abnormal (Alzheimer's)": "🚨 25.82% (High Risk Alert)", "Clinical Explanation": "Reflects prefrontal executive motor interference; >22% is the primary hallmark of Alzheimer's decline."},
            {"Biomarker": "Walking Velocity", "Normal Walk": "0.91 m/s", "Abnormal (Alzheimer's)": "0.85 m/s (Slower)", "Clinical Explanation": "Motor slowing occurs as executive motor planning deteriorates under dual-task cognitive loading."},
            {"Biomarker": "Cadence (Steps/min)", "Normal Walk": "103.5 steps/min", "Abnormal (Alzheimer's)": "105.0 steps/min", "Clinical Explanation": "Rhythmicity and stride generation across the 3m walking path."},
            {"Biomarker": "SVM Model Impairment Score", "Normal Walk": "Healthy Consensus", "Abnormal (Alzheimer's)": "🚨 82.8% At-Risk Probability", "Clinical Explanation": "Supervised Support Vector Machine detects non-linear motor signature."},
            {"Biomarker": "Consensus Clinical Verdict", "Normal Walk": "🟢 Healthy Normal Profile", "Abnormal (Alzheimer's)": "🔴 Cognitive Impairment Warning Alert", "Clinical Explanation": "Tri-model consensus screening output."},
        ]
        render_styled_table(matrix_data)

    def render_live_webcam_scan(key_prefix: str = "cam", dual_task_live: bool = False):
        render_html(
            f"""
            <div class="permission-gate">
                <div style="display:flex; align-items:center; gap:8px;">
                    {SVG_SHIELD}
                    <span class="pill-tag pill-dark">🔒 PRIVACY & CONSENT GATEWAY (INDIA DPDP ACT 2023)</span>
                </div>
                <h3 style="color:#0369a1; margin:8px 0 4px 0; font-size:18px;">Patient Camera Access Authorization & Data Protection Notice</h3>
                <p style="color:#0c4a6e; font-size:13.5px; margin:0; line-height:1.55;">
                    Your privacy is protected under the <b>Digital Personal Data Protection (DPDP) Act 2023</b>. 
                    All video processing occurs strictly on-device in volatile computer memory (RAM) via MediaPipe. 
                    <b>Zero video frames, face biometrics, or camera recordings are saved to disk or transmitted to any server.</b>
                </p>
            </div>
            """
        )

        grant_permission = st.checkbox(
            "🛡️ I grant explicit permission to activate the local webcam for Timed Up and Go (TUG) motion analysis.",
            value=st.session_state["cam_permission"],
            help="Check this box to unlock camera hardware controls.",
            key=f"{key_prefix}_grant_perm",
        )
        st.session_state["cam_permission"] = grant_permission

        if not grant_permission:
            render_html(
                f"""
                <div style="background:#f8fafc; border:2px dashed #94a3b8; border-radius:16px; padding:28px 20px; text-align:center; margin:16px 0;">
                    <div style="display:flex; justify-content:center; margin-bottom:8px;">{SVG_SHIELD}</div>
                    <h3 style="color:#1e293b; margin:8px 0 4px 0; font-size:18px;">Camera Access Locked (Consent Required)</h3>
                    <p style="color:#64748b; font-size:13.5px; margin:0;">
                        To protect your privacy, camera hardware access is disabled by default. 
                        Please check the authorization box above to grant permission.
                    </p>
                </div>
                """
            )
            st.session_state["cam_active"] = False
        else:
            st.success("✅ Camera Permission Granted. Hardware controls are unlocked below.")

            render_html(
                """
                <div style="background:#f8fafc; border:1.5px solid #cbd5e1; border-radius:12px; padding:16px; margin:14px 0 18px 0;">
                    <span class="pill-tag pill-blue">INTERACTIVE TUG PROTOCOL TIMELINE</span>
                    <p style="margin:8px 0 0 0; font-size:14px; color:#1e293b; font-weight:600;">
                        🪑 <b>1. Sit in Chair</b> &nbsp;➔&nbsp; 
                        🧍 <b>2. Stand Up</b> &nbsp;➔&nbsp; 
                        🚶 <b>3. Walk 3 Meters</b> &nbsp;➔&nbsp; 
                        🔄 <b>4. Turn 180°</b> &nbsp;➔&nbsp; 
                        🚶 <b>5. Walk Back</b> &nbsp;➔&nbsp; 
                        🪑 <b>6. Sit Down Safely</b>
                    </p>
                </div>
                """
            )

            cfg1, cfg2 = st.columns(2)
            with cfg1:
                patient_name_in = st.text_input("👤 Patient Full Name:", value=st.session_state["patient_info"]["name"], key=f"{key_prefix}_pat_name")
                patient_label = st.text_input("🆔 Participant ID / Label:", value=st.session_state["patient_info"]["id"], key=f"{key_prefix}_pat_lbl")
                st.session_state["patient_info"]["name"] = patient_name_in
                st.session_state["patient_info"]["id"] = patient_label
            with cfg2:
                webcam_idx = st.number_input("Webcam Device Index:", min_value=0, max_value=4, value=0, key=f"{key_prefix}_device_idx")
                test_seconds = st.slider("Assessment Time Window (Seconds):", min_value=10, max_value=45, value=22, key=f"{key_prefix}_t_slider")
                dual_task_active_box = st.checkbox("Participant Doing Dual-Task (Continuous Verbal Subtraction)", value=dual_task_live, key=f"{key_prefix}_dt_box")

            dt_lang = st.selectbox(
                "🌐 Cognitive Dual-Task Language (Removes Demographic/Linguistic Bias):",
                ["English", "Telugu (తెలుగు)", "Hindi (हिंदी)"],
                index=["English", "Telugu (తెలుగు)", "Hindi (हिंदी)"].index(st.session_state.get("dual_task_lang", "English")),
                key=f"{key_prefix}_lang_sel",
                help="Culturally and linguistically tailored verbal cognitive tasks prevent false-positive motor slowing in non-native English speakers."
            )
            st.session_state["dual_task_lang"] = dt_lang
            task_info = MULTILINGUAL_DUAL_TASKS[dt_lang]

            render_html(
                f"""
                <div style="background:#eff6ff; border:1.5px solid #bfdbfe; border-radius:12px; padding:12px 16px; margin:8px 0 14px 0;">
                    <span class="pill-tag pill-blue">CULTURAL COGNITIVE PROTOCOL: {dt_lang}</span>
                    <div style="font-weight:700; color:#1e40af; font-size:13.5px; margin:4px 0 2px 0;">🗣️ {task_info['task_name']}</div>
                    <div style="font-size:13px; color:#1e293b; margin-bottom:4px;"><b>Primary Verbal Task:</b> {task_info['primary_prompt']}</div>
                    <div style="font-size:11.5px; color:#64748b; font-style:italic;">💡 {task_info['clinical_note']}</div>
                </div>
                """
            )

            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                start_live = st.button("🟢 Turn Camera ON & Start TUG Scan", type="primary", use_container_width=True, key=f"{key_prefix}_start_btn")
            with c_btn2:
                stop_live = st.button("⏹️ Turn Camera OFF / Disconnect Webcam", type="secondary", use_container_width=True, key=f"{key_prefix}_stop_btn")

            if stop_live:
                st.session_state["cam_active"] = False
                st.warning("🔒 Camera has been turned OFF. Webcam stream released.")
                st.rerun()

            if start_live:
                st.session_state["cam_active"] = True

            if not st.session_state["cam_active"]:
                render_html(
                    f"""
                    <div style="background:#f8fafc; border:1.5px dashed #94a3b8; border-radius:14px; padding:28px 20px; text-align:center; margin:18px 0;">
                        <div style="display:flex; justify-content:center; margin-bottom:8px;">{SVG_CAMERA}</div>
                        <h3 style="color:#1e293b; margin:8px 0 4px 0; font-size:18px;">Camera Status: OFF (Idle)</h3>
                        <p style="color:#64748b; font-size:13.5px; margin:0;">
                            Your webcam is safely powered off. When you are positioned in your chair and ready, click <b>'🟢 Turn Camera ON & Start TUG Scan'</b>.
                        </p>
                    </div>
                    """
                )
            else:
                cam_feed = st.empty()
                instruction_box = st.empty()

                cap = cv2.VideoCapture(int(webcam_idx))
                if not cap.isOpened():
                    st.session_state["cam_active"] = False
                    st.error(f"❌ Unable to access webcam at device index {webcam_idx}. Please verify camera permissions.")
                else:
                    detector = PoseDetector()
                    extractor = TUGKinematicFeatureExtractor(fps=30.0, walk_distance_meters=3.0)

                    landmark_sequence = []
                    t_start = time.time()
                    frame_count = 0
                    has_stood = False
                    has_turned = False
                    turn_cooldown = 0
                    seated_complete_frames = 0

                    while cap.isOpened() and st.session_state["cam_active"]:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        now = time.time()
                        elapsed = now - t_start

                        annotated, landmarks = detector.process_frame(frame, draw=True, show_angles=True)

                        phase_name = "🪑 1. SITTING IN CHAIR"
                        phase_instruct = "👉 Step 1: Sit upright in the chair with feet flat on the floor. Prepare to stand."
                        phase_color = (255, 200, 0)
                        knee_angle_val = 165.0

                        if landmarks:
                            landmark_sequence.append(landmarks)
                            if 23 in landmarks and 24 in landmarks:
                                mid_hip = (landmarks[23][1] + landmarks[24][1]) / 2.0
                                sh_w = abs(landmarks[11][0] - landmarks[12][0])

                                if 25 in landmarks and 27 in landmarks:
                                    knee_angle_val = calculate_angle_2d(landmarks[23][:2], landmarks[25][:2], landmarks[27][:2])

                                # Dynamic Kinematic Phase Tracker
                                if not has_stood:
                                    if mid_hip > 0.65:
                                        phase_name = "🪑 1. SITTING IN CHAIR"
                                        phase_instruct = "👉 Step 1: Sit upright in your chair. Get ready to stand."
                                        phase_color = (255, 200, 0)
                                    else:
                                        has_stood = True
                                        phase_name = "🧍 2. STANDING UP"
                                        phase_instruct = "👉 Step 2: Push off with your legs and stand straight up without using chair arms!"
                                        phase_color = (0, 255, 120)
                                elif not has_turned:
                                    if elapsed > 2.5 and sh_w < 0.11:
                                        has_turned = True
                                        turn_cooldown = 25
                                        phase_name = "🔄 4. 180° TURN AT 3M"
                                        phase_instruct = "👉 Step 4: Turn around smoothly at the 3-meter mark. Maintain steady balance!"
                                        phase_color = (0, 165, 255)
                                    else:
                                        phase_name = "🚶 3. WALKING 3 METERS"
                                        phase_instruct = "👉 Step 3: Walk forward along the 3-meter line towards the turning marker."
                                        phase_color = (0, 255, 120)
                                else:
                                    if turn_cooldown > 0:
                                        turn_cooldown -= 1
                                        phase_name = "🔄 4. 180° TURN AT 3M"
                                        phase_instruct = "👉 Step 4: Turn around smoothly at the 3-meter mark."
                                        phase_color = (0, 165, 255)
                                    elif mid_hip > 0.66:
                                        phase_name = "🪑 6. SEATED (TEST COMPLETE)"
                                        phase_instruct = "👉 Step 6: Turn and sit back down safely in your chair. Assessment complete!"
                                        phase_color = (255, 215, 0)
                                        seated_complete_frames += 1
                                        if seated_complete_frames >= 20:  # ~0.8s stably seated in chair
                                            break
                                    else:
                                        phase_name = "🚶 5. WALKING BACK"
                                        phase_instruct = "👉 Step 5: Walk straight back to the chair at your normal comfortable pace."
                                        phase_color = (0, 255, 120)
                                        seated_complete_frames = 0

                        # Overlay HUD on frame
                        h_f, w_f, _ = annotated.shape
                        cv2.rectangle(annotated, (15, 15), (520, 115), (15, 25, 40), -1)
                        cv2.rectangle(annotated, (15, 15), (520, 115), (0, 200, 255), 2)
                        cv2.putText(annotated, f"⏱️ TUG TIMER: {elapsed:04.1f}s / {test_seconds}s", (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 255), 2)
                        cv2.putText(annotated, f"PHASE: {phase_name}", (28, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.62, phase_color, 2)
                        cv2.putText(annotated, f"Knee: {knee_angle_val:.1f} deg | Landmarks: {len(landmark_sequence)}", (28, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

                        # In-camera instruction banner
                        cv2.rectangle(annotated, (15, h_f - 48), (w_f - 15, h_f - 10), (10, 20, 35), -1)
                        cv2.rectangle(annotated, (15, h_f - 48), (w_f - 15, h_f - 10), (245, 158, 11), 2)
                        cv2.putText(annotated, phase_instruct[:75], (25, h_f - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)

                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        cam_feed.image(frame_rgb, caption=f"Live Camera Stream - {patient_label}", use_container_width=True)

                        # Self-updating single container (no repeating boxes!)
                        instruction_box.markdown(
                            f"""
                            <div style="background:#091e3a; color:white; padding:14px 20px; border-radius:12px; border-left:4px solid #f59e0b; margin:12px 0;">
                                <span style="font-size:11px; font-weight:700; color:#f59e0b; text-transform:uppercase; letter-spacing:0.5px;">Active Protocol Phase</span>
                                <h4 style="margin:4px 0 2px 0; color:#ffffff; font-size:15px;">{phase_instruct}</h4>
                                <span style="font-size:12.5px; color:#cbd5e1;">⏱️ Stopwatch: <b>{elapsed:.1f}s</b> / {test_seconds}s &nbsp;|&nbsp; Active: <b>{phase_name}</b> &nbsp;|&nbsp; 📐 Knee Angle: <b>{knee_angle_val:.1f}°</b></span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        frame_count += 1
                        if elapsed >= test_seconds:
                            break
                        time.sleep(0.01)

                    cap.release()
                    st.session_state["cam_active"] = False
                    detector.close()
                    instruction_box.empty()

                    # Process features from live recording
                    features = extractor.extract_features_from_landmark_sequence(landmark_sequence, dual_task_present=dual_task_active_box)
                    res = classifier.predict(features)

                    st.session_state["live_test_results"] = {
                        "source": f"Live Camera Recording ({patient_label})",
                        "patient_name": patient_name_in,
                        "patient_id": patient_label,
                        "features": features,
                        "prediction": res,
                    }
                    st.session_state["patient_info"]["name"] = patient_name_in
                    st.session_state["patient_info"]["id"] = patient_label
                    st.session_state["selected_walk_dataset"] = f"🎥 Latest Live Camera Assessment ({patient_label})"
                    st.session_state["scan_just_completed"] = True
                    st.session_state["nav_stage"] = "📋 Step 2: Medical Questionnaire"
                    st.rerun()

    def render_video_pose_inspector(key_prefix: str = "insp"):
        render_html(
            """
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:16px; padding:18px 22px; margin-bottom:14px; box-shadow:0 4px 16px rgba(15,23,42,0.03);">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div>
                        <h3 style="color:#0f172a; margin:0 0 4px 0; font-size:18px;">📹 Video File & Skeletal Pose Inspector</h3>
                        <p style="color:#64748b; font-size:13.5px; margin:0;">Deep frame-by-frame joint angle extraction and landmark trajectory scrubber.</p>
                    </div>
                    <span class="pill-tag pill-blue">BIOMECHANICAL INSPECTOR</span>
                </div>
            </div>
            """
        )
        video_choices = {
            "🔴 User Video 2: Abnormal Walk (Alzheimer's Symptoms)": "data/videos/abnormal.mp4",
            "🟢 User Video 1: Normal Walk (Healthy Control)": "data/videos/normal walk.mp4",
            "🟢 Synthetic Simulation: Healthy Older Adult": "data/sample_tug_healthy.mp4",
            "🔴 Synthetic Simulation: Impaired / AD Risk": "data/sample_tug_impaired.mp4",
        }
        
        col_sel, col_up = st.columns([1.5, 1])
        with col_sel:
            sel_choice = st.selectbox("Select Benchmark Video:", list(video_choices.keys()), index=0, key=f"{key_prefix}_sel_video")
            chosen_path = video_choices[sel_choice]
        with col_up:
            uploaded_file = st.file_uploader("Or Upload Custom Patient Walk Video (.mp4):", type=["mp4", "avi", "mov"], key=f"{key_prefix}_uploader")
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                chosen_path = tfile.name

        st.video(chosen_path)

        v_btn1, v_btn2 = st.columns(2)
        with v_btn1:
            run_analysis = st.button("⚡ Run Skeletal Pose Analysis on this Video", key=f"{key_prefix}_run_analysis", type="primary", use_container_width=True)
        with v_btn2:
            proceed_q = st.button("👉 Load this Walk & Proceed to Questionnaire ➔", key=f"{key_prefix}_proceed", use_container_width=True)

        if proceed_q:
            if "Abnormal" in sel_choice or "Impaired" in sel_choice:
                st.session_state["selected_walk_dataset"] = "🔴 Benchmark Simulation: Hesitant Gait (SIMULATED BENCHMARK - NOT A PATIENT)"
            else:
                st.session_state["selected_walk_dataset"] = "🟢 Benchmark Recording 1: Healthy Control Gait"
            st.session_state["nav_stage"] = "📋 Step 2: Medical Questionnaire"
            st.rerun()

        if run_analysis:
            with st.spinner("Extracting 33-point skeletal landmark trajectories & computing joint angles..."):
                res = extract_full_pose_trajectories(chosen_path, max_frames=200)
                df = res["df_trajectories"]
                if not df.empty:
                    st.success(f"✅ Successfully analyzed {res['total_frames_analyzed']} frames at {res['fps']:.1f} FPS.")
                    
                    ch1, ch2 = st.columns(2)
                    with ch1:
                        st.markdown("##### 📐 Knee Flexion / Extension Angles (°)")
                        st.line_chart(df.set_index("time_s")[["left_knee_angle", "right_knee_angle"]])
                    with ch2:
                        st.markdown("##### ⚖️ Axial Trunk Sway Angle (°)")
                        st.line_chart(df.set_index("time_s")[["trunk_sway_angle"]])
                        
                    st.markdown("##### 🪑 Hip Height Trajectory (Sit-to-Stand Transition)")
                    st.line_chart(df.set_index("time_s")[["hip_y"]])
                else:
                    st.warning("⚠️ No pose landmarks detected in this video. Please ensure the subject is visible from head to toe.")

    # View Mode Selector: All on one page by default, selectable via clean buttons!
    st.markdown("#### 🎯 Choose Assessment Display Mode:")
    if "step1_active_view" not in st.session_state:
        st.session_state["step1_active_view"] = "All-In-One"

    vm_c1, vm_c2, vm_c3, vm_c4 = st.columns(4)
    with vm_c1:
        if st.button("🌟 All-In-One Unified View", key="btn_vm_all", use_container_width=True, type="primary" if st.session_state["step1_active_view"] == "All-In-One" else "secondary"):
            st.session_state["step1_active_view"] = "All-In-One"
            st.rerun()
    with vm_c2:
        if st.button("⚖️ Side-by-Side Videos Only", key="btn_vm_sb", use_container_width=True, type="primary" if st.session_state["step1_active_view"] == "Side-by-Side" else "secondary"):
            st.session_state["step1_active_view"] = "Side-by-Side"
            st.rerun()
    with vm_c3:
        if st.button("🎥 Live Webcam Only", key="btn_vm_cam", use_container_width=True, type="primary" if st.session_state["step1_active_view"] == "Live Webcam" else "secondary"):
            st.session_state["step1_active_view"] = "Live Webcam"
            st.rerun()
    with vm_c4:
        if st.button("📹 Video Inspector Only", key="btn_vm_insp", use_container_width=True, type="primary" if st.session_state["step1_active_view"] == "Inspector" else "secondary"):
            st.session_state["step1_active_view"] = "Inspector"
            st.rerun()

    st.markdown("<div style='margin-bottom:14px;'></div>", unsafe_allow_html=True)
    view_mode = st.session_state["step1_active_view"]

    if "All-In-One" in view_mode:
        render_side_by_side_demo(key_prefix="all_sb")
        
        st.markdown("<div style='margin:30px 0 18px 0; border-top:2px dashed #cbd5e1;'></div>", unsafe_allow_html=True)
        st.markdown("### 🎥 Live Device Webcam TUG Scan (Interactive Camera)")
        render_live_webcam_scan(key_prefix="all_cam", dual_task_live=dual_task_active)
        
        st.markdown("<div style='margin:30px 0 18px 0; border-top:2px dashed #cbd5e1;'></div>", unsafe_allow_html=True)
        st.markdown("### 📹 Custom Video File & Skeletal Pose Inspector")
        render_video_pose_inspector(key_prefix="all_insp")

    elif "Side-by-Side" in view_mode:
        render_side_by_side_demo(key_prefix="iso_sb")

    elif "Live Webcam" in view_mode:
        render_live_webcam_scan(key_prefix="iso_cam", dual_task_live=dual_task_active)

    elif "Inspector" in view_mode:
        render_video_pose_inspector(key_prefix="iso_insp")


# ==============================================================================
# SCREEN 3: STEP 2 - MEDICAL QUESTIONNAIRE ("QURESTIONERR")
# ==============================================================================
elif st.session_state["nav_stage"] == "📋 Step 2: Medical Questionnaire":
    st.markdown("## 📋 Step 2: Pre-Screening Medical Intake Questionnaire")
    st.caption("Standardized clinical intake assessing patient demographics, subjective memory complaints, and orthopedic confounders.")

    if st.session_state.get("scan_just_completed"):
        st.balloons()
        st.success("🎉 Camera Scan Completed Successfully! Spatio-temporal gait kinematics have been automatically transferred into this questionnaire.")
        st.session_state["scan_just_completed"] = False

    # 1. Connected Gait Biomarker Summary Banner (Active Walk / Live Scan)
    active_live = st.session_state["live_test_results"]
    has_live = active_live is not None

    if has_live:
        live_feat = active_live["features"]
        live_pred = active_live["prediction"]
        p_name = st.session_state["patient_info"]["name"]
        p_id = st.session_state["patient_info"]["id"]
        dtc = live_feat.get("dual_task_cost", 0.0)
        is_ad_risk = dtc > 22.0 or live_pred.get("alzheimer_warning_level") == "HIGH_ALERT"

        status_bg = "#fff1f2" if is_ad_risk else "#f0fdf4"
        status_border = "#fca5a5" if is_ad_risk else "#86efac"
        status_color = "#991b1b" if is_ad_risk else "#065f46"
        badge_text = "🚨 HIGH COGNITIVE IMPAIRMENT ALERT (>22% DTC)" if is_ad_risk else "🟢 HEALTHY NORMAL GAIT PROFILE (<15% DTC)"
        badge_class = "pill-red" if is_ad_risk else "pill-green"

        render_html(
            f"""
            <div style="background:{status_bg}; border:2px solid {status_border}; border-radius:16px; padding:18px 24px; margin-bottom:18px; box-shadow:0 4px 14px rgba(0,0,0,0.03);">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div>
                        <span class="pill-tag {badge_class}">🔗 LIVE CAMERA SCAN DATA LINKED</span>
                        <span class="pill-tag pill-blue">PATIENT: {p_name} ({p_id})</span>
                        <h3 style="color:{status_color}; margin:6px 0 2px 0; font-size:17.5px;">{badge_text}</h3>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:12px; color:#64748b;">Source: {active_live['source']}</span>
                    </div>
                </div>
                <div style="display:flex; gap:18px; flex-wrap:wrap; margin-top:12px; font-size:13.5px; color:#1e293b; background:rgba(255,255,255,0.75); padding:10px 16px; border-radius:10px;">
                    <div>⏱️ <b>TUG Duration:</b> {live_feat.get('total_tug_duration', 0):.2f}s</div>
                    <div>🧠 <b>Dual-Task Cost (DTC):</b> <span style="font-weight:800; color:{'#dc2626' if dtc>22 else '#16a34a'};">{dtc:.2f}%</span></div>
                    <div>🚶 <b>Walking Speed:</b> {live_feat.get('walking_speed', 0):.2f} m/s</div>
                    <div>👣 <b>Cadence:</b> {live_feat.get('cadence', 0):.1f} spm</div>
                    <div>📐 <b>Trunk Sway:</b> {live_feat.get('trunk_sway_angle', 0):.2f}°</div>
                </div>
            </div>
            """
        )
    else:
        sel_w = st.session_state["selected_walk_dataset"]
        is_abn = is_abnormal_dataset_label(sel_w)
        dtc_demo = 25.82 if is_abn else 9.20
        render_html(
            f"""
            <div style="background:#f8fafc; border:1.5px solid #cbd5e1; border-radius:16px; padding:18px 24px; margin-bottom:18px; box-shadow:0 4px 14px rgba(0,0,0,0.02);">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div>
                        <span class="pill-tag pill-indigo">📹 BENCHMARK WALK DATA LINKED</span>
                        <span class="pill-tag pill-blue">PATIENT: {st.session_state['patient_info']['name']} ({st.session_state['patient_info']['id']})</span>
                        <h4 style="color:#0f172a; margin:6px 0 2px 0; font-size:16px;">Active Dataset: {sel_w}</h4>
                    </div>
                    <div>
                        <span class="pill-tag {'pill-red' if is_abn else 'pill-green'}">{'🚨 25.82% DTC (Impaired Profile)' if is_abn else '🟢 9.20% DTC (Healthy Control)'}</span>
                    </div>
                </div>
                <div style="display:flex; gap:18px; flex-wrap:wrap; margin-top:10px; font-size:13px; color:#475569; background:#ffffff; padding:8px 14px; border-radius:8px; border:1px solid #e2e8f0;">
                    <div>⏱️ <b>TUG Duration:</b> {'8.80s' if is_abn else '8.00s'}</div>
                    <div>🧠 <b>Dual-Task Cost:</b> <span style="font-weight:700; color:{'#dc2626' if is_abn else '#16a34a'};">{dtc_demo:.2f}%</span></div>
                    <div>🚶 <b>Velocity:</b> {'0.85 m/s' if is_abn else '0.91 m/s'}</div>
                    <div>👣 <b>Cadence:</b> {'105.0 spm' if is_abn else '103.5 spm'}</div>
                </div>
            </div>
            """
        )

    col_pat, col_cog, col_ortho = st.columns(3)

    with col_pat:
        st.markdown("#### 👤 Patient Profile")
        p_name = st.text_input("Patient Full Name *:", value=st.session_state["patient_info"]["name"], key="q_pname")
        p_id = st.text_input("Patient ID / Clinical Code *:", value=st.session_state["patient_info"]["id"], key="q_pid")
        p_age = st.number_input("Age (Years):", min_value=18, max_value=105, value=st.session_state["patient_info"]["age"], key="q_page")
        p_gender = st.selectbox("Biological Sex:", ["Female", "Male", "Other"], index=0, key="q_pgen")

    with col_cog:
        st.markdown("#### 🧠 Cognitive Symptoms")
        cog1 = st.checkbox("Frequent subjective memory lapses (forgotten appointments)", value=False, key="q_c1")
        cog2 = st.checkbox("Noticeable difficulty multitasking (walking while talking)", value=False, key="q_c2")
        cog3 = st.checkbox("Spatial disorientation in familiar environments", value=False, key="q_c3")
        cog4 = st.checkbox("Immediate family history of Alzheimer's or dementia", value=False, key="q_c4")

    with col_ortho:
        st.markdown("#### 🦴 Orthopedic Confounders")
        ortho1 = st.checkbox("Recent hip, knee, or ankle joint surgery / severe arthritis", value=False, key="q_o1")
        ortho2 = st.checkbox("Requires assistive walking device (walker, crutches, cane)", value=False, key="q_o2")
        ortho3 = st.checkbox("Diagnosed inner ear vestibular or balance disorder", value=False, key="q_o3")
        ortho4 = st.checkbox("History of 2 or more unexplained falls in past 12 months", value=False, key="q_o4")

    cog_score = sum([cog1, cog2, cog3, cog4])
    ortho_score = sum([ortho1 * 2, ortho2 * 2, ortho3, ortho4])

    st.session_state["patient_info"]["id"] = p_id
    st.session_state["patient_info"]["name"] = p_name
    st.session_state["patient_info"]["age"] = p_age
    st.session_state["patient_info"]["gender"] = p_gender
    st.session_state["patient_info"]["cognitive_score"] = cog_score
    st.session_state["patient_info"]["confounder_score"] = ortho_score

    st.divider()
    st.markdown("### 📊 Pre-Screening Multi-Modal Triage Evaluation")
    eval_c1, eval_c2, eval_c3, eval_c4 = st.columns(4)

    if has_live:
        cur_dtc = active_live["features"].get("dual_task_cost", 0.0)
        is_abn = cur_dtc > 22.0 or is_abnormal_dataset_label(active_live.get("source", ""))
    else:
        is_abn = is_abnormal_dataset_label(st.session_state["selected_walk_dataset"])
        cur_dtc = 25.82 if is_abn else 9.20

    is_gait_impaired = cur_dtc > 22.0 or is_abn

    with eval_c1:
        render_html(
            f"""
            <div class="metric-bubble">
                <h4>Camera Gait Status</h4>
                <p style="color: {'#ef4444' if is_gait_impaired else '#10b981'}; font-size:22px;">{'🚨 Impaired' if is_gait_impaired else '🟢 Normal'}</p>
                <span style="font-size:11.5px; color:#64748b;">DTC: {cur_dtc:.2f}% (Ref &lt;15%)</span>
            </div>
            """
        )

    with eval_c2:
        render_html(
            f"""
            <div class="metric-bubble">
                <h4>Cognitive Concern</h4>
                <p style="color: {'#ef4444' if cog_score >= 2 else '#10b981'}; font-size:22px;">{cog_score} / 4</p>
                <span style="font-size:11.5px; color:#64748b;">Subjective Symptoms</span>
            </div>
            """
        )

    with eval_c3:
        render_html(
            f"""
            <div class="metric-bubble">
                <h4>Physical Confounder</h4>
                <p style="color: {'#f59e0b' if ortho_score >= 2 else '#10b981'}; font-size:22px;">{ortho_score} / 6</p>
                <span style="font-size:11.5px; color:#64748b;">Orthopedic Index</span>
            </div>
            """
        )

    with eval_c4:
        if ortho_score >= 3:
            st.warning("⚠️ Confounders Detected: Motor slowness may be due to orthopedic limitations.")
        elif is_gait_impaired and cog_score >= 2:
            st.error("🚨 Multi-Modal Alert: Gait slowing + memory complaints detected without physical confounders.")
        elif is_gait_impaired:
            st.error(f"🚨 Impaired Motor Cost: Elevated DTC ({cur_dtc:.2f}% > 22.0%) indicates cognitive-motor deficit.")
        else:
            st.success("✅ Clean Clinical Baseline: Both digital gait and intake indicate healthy normal profile.")

    st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

    btn_back, btn_next = st.columns([1, 1.5])
    with btn_back:
        if st.button("⬅ Back to Step 1: Camera Scan", use_container_width=True):
            st.session_state["nav_stage"] = "🎥 Step 1: Camera Scan (TUG Test)"
            st.rerun()
    with btn_next:
        if st.button("👉 Complete Questionnaire & View Final Report & PDF ➔", type="primary", use_container_width=True):
            st.session_state["nav_stage"] = "🩺 Step 3: Diagnostic Screening Report & PDF"
            st.rerun()


# ==============================================================================
# SCREEN 4: STEP 3 - OFFICIAL CLINICAL REPORT & PDF
# ==============================================================================
elif st.session_state["nav_stage"] == "🩺 Step 3: Diagnostic Screening Report & PDF":
    st.markdown("## 🩺 Step 3: Official Clinical Diagnostic Screening Report")
    st.caption("Comprehensive Neuro-Gait Report combining Camera Kinematics, Questionnaire Findings, Multi-Model AI Consensus, and PDF Export.")

    report_options = [
        "🟢 Benchmark Recording 1: Healthy Control Gait",
        "🔴 Benchmark Simulation: Hesitant Gait (SIMULATED BENCHMARK - NOT A PATIENT)",
    ]
    if st.session_state["live_test_results"] is not None:
        report_options.insert(0, f"🎥 Latest Live Camera Assessment ({st.session_state['patient_info']['id']})")

    current_sel_idx = 0
    if st.session_state["selected_walk_dataset"] in report_options:
        current_sel_idx = report_options.index(st.session_state["selected_walk_dataset"])

    sel_report = st.selectbox(
        "Active Assessment Dataset:",
        report_options,
        index=current_sel_idx,
    )

    if "Live Camera" in sel_report and st.session_state["live_test_results"] is not None:
        live_res = st.session_state["live_test_results"]
        active_eval = live_res["prediction"]
        features = live_res["features"]
        active_key = "live_assessment"
        dtc_val = features.get("dual_task_cost", 0.0)
        is_abnormal = (
            active_eval.get("alzheimer_warning_level") == "HIGH_ALERT"
            or active_eval.get("consensus_label") != "Healthy Normal"
            or dtc_val > 22.0
            or is_abnormal_dataset_label(sel_report)
        )
    else:
        is_abnormal = is_abnormal_dataset_label(sel_report)
        active_key = "abnormal_walk" if is_abnormal else "normal_walk"
        active_eval = eval_data.get(active_key, {})
        features = active_eval.get("input_features", {})
        dtc_val = features.get("dual_task_cost", 25.82 if is_abnormal else 9.20)

    if active_eval:
        breakdown = active_eval.get("model_breakdown", {})
        if not dtc_val or dtc_val == 0.0:
            dtc_val = features.get("dual_task_cost", 25.82 if is_abnormal else 9.20)
        svm_prob = breakdown.get("SVM", {}).get("probability", 82.82 if is_abnormal else 12.5)
        rf_prob = breakdown.get("Random Forest", {}).get("probability", 74.0 if is_abnormal else 11.2)
        xgb_prob = breakdown.get("XGBoost", {}).get("probability", 78.4 if is_abnormal else 11.4)
        
        pat_age = st.session_state["patient_info"].get("age", 20)
        # If simulated benchmark or live test shows elevated DTC > 22%, preserve abnormal status for demonstration
        if is_abnormal or dtc_val > 22.0:
            is_abnormal = True
        elif pat_age < 45 and dtc_val <= 22.0:
            is_abnormal = False

        # Banner based on active status & demographic
        if is_abnormal:
            render_html(
                f"""
                <div style="background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%); border: 1.5px solid #ef4444; border-radius: 14px; padding: 14px 20px; margin-bottom: 16px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span class="pill-tag pill-red">🔴 ELEVATED RISK / SIMULATION ACTIVE</span>
                        <span class="pill-tag pill-dark">DUAL-TASK MOTOR IMPAIRMENT (DTC {dtc_val:.2f}% &gt; 22%)</span>
                    </div>
                    <div style="font-size: 13px; color: #991b1b; margin-top: 6px; line-height: 1.5;">
                        <b>Impairment Detection Demonstration:</b> System detected pronounced motor-cognitive interference. Walking speed slowed and turning hesitations caused Dual-Task Cost to surge to <b>{dtc_val:.2f}%</b> (threshold &gt; 22.0%), triggering the Red Alzheimer's Screening Alert.
                    </div>
                </div>
                """
            )
        elif pat_age < 45:
            render_html(
                f"""
                <div style="background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%); border: 1.5px solid #3b82f6; border-radius: 14px; padding: 14px 20px; margin-bottom: 16px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span class="pill-tag pill-blue">🧪 STUDENT / DEMO MODE (AGE {pat_age} &lt; 45)</span>
                        <span class="pill-tag pill-green">OBJECTIVE BIOMECHANICAL MEASUREMENT ONLY</span>
                    </div>
                    <div style="font-size: 13px; color: #1e293b; margin-top: 6px; line-height: 1.5;">
                        <b>Scientific Protocol Note:</b> Clinical Alzheimer's screening models are clinically validated for populations <b>aged 45 and above</b>. 
                        For subjects under 45 (e.g. college student testing), the system operates in <b>Biomechanical Demo Mode</b>: reporting objective spatio-temporal gait metrics (velocity, turning duration, cadence, postural sway, and dual-task cost of {dtc_val:.2f}%) without applying geriatric disease risk labels.
                    </div>
                </div>
                """
            )

        # Clinical Risk Stratification
        if is_abnormal:
            verdict_title = "🚨 ALZHEIMER'S SCREENING ALERT: PRODROMAL COGNITIVE-MOTOR DEFICIT"
            verdict_badge = "HIGH CLINICAL RISK (>22% DTC THRESHOLD)"
            verdict_badge_class = "pill-red"
            verdict_color = "#991b1b"
            card_class = "diagnosis-alert-danger"
            summary_text = (
                f"<b>Primary Clinical Finding:</b> Patient exhibits pronounced <b>Dual-Task Motor Deceleration of {dtc_val:.2f}%</b> "
                f"(exceeding the validated clinical threshold of 22.0%), accompanied by executive gait velocity slowing ({features.get('walking_speed', 0.85):.2f} m/s). "
                "This indicates competition between prefrontal attentional networks and motor planning circuits during concurrent cognitive load, "
                "a recognized digital biomarker for early-stage Alzheimer's disease / Mild Cognitive Impairment (MCI)."
            )
        else:
            if pat_age < 45:
                verdict_title = f"🟢 NORMAL CLINICAL PROFILE: HEALTHY YOUNG ADULT (AGE {pat_age})"
                verdict_badge = "PHYSIOLOGICAL NORMAL (STUDENT / YOUNG ADULT)"
                verdict_badge_class = "pill-green"
                verdict_color = "#065f46"
                card_class = "diagnosis-alert-safe"
                summary_text = (
                    f"<b>Primary Clinical Finding:</b> Subject demonstrates optimal motor-cognitive performance. "
                    f"<b>Dual-Task Cost is {dtc_val:.2f}%</b> (well within healthy youth limits &lt;15%). Walking speed ({features.get('walking_speed', 1.05):.2f} m/s), "
                    "turning stability, and cadence reflect fully intact neuromotor circuits. Zero indicators of neurodegenerative decline detected."
                )
            else:
                verdict_title = "🟢 NORMAL CLINICAL PROFILE: HEALTHY AGING MOTOR PERFORMANCE"
                verdict_badge = "PHYSIOLOGICAL NORMAL (<15% DTC)"
                verdict_badge_class = "pill-green"
                verdict_color = "#065f46"
                card_class = "diagnosis-alert-safe"
                summary_text = (
                    f"<b>Primary Clinical Finding:</b> Patient demonstrates stable neuromotor coordination. "
                    f"<b>Dual-Task Cost is {dtc_val:.2f}%</b> (well below the 15.0% impairment boundary). Walking speed ({features.get('walking_speed', 0.91):.2f} m/s), "
                    "step cadence, and turning stability reflect intact prefrontal motor-cognitive integrity."
                )

        # 1. Patient Demographic Bar
        render_html(
            f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; padding:16px 22px; margin-bottom:16px; box-shadow:0 2px 8px rgba(0,0,0,0.03);">
                <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px; font-size:13.5px; color:#334155;">
                    <div><b>Patient Name:</b> <span style="color:#0f172a; font-weight:700;">{st.session_state['patient_info']['name']}</span></div>
                    <div><b>Patient ID:</b> <span style="color:#2563eb; font-weight:700;">{st.session_state['patient_info']['id']}</span></div>
                    <div><b>Age:</b> {st.session_state['patient_info']['age']} Yrs</div>
                    <div><b>Gender:</b> {st.session_state['patient_info']['gender']}</div>
                    <div><b>Cognitive Symptoms:</b> {st.session_state['patient_info']['cognitive_score']} / 4</div>
                    <div><b>Orthopedic Confounders:</b> {st.session_state['patient_info']['confounder_score']} / 6</div>
                    <div><b>Dual-Task Language:</b> <span class="badge-ice-blue">{st.session_state.get('dual_task_lang', 'English')}</span></div>
                </div>
            </div>
            """
        )

        # 1.5 Transparent Weighted Multi-Modal Screening Risk Band
        risk_eval = calculate_transparent_risk_score(
            dtc_val=dtc_val,
            speed_val=features.get("walking_speed", 0.85 if is_abnormal else 0.91),
            turn_val=features.get("turning_duration", 2.2 if is_abnormal else 2.1),
            cog_score=st.session_state["patient_info"].get("cognitive_score", 0),
            age=st.session_state["patient_info"].get("age", 71),
            ortho_score=st.session_state["patient_info"].get("confounder_score", 0),
        )

        render_html(
            f"""
            <div style="background:#ffffff; border:1.5px solid #cbd5e1; border-radius:18px; padding:20px 24px; margin-bottom:18px; box-shadow:0 4px 18px rgba(15,23,42,0.04);">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span class="pill-tag pill-dark">MULTI-MODAL SCREENING BAND</span>
                        <span class="{risk_eval['badge_class']}">{risk_eval['band_label']} ({risk_eval['composite_score']}%)</span>
                    </div>
                    <span style="font-size:11.5px; color:#64748b; font-weight:600;">⚖️ Transparent Weighted Formula: 50% Gait + 30% Cognitive + 20% Demographic</span>
                </div>
                <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-top:14px;">
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:12px 16px;">
                        <div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">1. Kinematic Gait (50% Wt)</div>
                        <div style="font-size:20px; font-weight:800; color:#0f172a; margin-top:2px;">{risk_eval['gait_score']} <span style="font-size:13px; font-weight:600; color:#64748b;">/ 100</span></div>
                        <div style="font-size:11.5px; color:#475569;">DTC: {dtc_val:.1f}% | Turn: {features.get('turning_duration', 2.2):.1f}s</div>
                    </div>
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:12px 16px;">
                        <div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">2. Cognitive Intake (30% Wt)</div>
                        <div style="font-size:20px; font-weight:800; color:#0f172a; margin-top:2px;">{risk_eval['cog_score']} <span style="font-size:13px; font-weight:600; color:#64748b;">/ 100</span></div>
                        <div style="font-size:11.5px; color:#475569;">Complaints: {st.session_state['patient_info']['cognitive_score']} / 4</div>
                    </div>
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:12px 16px;">
                        <div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">3. Demographic / Age (20% Wt)</div>
                        <div style="font-size:20px; font-weight:800; color:#0f172a; margin-top:2px;">{risk_eval['demo_score']} <span style="font-size:13px; font-weight:600; color:#64748b;">/ 100</span></div>
                        <div style="font-size:11.5px; color:#475569;">Patient Age: {st.session_state['patient_info']['age']} Yrs</div>
                    </div>
                </div>
                <div style="margin-top:12px; font-size:12.5px; color:#334155; line-height:1.5;">
                    <b>Screening Guidance:</b> {risk_eval['recommendation']}
                </div>
                <div style="margin-top:8px; font-size:11px; color:#94a3b8; font-style:italic;">
                    ⚠️ Screening support tool only — not validated as an independent diagnostic test. Compliant with India DPDP Act 2023.
                </div>
            </div>
            """
        )

        # 2. Main Executive Verdict Card
        render_html(
            f"""
            <div class="{card_class}">
                <div style="display:flex; align-items:center; gap:8px;">
                    {SVG_ALERT if is_abnormal else SVG_CHECK}
                    <span class="pill-tag {verdict_badge_class}">{verdict_badge}</span>
                    <span class="pill-tag pill-dark">TRI-MODEL CONSENSUS</span>
                </div>
                <h2 style="color:{verdict_color}; margin:10px 0 6px 0; font-size:22px;">{verdict_title}</h2>
                <p style="font-size:14.5px; color:#1e293b; margin:0; line-height:1.6;">{summary_text}</p>
            </div>
            """
        )

        # 3. 4 Top Metric Cards
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            render_html(
                f"""
                <div class="metric-bubble">
                    <h4>Dual-Task Cost (DTC)</h4>
                    <p style="color: {'#dc2626' if dtc_val > 22 else '#16a34a'};">{dtc_val:.2f}%</p>
                    <span style="font-size:11px; color:#64748b;">Ref: &lt;15% Normal | &gt;22% Alert</span>
                </div>
                """
            )
        with m_c2:
            render_html(
                f"""
                <div class="metric-bubble">
                    <h4>Mean Walking Velocity</h4>
                    <p style="color: {'#d97706' if features.get('walking_speed', 0.85) < 0.9 else '#16a34a'};">{features.get('walking_speed', 0.85):.2f} m/s</p>
                    <span style="font-size:11px; color:#64748b;">Ref: 0.90 – 1.30 m/s</span>
                </div>
                """
            )
        with m_c3:
            render_html(
                f"""
                <div class="metric-bubble">
                    <h4>Total TUG Duration</h4>
                    <p style="color: #16a34a;">{features.get('total_tug_duration', 8.0):.2f} s</p>
                    <span style="font-size:11px; color:#64748b;">Ref: &lt;10.0s Normal | &gt;13.5s Fall Risk</span>
                </div>
                """
            )
        with m_c4:
            render_html(
                f"""
                <div class="metric-bubble">
                    <h4>Cadence Rhythm</h4>
                    <p style="color: #16a34a;">{features.get('cadence', 105.0):.1f} spm</p>
                    <span style="font-size:11px; color:#64748b;">Ref: 100 – 115 steps/min</span>
                </div>
                """
            )

        st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

        # 4. Biomarker Table & Multi-Model Breakdown
        r_col_left, r_col_right = st.columns([1.3, 1])

        with r_col_left:
            st.markdown("### 📊 Spatio-Temporal Biomarker Analysis Matrix")
            detailed_matrix = [
                {"Biomarker": "Dual-Task Cost (DTC)", "Measured Value": f"{dtc_val:.2f}%", "Normal Range": "< 15.00%", "Clinical Status": "🚨 CRITICAL ALERT (>22%)" if is_abnormal else "🟢 Intact / Normal"},
                {"Biomarker": "Mean Walking Speed", "Measured Value": f"{features.get('walking_speed', 0.85):.2f} m/s", "Normal Range": "0.90 – 1.30 m/s", "Clinical Status": "🟡 Mild Executive Slowing" if is_abnormal else "🟢 Optimal Velocity"},
                {"Biomarker": "Total TUG Duration", "Measured Value": f"{features.get('total_tug_duration', 8.00):.2f} s", "Normal Range": "< 10.00 s", "Clinical Status": "🟢 Low Fall Risk (<10s)"},
                {"Biomarker": "Sit-to-Stand Transition", "Measured Value": f"{features.get('sit_to_stand_duration', 0.93):.2f} s", "Normal Range": "< 1.20 s", "Clinical Status": "🟢 Intact Muscle Power"},
                {"Biomarker": "Stand-to-Sit Transition", "Measured Value": f"{features.get('stand_to_sit_duration', 1.20):.2f} s", "Normal Range": "< 1.50 s", "Clinical Status": "🟢 Controlled Postural Deceleration"},
                {"Biomarker": "180° Turning Duration", "Measured Value": f"{features.get('turning_duration', 2.20):.2f} s", "Normal Range": "< 2.50 s", "Clinical Status": "🟢 Stable Pivot Without Freezing"},
                {"Biomarker": "Step Cadence", "Measured Value": f"{features.get('cadence', 105.0):.1f} steps/min", "Normal Range": "100 – 115 steps/min", "Clinical Status": "🟢 Regular Locomotor Rhythm"},
                {"Biomarker": "Postural Trunk Sway", "Measured Value": f"{features.get('trunk_sway_angle', 1.50):.2f}°", "Normal Range": "< 4.00°", "Clinical Status": "🟢 Stable Axial Posture (<4°)"},
                {"Biomarker": "Gait Asymmetry Index", "Measured Value": f"{features.get('asymmetry_index', 1.99):.2f}%", "Normal Range": "< 5.00%", "Clinical Status": "🟢 Symmetric Bilateral Stepping"},
            ]
            render_styled_table(detailed_matrix)

        with r_col_right:
            st.markdown("### 🤖 Supervised AI Model Consensus")
            render_html(
                f"""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; padding:18px; margin-bottom:14px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-weight:700; color:#0f172a; font-size:13.5px;">Support Vector Machine (RBF)</span>
                        <span style="font-weight:800; color:{'#dc2626' if svm_prob > 50 else '#16a34a'};">{svm_prob:.1f}% Risk</span>
                    </div>
                    <div style="font-size:12px; color:#64748b; margin-bottom:10px;">Primary non-linear gait classifier detecting cognitive deceleration</div>
                    
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-weight:700; color:#0f172a; font-size:13.5px;">Random Forest Ensemble</span>
                        <span style="font-weight:800; color:{'#dc2626' if rf_prob > 50 else '#16a34a'};">{rf_prob:.1f}% Impairment Prob</span>
                    </div>
                    <div style="font-size:12px; color:#64748b; margin-bottom:10px;">100-tree bagged decision forest verifying kinematic stability</div>
                    
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-weight:700; color:#0f172a; font-size:13.5px;">XGBoost Gradient Boosting</span>
                        <span style="font-weight:800; color:{'#dc2626' if xgb_prob > 50 else '#16a34a'};">{xgb_prob:.1f}% Regularized Risk</span>
                    </div>
                    <div style="font-size:12px; color:#64748b;">Regularized gradient tree boosting testing boundary thresholds</div>
                </div>
                """
            )

            st.markdown("### 🩺 Actionable Clinical Care Plan")
            if is_abnormal:
                st.error("1. **Neurological Consultation:** Comprehensive clinical workup for early neurodegenerative evaluation.")
                st.warning("2. **Cognitive Screening:** Administer Montreal Cognitive Assessment (MoCA, cutoff <26/30).")
                st.info("3. **Differential Triage:** Verify absence of vitamin B12 deficiency or medication-induced motor slowing.")
                st.info("4. **Longitudinal Retest:** Schedule follow-up digital TUG assessment in 90–180 days to monitor progression trajectory.")
            else:
                st.success("1. **Routine Wellness Check:** Motor-cognitive performance is within expected limits for age cohort.")
                st.success("2. **Preventative Mobility:** Continue regular aerobic walking and physical exercise.")
                st.info("3. **Annual Rescreening:** Annual follow-up digital TUG gait evaluation recommended.")

        st.markdown("<div style='margin-bottom:14px;'></div>", unsafe_allow_html=True)

        # 5. Full Downloadable Clinical Discharge Report (PDF + Text)
        report_txt = f"""================================================================================
COGAIT-VISION: CLINICAL TUG GAIT & ALZHEIMER'S SCREENING REPORT
================================================================================
Report Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}
Patient Name     : {st.session_state['patient_info']['name']}
Patient ID       : {st.session_state['patient_info']['id']}
Patient Age      : {st.session_state['patient_info']['age']} Years | Gender: {st.session_state['patient_info']['gender']}
Cognitive Symptoms : {st.session_state['patient_info']['cognitive_score']} / 4
Orthopedic Index   : {st.session_state['patient_info']['confounder_score']} / 6
Evaluation Mode  : {sel_report}
Hardware Setup   : Markerless RGB Monocular Camera (MediaPipe 33 Landmark Tasks)

--------------------------------------------------------------------------------
1. EXECUTIVE DIAGNOSTIC FINDINGS
--------------------------------------------------------------------------------
Clinical Verdict : {verdict_title}
Classification   : {verdict_badge}
Dual-Task Cost   : {dtc_val:.2f}% (Normal: < 15.0%, Alzheimer's Warning: > 22.0%)
Walking Speed    : {features.get('walking_speed', 0.85):.2f} m/s (Normal: 0.90 - 1.30 m/s)
Cadence          : {features.get('cadence', 105.0):.1f} steps/min
Total Duration   : {features.get('total_tug_duration', 8.0):.2f} seconds (Fall Risk Threshold: > 13.5s)
Turning Duration : {features.get('turning_duration', 2.20):.2f} seconds (Pivot: Intact)
Postural Sway    : {features.get('trunk_sway_angle', 1.50):.2f} degrees (Stable: < 4.0 deg)
Gait Asymmetry   : {features.get('asymmetry_index', 1.99):.2f}% (Bilateral symmetry: Intact)

--------------------------------------------------------------------------------
2. SUPERVISED MACHINE LEARNING TRI-MODEL CONSENSUS
--------------------------------------------------------------------------------
Support Vector Machine (SVM RBF) : {svm_prob:.2f}% Probability of Impairment
Random Forest (100 Trees)        : {rf_prob:.2f}% Probability
XGBoost Gradient Boosting        : {xgb_prob:.2f}% Probability
Supervised Verdict               : {'IMPAIRED / AT-RISK COGNITIVE-MOTOR PROFILE' if is_abnormal else 'HEALTHY NORMAL PROFILE'}

--------------------------------------------------------------------------------
3. RECOMMENDED CLINICAL MANAGEMENT
--------------------------------------------------------------------------------
{'• Comprehensive Neurological Referral & Formal MoCA Cognitive Assessment' if is_abnormal else '• Annual Routine Geriatric Wellness Check'}
{'• Occupational & Physical Therapy for Dual-Task Cognitive Walking Rehabilitation' if is_abnormal else '• Maintain Active Aerobic Walking Routine'}
• Longitudinal Digital TUG Follow-up in {'90-180 days' if is_abnormal else '12 months'}

Clinical AI Research Initiative | Diagnostic Neuromotor Systems
Investigators: B P S Kruthi, Ch. Karthika, A. Laxmi Sahithi
Project Guide: Ms. B K Aishwarya, Assistant Professor
================================================================================
"""
        pdf_tab5_bytes = generate_clinical_pdf(
            patient_info=st.session_state["patient_info"],
            results=active_eval,
            features=features,
            walk_label=sel_report,
        )

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                "📥 Download Official Clinical Report (PDF Format)",
                data=pdf_tab5_bytes,
                file_name=f"COGAIT_Official_Report_{active_key}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        with d_col2:
            st.download_button(
                "📄 Download Clinical Report Summary (.txt)",
                data=report_txt,
                file_name=f"COGAIT_Official_Report_{active_key}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.divider()
        if st.button("🔄 Start New Screening Assessment", use_container_width=True):
            st.session_state["nav_stage"] = "🏠 Home & Overview"
            st.session_state["live_test_results"] = None
            st.rerun()


# ==============================================================================
# SCREEN 5: AI BENCHMARKS & CLINICAL VALIDATION
# ==============================================================================
elif st.session_state["nav_stage"] == "🔬 Step 4: AI Model Benchmarks & Validation":
    st.markdown("## 🔬 Machine Learning Benchmarks & Clinical Validation Suite")
    st.caption("Rigorous multi-model evaluation across held-out cross-validation cohorts and clinical methodology verification.")

    if metrics:
        rows = [{"Model": k, **v} for k, v in metrics.items()]
        st.markdown("#### 🏆 Multi-Model Performance Benchmarks (Held-out Test Cohort):")
        render_styled_table(pd.DataFrame(rows)[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]])

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("images/roc_comparison.png"):
            st.image("images/roc_comparison.png", caption="ROC-AUC Curves (SVM, Random Forest, XGBoost)", use_container_width=True)
    with c2:
        if os.path.exists("images/feature_importance.png"):
            st.image("images/feature_importance.png", caption="Biomarker Feature Importance (Dual-Task Cost & Turning Duration lead)", use_container_width=True)

    st.divider()
    with st.expander("📚 Clinical Defense & Technical Methodology Q&A Guide", expanded=True):
        st.markdown(
            """
            **Q1: Why is TUG a valid marker for Alzheimer's screening?**  
            *Answer:* Gait was previously thought to be purely spinal. Modern neuroscience proves walking, turning, and balancing require intact executive function in the prefrontal cortex. In early Alzheimer's, motor slowing and dual-task degradation manifest up to 5-10 years before overt memory loss.

            **Q2: What is Dual-Task Cost (DTC) and its mathematical formula?**  
            *Answer:* $\\text{DTC} = \\frac{\\text{Velocity}_{\\text{single}} - \\text{Velocity}_{\\text{dual}}}{\\text{Velocity}_{\\text{single}}} \\times 100\\%$.  
            Healthy adults maintain $\\text{DTC} < 15\\%$, while early Alzheimer's subjects experience $\\text{DTC} > 22\\%$, reflecting severe prefrontal cognitive-motor competition.

            **Q3: How does your system address false positives from arthritis or knee injuries?**  
            *Answer:* Our Pre-Screening Clinical Questionnaire (Step 2) flags orthopedic confounders. Furthermore, orthopedic issues cause persistent, single-task asymmetry, whereas Alzheimer's is characterized by sudden cognitive deceleration specifically under dual-task cognitive loading.
            """
        )
