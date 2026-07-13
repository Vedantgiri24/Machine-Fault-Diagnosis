import streamlit as st
import tensorflow as tf
import numpy as np
import time

# ==========================================================
#  PROJECT: Machine Fault Detection using CNN
# ==========================================================

PROJECT_INFO = {
    "name"         : "Machine Fault Diagnosis Using Deep Learning Approach",
    "version"      : "1.0.0",
    "description"  : "Vibration signal image classification",
    "classes"      : ["Bearing Fault","Bent Shaft","Foundation Looseness","Healthy","Misalignment"],
    "channels"     : ["CH1", "CH2", "CH3"],
    "img_size"     : (224, 224),
    "num_classes"  : 5,
    "framework"    : "TensorFlow / Keras",
}

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
import os
MODEL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.keras")
IMG_H, IMG_W = 128, 256

CLASS_NAMES = [
    "Bearing Fault",
    "Bent Shaft",
    "Foundation Looseness",
    "Healthy",
    "Misalignment",
]

CLASS_INFO = {
    "Bearing Fault":
        "A bearing fault refers to damage or defects in the rolling elements, "
        "inner race, or outer race of a bearing. This causes periodic impulses "
        "in the vibration signal at characteristic defect frequencies (BPFO, BPFI, BSF, FTF).",
    "Bent Shaft":
        "A bent shaft causes excessive vibration at 1× and 2× the running speed. "
        "It leads to unbalanced rotational forces, increased bearing load, and "
        "accelerated wear of connected components.",
    "Foundation Looseness":
        "Foundation looseness occurs when the machine base or structural mounts "
        "are not properly secured. This creates non-linear vibration patterns "
        "and can cause secondary damage if left unaddressed.",
    "Healthy":
        "The machine is operating under normal healthy conditions. No faults "
        "detected in the vibration signal. Routine monitoring and scheduled "
        "maintenance is recommended to maintain this condition.",
    "Misalignment":
        "Shaft misalignment occurs when two coupled shafts are not collinear. "
        "Angular or parallel misalignment generates high vibration at 1× and 2× "
        "frequencies and causes premature bearing and coupling failure.",
}

CLASS_ACTION = {
    "Bearing Fault":
        "🔧 Schedule immediate bearing inspection. Check lubrication levels and "
        "bearing clearances. Replace damaged bearing within the next maintenance window. "
        "Monitor temperature and vibration amplitude until replacement.",
    "Bent Shaft":
        "🔧 Shut down the machine for shaft inspection. Perform dial-indicator runout "
        "measurement. Replace or straighten the shaft before resuming operation. "
        "Inspect associated couplings and bearings for secondary damage.",
    "Foundation Looseness":
        "🔧 Inspect all anchor bolts and mounting hardware. Re-torque foundation bolts "
        "to specification. Check for cracks in the machine base or mounting surface. "
        "Perform resonance test after re-tightening.",
    "Healthy":
        "✅ No immediate action required. Continue routine vibration monitoring as per "
        "maintenance schedule. Log this reading for trend analysis and baseline comparison.",
    "Misalignment":
        "🔧 Perform precision shaft alignment using laser alignment tools. Check coupling "
        "condition and re-align to manufacturer tolerance before next operation. "
        "Record alignment readings before and after correction.",
}

CLASS_SEVERITY = {
    "Bearing Fault":        ("HIGH",    "#E53E3E"),
    "Bent Shaft":           ("HIGH",    "#E53E3E"),
    "Foundation Looseness": ("MEDIUM",  "#DD6B20"),
    "Healthy":              ("NONE",    "#38A169"),
    "Misalignment":         ("MEDIUM",  "#DD6B20"),
}

CLASS_ICONS = {
    "Bearing Fault":        "⊙",
    "Bent Shaft":           "↬",
    "Foundation Looseness": "⚠",
    "Healthy":              "✔",
    "Misalignment":         "⇹",
}

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Machine Fault Diagnosis | CNN",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #0D1117 !important;
    color: #E2E8F0 !important;
}

.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1400px !important;
}

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"] {
    background: #161B22 !important;
    border-right: 1px solid #21262D !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.25rem 1rem !important; }

[data-testid="stSidebar"] .stButton > button {
    background: #1C2333 !important;
    color: #94A3B8 !important;
    border: 1px solid #21262D !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
    text-align: left !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1E3A5F !important;
    color: #60A5FA !important;
    border-color: #2563EB !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #1E3A5F !important;
    color: #60A5FA !important;
    border-color: #2563EB !important;
    font-weight: 600 !important;
}

.sb-logo {
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid #21262D;
    margin-bottom: 1.1rem;
}
.sb-logo-icon {
    width: 36px; height: 36px; border-radius: 9px;
    background: linear-gradient(135deg, #1D4ED8, #3B82F6);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
    box-shadow: 0 0 14px rgba(59,130,246,0.35);
}
.sb-logo-text { font-size: 0.8rem; font-weight: 700; color: #F1F5F9; line-height: 1.2; }
.sb-logo-sub  { font-size: 0.68rem; color: #64748B; font-weight: 400; }

.sb-nav-label {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #4B5563; margin: 1rem 0 0.45rem;
}

.sb-fault-item {
    display: flex; align-items: center; gap: 10px;
    padding: 0.5rem 0.7rem; border-radius: 8px;
    background: #1C2333; margin-bottom: 5px;
    border: 1px solid #21262D;
    transition: border-color 0.15s;
}
.sb-fault-item:hover { border-color: #2563EB; }
.sb-fault-icon { font-size: 1.15rem; width: 26px; text-align: center; flex-shrink: 0; }
.sb-fault-name { font-size: 0.79rem; font-weight: 600; color: #E2E8F0; line-height: 1.2; }
.sb-fault-sev  { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.05em; }

.sb-stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 4px; }
.sb-stat {
    background: #1C2333; border: 1px solid #21262D;
    border-radius: 8px; padding: 0.55rem 0.5rem; text-align: center;
}
.sb-stat-val { font-size: 1rem; font-weight: 700; color: #60A5FA; }
.sb-stat-lbl { font-size: 0.62rem; color: #64748B; margin-top: 1px; }

.sb-footer {
    margin-top: 1.25rem; padding-top: 0.85rem;
    border-top: 1px solid #21262D;
    font-size: 0.68rem; color: #4B5563; text-align: center; line-height: 1.7;
}

/* ══ NAV ROW (top of each page) ══ */
div[data-testid="stButton"]:has(button[data-testid="topbar_about"]) button,
div[data-testid="stButton"]:has(button[data-testid="back_to_diag"]) button {
    background: #1E3A5F !important;
    color: #60A5FA !important;
    border: 1.5px solid #2563EB !important;
    border-radius: 8px !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    padding: 0.32rem 0.95rem !important;
    height: 2.1rem !important;
    white-space: nowrap !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 0 10px rgba(37,99,235,0.2) !important;
}
div[data-testid="stButton"]:has(button[data-testid="topbar_about"]) button:hover,
div[data-testid="stButton"]:has(button[data-testid="back_to_diag"]) button:hover {
    background: #1D4ED8 !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 16px rgba(37,99,235,0.45) !important;
}

/* ══ TOPBAR ══ */
.topbar {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 60%, #1D4ED8 100%);
    border: 1px solid #2563EB33;
    border-radius: 14px;
    padding: 1.75rem 1.75rem;
    margin-bottom: 1.25rem;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 0 30px rgba(37,99,235,0.15);
    gap: 1rem; position: relative; overflow: hidden;
    color: white;
}
.topbar::before {
    content: '⚙'; position: absolute; right: 1.75rem; top: 50%;
    transform: translateY(-50%); font-size: 7rem; opacity: 0.06;
    line-height: 1; pointer-events: none;
}
.topbar-left h1 {
    font-size: 1.45rem; font-weight: 800; color: #FFFFFF;
    margin: 0 0 0.2rem; letter-spacing: -0.4px;
}
.topbar-left p {
    font-size: 0.855rem; color: rgba(255,255,255,0.72);
    margin: 0; font-weight: 400; line-height: 1.5;
}
.topbar-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.topbar-badge {
    padding: 0.28rem 0.8rem; border-radius: 20px;
    font-size: 0.7rem; font-weight: 600;
    background: rgba(255,255,255,0.12); color: #FFFFFF;
    border: 1px solid rgba(255,255,255,0.22);
}
.topbar-badge.green {
    background: rgba(74,222,128,0.18); color: #4ADE80;
    border-color: rgba(74,222,128,0.35);
}

/* ══ STATUS BAR ══ */
.status-bar {
    background: #14532D22; border: 1px solid #16A34A33; border-radius: 9px;
    padding: 0.55rem 1.1rem; margin-bottom: 1.25rem;
    display: flex; align-items: center; gap: 0.85rem;
    font-size: 0.78rem; color: #4ADE80; font-weight: 500; flex-wrap: wrap;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22C55E; flex-shrink: 0;
    box-shadow: 0 0 6px #22C55E;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.status-divider { color: #16A34A66; }

/* ══ CARDS ══ */
.card {
    background: #161B22; border: 1px solid #21262D;
    border-radius: 12px; padding: 1.25rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3); margin-bottom: 1rem;
}
.card-header {
    display: flex; align-items: center; gap: 7px;
    font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #64748B;
    margin-bottom: 1rem; padding-bottom: 0.75rem;
    border-bottom: 1px solid #21262D;
}
.card-header-icon { font-size: 0.9rem; }
.card-header-lg {
    display: flex; align-items: center; gap: 7px;
    font-size: 0.92rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #94A3B8;
    margin-bottom: 1rem; padding-bottom: 0.75rem;
    border-bottom: 1px solid #21262D;
}

/* ══ CHIP ROW ══ */
.chip-row { display: flex; gap: 7px; flex-wrap: wrap; margin-bottom: 1rem; }
.chip {
    display: flex; align-items: center; gap: 5px;
    background: #1C2333; border: 1px solid #21262D;
    border-radius: 7px; padding: 0.3rem 0.75rem;
    font-size: 0.75rem; font-weight: 500; color: #94A3B8;
}
.chip-icon { font-size: 0.8rem; }

/* ══ RESULT CARD ══ */
.result-card {
    border-radius: 12px; padding: 1.5rem 1.25rem 1.25rem;
    text-align: center; border: 1.5px solid;
    margin-bottom: 0; position: relative; overflow: hidden;
    display: flex; flex-direction: column; align-items: center;
    height: 100%;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: currentColor;
}
.result-icon   { font-size: 2.5rem; margin-bottom: 0.5rem; display: block; line-height: 1; }
.result-eyebrow {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #64748B; margin-bottom: 0.25rem;
}
.result-class  { font-size: 1.35rem; font-weight: 700; margin-bottom: 0.6rem; line-height: 1.2; }
.result-conf   { font-size: 2.4rem; font-weight: 800; line-height: 1; }
.result-conf-sub { font-size: 0.7rem; color: #64748B; margin-top: 0.2rem; margin-bottom: 0.75rem; }
.sev-pill {
    display: inline-block; padding: 0.22rem 0.9rem; border-radius: 20px;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; border: 1.5px solid;
}

/* ══ INFO & ACTION BOX ══ */
.info-box {
    background: #1E3A5F22; border-left: 3px solid #3B82F6;
    border-radius: 0 8px 8px 0; padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem; font-size: 0.855rem;
    color: #CBD5E1; line-height: 1.75;
}
.action-box {
    background: #14532D22; border-left: 3px solid #22C55E;
    border-radius: 0 8px 8px 0; padding: 0.9rem 1.1rem;
    font-size: 0.855rem; color: #CBD5E1; line-height: 1.75;
}
.box-label {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 0.35rem; display: block;
}
.box-label.blue  { color: #3B82F6; }
.box-label.green { color: #22C55E; }

/* ══ FAULT GRID ══ */
.fault-grid-wrapper {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 10px; align-items: stretch;
}
.fault-grid-item {
    background: #161B22; border: 1px solid #21262D;
    border-radius: 11px; padding: 1.1rem 0.85rem; text-align: center;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 100%; transition: border-color 0.15s, transform 0.15s;
}
.fault-grid-item:hover { border-color: #2563EB; transform: translateY(-2px); }
.fault-grid-icon-box {
    width: 54px; height: 54px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.65rem; line-height: 1; font-weight: 700;
    margin: 0 auto 0.65rem;
    background: #1C2333; border: 1px solid #30374A;
}
.fault-grid-name { font-size: 0.79rem; font-weight: 600; color: #E2E8F0; margin-bottom: 0.35rem; }
.fault-grid-sev {
    display: inline-block; padding: 0.18rem 0.6rem;
    border-radius: 12px; font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
}

/* ══ STEP CARDS ══ */
.step-grid-wrapper {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 10px; align-items: stretch; margin-bottom: 1rem;
}
.step-card {
    background: #161B22; border: 1px solid #21262D;
    border-radius: 11px; padding: 1.4rem 1.1rem; text-align: center;
    display: flex; flex-direction: column; align-items: center; justify-content: flex-start;
    height: 100%;
}
.step-num {
    width: 38px; height: 38px; border-radius: 11px;
    background: linear-gradient(135deg, #1D4ED8, #3B82F6);
    color: white; font-size: 0.95rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 0.85rem;
    box-shadow: 0 0 12px rgba(59,130,246,0.35);
}
.step-title { font-size: 0.88rem; font-weight: 700; color: #F1F5F9; margin-bottom: 0.3rem; }
.step-desc  { font-size: 0.76rem; color: #64748B; line-height: 1.55; }

/* ══ ABOUT PAGE ══ */
.about-hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 60%, #1D4ED8 100%);
    border-radius: 14px; padding: 2.25rem 2rem;
    margin-bottom: 1.25rem; color: white;
    position: relative; overflow: hidden;
    border: 1px solid #2563EB33;
    box-shadow: 0 0 30px rgba(37,99,235,0.15);
}
.about-hero::before {
    content: '⚙'; position: absolute; right: 1.75rem; top: 50%;
    transform: translateY(-50%); font-size: 7rem; opacity: 0.06; line-height: 1;
}
.about-hero h2 { font-size: 1.5rem; font-weight: 800; margin: 0 0 0.35rem; letter-spacing: -0.3px; }
.about-hero p  { font-size: 0.875rem; opacity: 0.8; margin: 0; max-width: 580px; line-height: 1.65; }
.about-hero-badges { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 1rem; }
.about-hero-badge {
    background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px; padding: 0.22rem 0.8rem;
    font-size: 0.7rem; font-weight: 600; color: rgba(255,255,255,0.9);
}

/* ══ PROJECT INFO TABLE ══ */
.proj-info-table { width: 100%; border-collapse: collapse; font-size: 0.845rem; }
.proj-info-table tr { border-bottom: 1px solid #21262D; }
.proj-info-table tr:last-child { border-bottom: none; }
.proj-info-table td { padding: 0.65rem 0.5rem; vertical-align: top; line-height: 1.5; }
.proj-info-table td:first-child { width: 36%; padding-right: 0.75rem; }
.proj-info-table .td-label { display: flex; align-items: flex-start; gap: 7px; color: #64748B; font-weight: 500; }
.proj-info-table .td-icon  { font-size: 0.85rem; margin-top: 1px; flex-shrink: 0; }
.proj-info-table .td-key   { font-size: 0.8rem; font-weight: 600; color: #94A3B8; }
.proj-info-table .td-val   { color: #E2E8F0; font-weight: 500; }
.proj-info-table .td-badge {
    display: inline-block; background: #1E3A5F; color: #60A5FA;
    border: 1px solid #2563EB44; border-radius: 6px; padding: 0.15rem 0.6rem;
    font-size: 0.75rem; font-weight: 600; font-family: 'JetBrains Mono', monospace;
}

/* ══ OBJECTIVES TIMELINE ══ */
.obj-timeline { display: flex; flex-direction: column; gap: 0; }
.obj-step { display: flex; gap: 12px; align-items: flex-start; }
.obj-line { display: flex; flex-direction: column; align-items: center; flex-shrink: 0; }
.obj-dot {
    width: 28px; height: 28px; border-radius: 8px;
    background: linear-gradient(135deg, #1D4ED8, #3B82F6);
    color: #fff; font-size: 0.68rem; font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 1px;
    box-shadow: 0 0 10px rgba(37,99,235,0.4);
    font-family: 'JetBrains Mono', monospace;
}
.obj-connector {
    width: 2px; background: linear-gradient(to bottom, #2563EB44, #21262D);
    flex: 1; min-height: 22px; margin-top: 3px;
}
.obj-content { padding-bottom: 1rem; padding-top: 2px; }
.obj-content .o-title { font-size: 0.845rem; font-weight: 600; color: #E2E8F0; }
.obj-content .o-desc  { font-size: 0.775rem; color: #64748B; margin-top: 3px; line-height: 1.5; }

/* ══ METHODOLOGY TIMELINE ══ */
.method-timeline { display: flex; flex-direction: column; gap: 0; }
.method-step { display: flex; gap: 12px; align-items: flex-start; }
.method-line { display: flex; flex-direction: column; align-items: center; }
.method-dot {
    width: 11px; height: 11px; border-radius: 50%;
    background: #2563EB; flex-shrink: 0; margin-top: 4px;
    box-shadow: 0 0 8px rgba(37,99,235,0.5);
}
.method-connector {
    width: 2px; background: linear-gradient(to bottom, #2563EB44, #21262D);
    flex: 1; min-height: 28px; margin-top: 3px;
}
.method-content { padding-bottom: 1.1rem; }
.method-content .m-title { font-size: 0.845rem; font-weight: 600; color: #E2E8F0; }
.method-content .m-desc  { font-size: 0.775rem; color: #64748B; margin-top: 2px; line-height: 1.5; }

/* ══ TEAM ══ */
.guide-card {
    background: #1E3A5F22; border: 1.5px solid #2563EB44;
    border-radius: 11px; padding: 1rem 1.1rem;
    display: flex; align-items: center; gap: 12px; margin-bottom: 10px;
    box-shadow: 0 0 16px rgba(37,99,235,0.1);
}
.guide-avatar {
    width: 44px; height: 44px; border-radius: 11px;
    background: linear-gradient(135deg, #0F3D99, #1D4ED8);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; color: white; flex-shrink: 0;
    box-shadow: 0 0 12px rgba(29,78,216,0.4);
}
.guide-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #60A5FA; }
.guide-name  { font-size: 0.875rem; font-weight: 700; color: #F1F5F9; margin-top: 1px; }
.guide-dept  { font-size: 0.73rem; color: #64748B; margin-top: 1px; }
.team-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.team-card {
    background: #1C2333; border: 1px solid #21262D;
    border-radius: 10px; padding: 0.85rem 1rem;
    display: flex; align-items: center; gap: 10px; transition: border-color 0.15s;
}
.team-card:hover { border-color: #2563EB44; }
.team-avatar {
    width: 38px; height: 38px; border-radius: 9px;
    background: linear-gradient(135deg, #1D4ED8, #3B82F6);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem; color: white; flex-shrink: 0;
}
.team-name { font-size: 0.845rem; font-weight: 600; color: #E2E8F0; }
.team-id   { font-size: 0.72rem; color: #64748B; margin-top: 1px; }

/* ══ DATASET STATS (About page) ══ */
.dataset-stat-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 8px; margin-bottom: 0;
}
.dataset-stat-item {
    background: #1C2333; border: 1px solid #21262D;
    border-radius: 10px; padding: 0.85rem 0.9rem; text-align: center;
}
.dataset-stat-val {
    font-size: 1.45rem; font-weight: 800; color: #60A5FA;
    font-family: 'JetBrains Mono', monospace; line-height: 1;
}
.dataset-stat-lbl {
    font-size: 0.76rem; font-weight: 600; color: #E2E8F0; margin-top: 4px;
}
.dataset-stat-sub {
    font-size: 0.65rem; color: #4B5563; margin-top: 2px; line-height: 1.3;
}

/* ══ TECH GRID ══ */
.tech-group-label {
    font-size: 0.6rem; font-weight: 800; letter-spacing: 0.14em;
    text-transform: uppercase; color: #4B5563;
    margin: 0.85rem 0 0.4rem; display: flex; align-items: center; gap: 6px;
}
.tech-group-label::after {
    content: ''; flex: 1; height: 1px; background: #21262D;
}
.tech-item-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px;
}
.tech-item {
    background: #1C2333; border: 1px solid #21262D;
    border-radius: 9px; padding: 0.55rem 0.7rem;
    display: flex; align-items: center; gap: 8px;
    transition: border-color 0.15s, background 0.15s;
}
.tech-item:hover { border-color: #2563EB55; background: #1E2D45; }
.tech-icon {
    font-size: 1rem; flex-shrink: 0; width: 22px; text-align: center;
}
.tech-name { font-size: 0.75rem; font-weight: 500; color: #CBD5E1; line-height: 1.2; }
.tech-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: #1C2333; border: 1px solid #21262D;
    border-radius: 7px; padding: 0.38rem 0.75rem;
    font-size: 0.775rem; font-weight: 500; color: #94A3B8; margin: 3px;
    transition: border-color 0.15s;
}
.tech-pill:hover { border-color: #2563EB44; color: #60A5FA; }

/* ══ CNN FLOW (vertical) ══ */
.cnn-flow { display: flex; flex-direction: column; gap: 0; }
.cnn-arrow {
    text-align: center; color: #2563EB; font-size: 1rem;
    line-height: 1; margin: 2px 0; opacity: 0.7;
}
.cnn-row {
    display: flex; align-items: center; gap: 12px;
    padding: 0.7rem 1rem; border-radius: 10px;
    border: 1px solid #21262D; background: #1C2333;
    transition: border-color 0.15s;
}
.cnn-row:hover { border-color: #2563EB55; }
.cnn-tag {
    font-size: 0.58rem; font-weight: 800; letter-spacing: 0.1em;
    text-transform: uppercase; border-radius: 6px;
    padding: 0.2rem 0.55rem; flex-shrink: 0; width: 62px;
    text-align: center; font-family: 'JetBrains Mono', monospace;
}
.tag-input  { background: #1E3A5F33; color: #60A5FA; border: 1px solid #2563EB44; }
.tag-block  { background: #31233A44; color: #A78BFA; border: 1px solid #7C3AED44; }
.tag-head   { background: #0F3D2255; color: #34D399; border: 1px solid #059669AA; }
.tag-output { background: #3B1A1A44; color: #F87171; border: 1px solid #DC262644; }
.cnn-block-body { flex: 1; display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.cnn-title {
    font-size: 0.82rem; font-weight: 700; color: #E2E8F0;
    display: flex; align-items: center; gap: 7px;
}
.cnn-filters {
    font-size: 0.67rem; font-weight: 600; color: #A78BFA;
    background: #31233A55; border: 1px solid #7C3AED33;
    border-radius: 5px; padding: 0.1rem 0.45rem;
    font-family: 'JetBrains Mono', monospace;
}
.cnn-ops {
    font-size: 0.74rem; color: #64748B; line-height: 1.45;
    font-family: 'JetBrains Mono', monospace;
}
.cnn-detail { font-size: 0.78rem; color: #64748B; line-height: 1.5; }
.cnn-shape {
    font-size: 0.72rem; font-weight: 700; color: #60A5FA;
    font-family: 'JetBrains Mono', monospace;
    background: #1E3A5F33; border: 1px solid #2563EB33;
    border-radius: 7px; padding: 0.3rem 0.65rem;
    flex-shrink: 0; white-space: nowrap; text-align: center;
    min-width: 100px;
}
/* per-block accent colours */
.cnn-input  { border-color: #2563EB44; }
.cnn-block1 { border-color: #7C3AED33; }
.cnn-block1:hover { border-color: #7C3AED88 !important; }
.cnn-block2 { border-color: #7C3AED33; }
.cnn-block2:hover { border-color: #7C3AED88 !important; }
.cnn-block3 { border-color: #7C3AED33; }
.cnn-block3:hover { border-color: #7C3AED88 !important; }
.cnn-block4 { border-color: #7C3AED33; }
.cnn-block4:hover { border-color: #7C3AED88 !important; }
.cnn-head   { border-color: #05966944; }
.cnn-head:hover { border-color: #059669AA !important; }
.cnn-output { border-color: #DC262644; }
.cnn-output:hover { border-color: #DC2626AA !important; }

.problem-text { font-size: 0.855rem; color: #94A3B8; line-height: 1.8; }

/* ══ EXPANDER ══ */
[data-testid="stExpander"] {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #64748B !important; font-size: 0.82rem !important; }
pre, code {
    font-family: 'JetBrains Mono', monospace !important;
    background: #0D1117 !important; border-radius: 7px !important;
    font-size: 0.78rem !important; color: #94A3B8 !important;
    border: 1px solid #21262D !important;
}

/* ══ FILE UPLOADER ══ */
[data-testid="stFileUploader"] {
    background: #161B22 !important;
    border: 1.5px dashed #21262D !important;
    border-radius: 10px !important; padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: #2563EB !important; }
.stSpinner > div { border-top-color: #3B82F6 !important; }

/* ══ FOOTER ══ */
.footer {
    text-align: center; padding: 1.25rem; color: #374151; font-size: 0.73rem;
    border-top: 1px solid #21262D; margin-top: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ══════════════════════════════════════════════════════════════
#  PREPROCESS
# ══════════════════════════════════════════════════════════════
def preprocess(uploaded_file) -> np.ndarray:
    raw_bytes = uploaded_file.getvalue()
    img = tf.image.decode_png(raw_bytes, channels=1)
    img = tf.image.resize(img, [IMG_H, IMG_W])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    return img.numpy()


# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "diagnosis"


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class='sb-logo'>
        <div class='sb-logo-icon'>⚙️</div>
        <div>
            <div class='sb-logo-text'>Machine Fault Diagnosis</div>
            <div class='sb-logo-sub'>CNN · Deep Learning System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sb-nav-label'>Navigation</div>", unsafe_allow_html=True)

    if st.button("🔍  Fault Diagnosis", key="sb_diag",
                 use_container_width=True,
                 type="primary" if st.session_state.page == "diagnosis" else "secondary"):
        st.session_state.page = "diagnosis"
        st.rerun()

    if st.button("📘  Project Info", key="sb_about",
                 use_container_width=True,
                 type="primary" if st.session_state.page == "about" else "secondary"):
        st.session_state.page = "about"
        st.rerun()

    st.markdown("<div class='sb-nav-label'>Fault Reference</div>", unsafe_allow_html=True)
    for cls in CLASS_NAMES:
        severity, sev_color = CLASS_SEVERITY[cls]
        icon = CLASS_ICONS[cls]
        st.markdown(
            "<div class='sb-fault-item'>"
            f"<span class='sb-fault-icon'>{icon}</span>"
            "<div>"
            f"<div class='sb-fault-name'>{cls}</div>"
            f"<div class='sb-fault-sev' style='color:{sev_color}'>{severity} SEVERITY</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("""
    <div class='sb-footer'>
        Final Year Project · 2026–27<br>
        Machine Fault Diagnosis Using Deep Learning Approach<br>
        Dept. of Mechanical Engineering
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: ABOUT PROJECT
# ══════════════════════════════════════════════════════════════
if st.session_state.page == "about":

    # Nav row: spacer (left) | Back button (right)
    _spacer, _btn_col = st.columns([12, 2])
    with _btn_col:
        if st.button("← Back to Diagnosis", key="back_to_diag"):
            st.session_state.page = "diagnosis"
            st.rerun()

    st.markdown("""
    <div class='about-hero'>
        <h2>Machine Fault Diagnosis using Deep Learning</h2>
        <p>An AI-powered predictive maintenance system that classifies rotating machine
        faults from vibration signal images using a custom 4-block Convolutional Neural Network
        trained on multi-channel accelerometer data.</p>
        <div class='about-hero-badges'>
            <span class='about-hero-badge'>🎓 Final Year Project</span>
            <span class='about-hero-badge'>🤖 Deep Learning</span>
            <span class='about-hero-badge'>📡 Vibration Analysis</span>
            <span class='about-hero-badge'>🏭 Predictive Maintenance</span>
            <span class='about-hero-badge'>🔬 Signal Processing</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:

        # Project Info table
        st.markdown("""
        <div class='card'>
            <div class='card-header'><span class='card-header-icon'>🏫</span> Project Information</div>
            <table class='proj-info-table'>
                <tr>
                    <td><div class='td-label'><span class='td-icon'>📌</span><span class='td-key'>Project Title</span></div></td>
                    <td class='td-val'>Machine Fault Diagnosis using Deep Learning Approach</td>
                </tr>
                <tr>
                    <td><div class='td-label'><span class='td-icon'>🏛️</span><span class='td-key'>College</span></div></td>
                    <td class='td-val'>St. Vincent Pallotti College of Engineering and Technology, Nagpur</td>
                </tr>
                <tr>
                    <td><div class='td-label'><span class='td-icon'>⚙️</span><span class='td-key'>Department</span></div></td>
                    <td class='td-val'>Mechanical Engineering</td>
                </tr>
                <tr>
                    <td><div class='td-label'><span class='td-icon'>📅</span><span class='td-key'>Academic Year</span></div></td>
                    <td class='td-val'><span class='td-badge'>2026 – 2027</span>&nbsp; Final Year · Sem VIII</td>
                </tr>
                <tr>
                    <td><div class='td-label'><span class='td-icon'>🧠</span><span class='td-key'>Framework</span></div></td>
                    <td class='td-val'><span class='td-badge'>TensorFlow 2.x</span>&nbsp;<span class='td-badge'>Keras</span></td>
                </tr>
                <tr>
                    <td><div class='td-label'><span class='td-icon'>🚀</span><span class='td-key'>Deployment</span></div></td>
                    <td class='td-val'><span class='td-badge'>Streamlit</span>&nbsp; Web Application</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Problem Statement
        st.markdown("""
        <div class='card'>
            <div class='card-header'><span class='card-header-icon'>❗</span> Problem Statement</div>
            <div class='problem-text'>
                Faults in rotating machinery pose significant risks to operational reliability, safety,
                and maintenance efficiency. Conventional fault diagnosis techniques rely heavily on manual
                feature extraction and expert knowledge, limiting their accuracy under complex operating
                conditions. This project addresses the need for an automated and reliable machine fault 
                diagnosis system by employing 
                deep learning techniques for accurate detection and classification of machine faults.
                <br>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Objectives — timeline style matching methodology
        objectives = [
            ("🗂️", "Collect & Organise Data",
             "Build a balanced vibration signal image dataset across 5 fault classes and 3 sensor channels (CH1, CH2, CH3)."),
            ("🔧", "Preprocess Signals",
             "Convert raw time-domain vibration signals to 2D grayscale image representations; normalise and augment for robust training."),
            ("🧠", "Design CNN Architecture",
             "Develop a custom 4-block CNN (dual Conv→BN→ReLU per block) with GlobalAveragePooling2D and Dense head for efficient feature extraction."),
            ("⚙️", "Train & Optimise",
             "Train using Adam optimizer with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks."),
            ("📊", "Evaluate Rigorously",
             "Assess model using accuracy, precision, recall, F1-score, and confusion matrix on a held-out test set."),
            ("🚀", "Deploy as Web Application",
             "Build a real-time Streamlit diagnostic application for live vibration signal image classification."),
        ]
        obj_parts = [
            "<div class='card'>",
            "<div class='card-header'><span class='card-header-icon'>🎯</span> Project Objectives</div>",
            "<div class='obj-timeline'>",
        ]
        for idx, (icon, title, desc) in enumerate(objectives):
            connector = "" if idx == len(objectives) - 1 else "<div class='obj-connector'></div>"
            obj_parts.append(
                "<div class='obj-step'>"
                "<div class='obj-line'>"
                f"<div class='obj-dot'>{idx + 1}</div>"
                + connector +
                "</div>"
                "<div class='obj-content'>"
                f"<div class='o-title'>{icon}&nbsp; {title}</div>"
                f"<div class='o-desc'>{desc}</div>"
                "</div>"
                "</div>"
            )
        obj_parts.append("</div></div>")
        st.markdown("".join(obj_parts), unsafe_allow_html=True)

    with col2:

        # Team
        st.markdown("""
        <div class='card'>
            <div class='card-header'><span class='card-header-icon'>👥</span> Project Team</div>
            <div class='guide-card'>
                <div class='guide-avatar'>🎓</div>
                <div>
                    <div class='guide-label'>Project Guide</div>
                    <div class='guide-name'>Dr. Amit R Bhende</div>
                    <div class='guide-dept'>Department of Mechanical Engineering</div>
                </div>
            </div>
            <div style='font-size:0.62rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#4B5563;margin:0.85rem 0 0.5rem;'>
                Student Members
            </div>
            <div class='team-grid'>
                <div class='team-card'>
                    <div class='team-avatar'>👨‍💻</div>
                    <div><div class='team-name'>Vedant Giri</div><div class='team-id'>Member 1</div></div>
                </div>
                <div class='team-card'>
                    <div class='team-avatar'>👨‍💻</div>
                    <div><div class='team-name'>Tushar Kamble</div><div class='team-id'>Member 2</div></div>
                </div>
                <div class='team-card'>
                    <div class='team-avatar'>👨‍💻</div>
                    <div><div class='team-name'>Sanskar Patil</div><div class='team-id'>Member 3</div></div>
                </div>
                <div class='team-card'>
                    <div class='team-avatar'>👨‍💻</div>
                    <div><div class='team-name'>Ansheel Salodkar</div><div class='team-id'>Member 4</div></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Methodology timeline  ← NOW FIRST in col2
        method_steps = [
            ("Data Collection",
             "Vibration signals recorded via accelerometers at CH1, CH2, CH3 across 5 fault conditions."),
            ("Signal to Image",
             "Raw time-domain signals converted to 2D grayscale image representations (128 × 256 px)."),
            ("Preprocessing",
             "Images normalised to [0,1]; data augmentation applied on training set only."),
            ("CNN Training",
             "4-block CNN (Conv→BN→ReLU×2 → MaxPool → Dropout) + GAP + Dense(256) + Dropout(0.5); Adam optimizer with smart callbacks."),
            ("Evaluation",
             "Confusion matrix, per-class F1-score, precision, recall on 300-image held-out test set."),
            ("Deployment",
             "Interactive Streamlit web application for real-time image upload and diagnosis."),
        ]
        parts = [
            "<div class='card'>",
            "<div class='card-header'><span class='card-header-icon'>🔄</span> Methodology</div>",
            "<div class='method-timeline'>",
        ]
        for idx, (title, desc) in enumerate(method_steps):
            connector = "" if idx == len(method_steps) - 1 else "<div class='method-connector'></div>"
            parts.append(
                "<div class='method-step'>"
                "<div class='method-line'>"
                "<div class='method-dot'></div>"
                + connector +
                "</div>"
                "<div class='method-content'>"
                "<div class='m-title'>" + title + "</div>"
                "<div class='m-desc'>" + desc + "</div>"
                "</div>"
                "</div>"
            )
        parts.append("</div></div>")
        st.markdown("".join(parts), unsafe_allow_html=True)

        # Dataset Statistics  ← NOW SECOND in col2
        st.markdown("""
        <div class='card'>
            <div class='card-header'><span class='card-header-icon'>🗄️</span> Dataset Statistics</div>
            <div class='dataset-stat-grid'>
                <div class='dataset-stat-item'>
                    <div class='dataset-stat-val'>2,400</div>
                    <div class='dataset-stat-lbl'>Training Images</div>
                    <div class='dataset-stat-sub'>800 per channel × 3 channels</div>
                </div>
                <div class='dataset-stat-item'>
                    <div class='dataset-stat-val'>300</div>
                    <div class='dataset-stat-lbl'>Validation Images</div>
                    <div class='dataset-stat-sub'>Balanced across classes</div>
                </div>
                <div class='dataset-stat-item'>
                    <div class='dataset-stat-val'>300</div>
                    <div class='dataset-stat-lbl'>Test Images</div>
                    <div class='dataset-stat-sub'>Held-out evaluation set</div>
                </div>
                <div class='dataset-stat-item'>
                    <div class='dataset-stat-val'>5</div>
                    <div class='dataset-stat-lbl'>Fault Classes</div>
                    <div class='dataset-stat-sub'>Balanced distribution</div>
                </div>
            </div>
            <div style='margin-top:0.85rem;padding-top:0.75rem;border-top:1px solid #21262D;'>
                <div style='font-size:0.62rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#4B5563;margin-bottom:0.45rem;'>Input Specifications</div>
                <div style='display:flex;gap:8px;flex-wrap:wrap;'>
                    <span class='tech-pill'>📐 128 × 256 px</span>
                    <span class='tech-pill'>🎨 Grayscale (1-ch)</span>
                    <span class='tech-pill'>📡 CH1 · CH2 · CH3</span>
                    <span class='tech-pill'>🔢 Normalised [0, 1]</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Technologies — icon grid
        tech_data = {
            "🧠  Deep Learning": [
                ("🟠", "TensorFlow 2.x"),
                ("🔴", "Keras"),
                ("🔢", "NumPy"),
            ],
            "📊  Data & Visualisation": [
                ("📈", "Matplotlib"),
                ("🌊", "Seaborn"),
                ("🔬", "Scikit-learn"),
            ],
            "🚀  Deployment": [
                ("🌐", "Streamlit"),
                ("🐍", "Python 3.x"),
            ],
            "🛠️  Environment": [
                ("🐍", "Anaconda"),
                ("📓", "Jupyter Notebook"),
            ],
        }
        tech_html = "<div class='card'><div class='card-header'><span class='card-header-icon'>🛠️</span> Technologies Used</div>"
        for group, items in tech_data.items():
            tech_html += f"<div class='tech-group-label'>{group}</div>"
            tech_html += "<div class='tech-item-grid'>"
            for icon, name in items:
                tech_html += (
                    f"<div class='tech-item'>"
                    f"<span class='tech-icon'>{icon}</span>"
                    f"<span class='tech-name'>{name}</span>"
                    f"</div>"
                )
            tech_html += "</div>"
        tech_html += "</div>"
        st.markdown(tech_html, unsafe_allow_html=True)

    # CNN Architecture — full-width, vertical flow
    st.markdown(
        "<div class='card'>"
        "<div class='card-header'><span class='card-header-icon'>🧠</span> CNN Architecture</div>"
        "<div class='cnn-flow'>"

        "<div class='cnn-row cnn-input'>"
        "<div class='cnn-tag tag-input'>INPUT</div>"
        "<div class='cnn-block-body'>"
        "<span class='cnn-title'>Input Layer</span>"
        "<span class='cnn-detail'>Grayscale spectrogram &nbsp;·&nbsp; shape <code>(128, 256, 1)</code> &nbsp;·&nbsp; normalised to [0, 1]</span>"
        "</div>"
        "<div class='cnn-shape'>128 &times; 256 &times; 1</div>"
        "</div>"
        "<div class='cnn-arrow'>&#9660;</div>"

        "<div class='cnn-row cnn-block1'>"
        "<div class='cnn-tag tag-block'>BLOCK 1</div>"
        "<div class='cnn-block-body'>"
        "<span class='cnn-title'>Conv Block 1 &nbsp;<span class='cnn-filters'>32 filters</span></span>"
        "<span class='cnn-ops'>Conv2D(32, 3&times;3) &rarr; BN &rarr; ReLU &rarr; Conv2D(32, 3&times;3) &rarr; BN &rarr; ReLU</span>"
        "<span class='cnn-ops'>MaxPool(2&times;2) &rarr; Dropout(0.1)</span>"
        "</div>"
        "<div class='cnn-shape'>64 &times; 128 &times; 32</div>"
        "</div>"
        "<div class='cnn-arrow'>&#9660;</div>"

        "<div class='cnn-row cnn-block2'>"
        "<div class='cnn-tag tag-block'>BLOCK 2</div>"
        "<div class='cnn-block-body'>"
        "<span class='cnn-title'>Conv Block 2 &nbsp;<span class='cnn-filters'>64 filters</span></span>"
        "<span class='cnn-ops'>Conv2D(64, 3&times;3) &rarr; BN &rarr; ReLU &rarr; Conv2D(64, 3&times;3) &rarr; BN &rarr; ReLU</span>"
        "<span class='cnn-ops'>MaxPool(2&times;2) &rarr; Dropout(0.2)</span>"
        "</div>"
        "<div class='cnn-shape'>32 &times; 64 &times; 64</div>"
        "</div>"
        "<div class='cnn-arrow'>&#9660;</div>"

        "<div class='cnn-row cnn-block3'>"
        "<div class='cnn-tag tag-block'>BLOCK 3</div>"
        "<div class='cnn-block-body'>"
        "<span class='cnn-title'>Conv Block 3 &nbsp;<span class='cnn-filters'>128 filters</span></span>"
        "<span class='cnn-ops'>Conv2D(128, 3&times;3) &rarr; BN &rarr; ReLU &rarr; Conv2D(128, 3&times;3) &rarr; BN &rarr; ReLU</span>"
        "<span class='cnn-ops'>MaxPool(2&times;2) &rarr; Dropout(0.3)</span>"
        "</div>"
        "<div class='cnn-shape'>16 &times; 32 &times; 128</div>"
        "</div>"
        "<div class='cnn-arrow'>&#9660;</div>"

        "<div class='cnn-row cnn-block4'>"
        "<div class='cnn-tag tag-block'>BLOCK 4</div>"
        "<div class='cnn-block-body'>"
        "<span class='cnn-title'>Conv Block 4 &nbsp;<span class='cnn-filters'>256 filters</span></span>"
        "<span class='cnn-ops'>Conv2D(256, 3&times;3) &rarr; BN &rarr; ReLU &rarr; Conv2D(256, 3&times;3) &rarr; BN &rarr; ReLU</span>"
        "<span class='cnn-ops'>MaxPool(2&times;2) &rarr; Dropout(0.3)</span>"
        "</div>"
        "<div class='cnn-shape'>8 &times; 16 &times; 256</div>"
        "</div>"
        "<div class='cnn-arrow'>&#9660;</div>"

        "<div class='cnn-row cnn-head'>"
        "<div class='cnn-tag tag-head'>HEAD</div>"
        "<div class='cnn-block-body'>"
        "<span class='cnn-title'>Classification Head</span>"
        "<span class='cnn-ops'>GlobalAveragePooling2D &nbsp;&rarr;&nbsp; Dense(256) &rarr; BN &rarr; ReLU &rarr; Dropout(0.5)</span>"
        "</div>"
        "<div class='cnn-shape'>256-d vector</div>"
        "</div>"
        "<div class='cnn-arrow'>&#9660;</div>"

        "<div class='cnn-row cnn-output'>"
        "<div class='cnn-tag tag-output'>OUTPUT</div>"
        "<div class='cnn-block-body'>"
        "<span class='cnn-title'>Output Layer</span>"
        "<span class='cnn-detail'>Dense(5) &rarr; Softmax &nbsp;·&nbsp; probability distribution over 5 fault classes</span>"
        "</div>"
        "<div class='cnn-shape'>5 classes</div>"
        "</div>"

        "</div>"
        "</div>",
        unsafe_allow_html=True
    )

    # Project Description
    st.markdown("""
    <div class='card'>
        <div class='card-header'><span class='card-header-icon'>📄</span> Project Description</div>
        <div class='problem-text'>
            This project presents an end-to-end deep learning pipeline for automated
            machine fault detection and classification. Vibration signals from rotating
            machinery are acquired using accelerometers placed at three positions on the
            machine (Channel 1, CH2, and CH3), and the raw time-domain signals are converted
            into 2D grayscale image representations of size <strong style='color:#E2E8F0'>128 × 256 pixels</strong>,
            making them suitable for CNN-based visual feature extraction.
            <br><br>
            The custom CNN model consists of <strong style='color:#E2E8F0'>four progressively deeper
            convolutional blocks</strong>. Each block applies two consecutive Conv2D layers with Batch
            Normalisation and ReLU activation, followed by MaxPooling(2×2) for spatial downsampling
            and Dropout for regularisation. Filters grow from 32 → 64 → 128 → 256, enabling the
            network to capture increasingly abstract fault-related features. After the convolutional
            backbone, <strong style='color:#E2E8F0'>Global Average Pooling</strong> replaces Flatten to
            reduce parameters and control overfitting, feeding into a Dense(256) → BN → ReLU →
            Dropout(0.5) head before the 5-class softmax output.
            <br><br>
            The training strategy employs the Adam optimiser with a starting learning rate
            of 0.001, complemented by three callbacks: <strong style='color:#E2E8F0'>EarlyStopping</strong>
            (patience=10) to prevent overfitting, <strong style='color:#E2E8F0'>ReduceLROnPlateau</strong>
            (factor=0.5, patience=5) to escape training plateaus, and
            <strong style='color:#E2E8F0'>ModelCheckpoint</strong> to automatically preserve the
            best-performing weights based on validation accuracy.
            <br><br>
            Each sensor channel is treated as an independent sample, tripling the effective
            training dataset to <strong style='color:#E2E8F0'>2,400 training images</strong> across
            5 fault classes. The model is evaluated on a completely held-out test set of
            <strong style='color:#E2E8F0'>300 images</strong> using accuracy, per-class F1-score,
            precision, recall, and confusion matrix analysis to ensure robust, generalisable
            performance across all five fault categories.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='footer'>
        Machine Fault Diagnosis System &nbsp;·&nbsp; St. Vincent Pallotti College of Engineering &nbsp;·&nbsp;
        Mechanical Engineering &nbsp;·&nbsp; Final Year Project 2026–27 &nbsp;·&nbsp;
        TensorFlow &amp; Streamlit
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: FAULT DIAGNOSIS
# ══════════════════════════════════════════════════════════════
else:

    # Nav row: spacer (left) | Project Info button (right)
    _spacer, _btn_col = st.columns([15, 2])
    with _btn_col:
        if st.button("📘  Project Info", key="topbar_about"):
            st.session_state.page = "about"
            st.rerun()

    st.markdown("""
    <div class='topbar'>
        <div class='topbar-left'>
            <h1>Machine Fault Diagnosis System</h1>
            <p>CNN-based vibration signal analysis for predictive maintenance and condition monitoring.</p>
        </div>
        <div class='topbar-right'>
            <span class='topbar-badge green'>🟢 System Ready</span>
            <span class='topbar-badge'>⚙️ CNN Model</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Initialising model..."):
        model = load_model()

    st.markdown(
        "<div class='status-bar'>"
        "<div class='status-dot'></div>"
        "<span>Model loaded successfully</span>"
        "<span class='status-divider'>|</span>"
        f"<span>Input: {model.input_shape}</span>"
        "<span class='status-divider'>|</span>"
        f"<span>Parameters: {model.count_params():,}</span>"
        "<span class='status-divider'>|</span>"
        f"<span>Classes: {len(CLASS_NAMES)}</span>"
        "<span class='status-divider'>|</span>"
        "<span>✅ Ready for inference</span>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class='card'>
        <div class='card-header-lg'><span style='font-size:1rem'>📤</span>&nbsp; Upload Vibration Signal Image</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a vibration signal graph image (PNG / JPG) from CH1, CH2, or CH3",
        type=["png", "jpg", "jpeg"],
        label_visibility="visible"
    )

    # ── EMPTY STATE ───────────────────────────────────────────
    if uploaded is None:
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class='step-grid-wrapper'>
            <div class='step-card'>
                <div class='step-num'>1</div>
                <div class='step-title'>Upload Image</div>
                <div class='step-desc'>Select a PNG/JPG vibration signal graph from sensor channels CH1, CH2, or CH3.</div>
            </div>
            <div class='step-card'>
                <div class='step-num'>2</div>
                <div class='step-title'>CNN Analysis</div>
                <div class='step-desc'>The deep learning model automatically extracts fault features from the signal image.</div>
            </div>
            <div class='step-card'>
                <div class='step-num'>3</div>
                <div class='step-title'>Get Diagnosis</div>
                <div class='step-desc'>View fault class, confidence score, engineering explanation, and recommended action.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class='card'>
            <div class='card-header-lg'><span style='font-size:1rem'>📋</span>&nbsp; Detectable Fault Classes</div>
        """, unsafe_allow_html=True)

        fault_html = "<div class='fault-grid-wrapper'>"
        for cls in CLASS_NAMES:
            severity, sev_color = CLASS_SEVERITY[cls]
            icon = CLASS_ICONS[cls]
            fault_html += (
                "<div class='fault-grid-item'>"
                f"<div class='fault-grid-icon-box'>{icon}</div>"
                f"<div class='fault-grid-name'>{cls}</div>"
                f"<span class='fault-grid-sev' style='background:{sev_color}18;color:{sev_color};border:1px solid {sev_color}44'>"
                f"{severity}"
                "</span>"
                "</div>"
            )
        fault_html += "</div>"
        st.markdown(fault_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── PREDICTION STATE ──────────────────────────────────────
    else:
        with st.spinner("🔍 Analysing vibration signal..."):
            t0      = time.time()
            arr     = preprocess(uploaded)
            preds   = model.predict(arr, verbose=0)[0]
            elapsed = time.time() - t0

        pred_idx            = int(np.argmax(preds))
        pred_class          = CLASS_NAMES[pred_idx]
        confidence          = float(preds[pred_idx]) * 100
        severity, sev_color = CLASS_SEVERITY[pred_class]
        icon                = CLASS_ICONS[pred_class]

        st.markdown(
            "<div class='chip-row'>"
            f"<span class='chip'><span class='chip-icon'>📁</span>{uploaded.name}</span>"
            f"<span class='chip'><span class='chip-icon'>⏱️</span>{elapsed*1000:.0f} ms inference</span>"
            f"<span class='chip'><span class='chip-icon'>📐</span>{IMG_W} × {IMG_H} px input</span>"
            "<span class='chip'><span class='chip-icon'>🧠</span>CNN · Softmax output</span>"
            "</div>",
            unsafe_allow_html=True
        )

        left, right = st.columns([1.1, 1], gap="large")

        with left:
            st.markdown("""
            <div class='card'>
                <div class='card-header'><span class='card-header-icon'>🖼️</span> Uploaded Vibration Signal</div>
            """, unsafe_allow_html=True)
            st.image(
                uploaded.getvalue(),
                caption=f"{uploaded.name}  |  Resized to {IMG_W}×{IMG_H} for inference",
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown(
                "<div class='card' style='height:100%;display:flex;flex-direction:column;'>"
                "<div class='card-header'><span class='card-header-icon'>🔍</span> Diagnosis Result</div>"
                f"<div class='result-card' style='border-color:{sev_color};background:{sev_color}0D;color:{sev_color};flex:1;'>"
                f"<span class='result-icon'>{icon}</span>"
                "<div class='result-eyebrow'>Detected Fault Condition</div>"
                f"<div class='result-class' style='color:{sev_color}'>{pred_class}</div>"
                f"<div class='result-conf' style='color:{sev_color}'>{confidence:.1f}%</div>"
                "<div class='result-conf-sub'>Model Confidence Score</div>"
                f"<div class='sev-pill' style='background:{sev_color}18;color:{sev_color};border-color:{sev_color}55'>"
                f"{severity} SEVERITY"
                "</div>"
                "</div>"
                "</div>",
                unsafe_allow_html=True
            )

        st.markdown(
            "<div class='card'>"
            "<div class='card-header'><span class='card-header-icon'>📖</span> Fault Explanation &amp; Recommended Action</div>"
            f"<span class='box-label blue'>Diagnosis — {icon} {pred_class}</span>"
            f"<div class='info-box'>{CLASS_INFO[pred_class]}</div>"
            "<span class='box-label green'>Recommended Action</span>"
            f"<div class='action-box'>{CLASS_ACTION[pred_class]}</div>"
            "</div>",
            unsafe_allow_html=True
        )

        with st.expander("🔬 Technical Details — Raw Prediction Data"):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Preprocessed Tensor Info**")
                st.code(
                    f"Shape      : {arr.shape}\n"
                    f"Dtype      : {arr.dtype}\n"
                    f"Pixel min  : {arr.min():.4f}\n"
                    f"Pixel max  : {arr.max():.4f}\n"
                    f"Pixel mean : {arr.mean():.4f}\n"
                    f"Inference  : {elapsed*1000:.1f} ms"
                )
            with d2:
                st.markdown("**Raw Softmax Probabilities**")
                for cls, p in zip(CLASS_NAMES, preds):
                    bar = "█" * int(p * 28)
                    st.code(f"{cls:<22}: {p*100:>6.3f}%  {bar}")

    st.markdown("""
    <div class='footer'>
        Machine Fault Diagnosis Using Deep Learning &nbsp;·&nbsp;
        Final Year Project 2026–27 &nbsp;·&nbsp; Built with TensorFlow &amp; Streamlit
    </div>
    """, unsafe_allow_html=True)