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
.g
