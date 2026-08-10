import streamlit as st
import tensorflow as tf
import numpy as np
import time
import os

# ==========================================================
#  PROJECT: Machine Fault Detection using CNN
# ==========================================================

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
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
    overflow-x: hidden !important;
}

.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1400px !important;
    overflow-x: hidden !important;
}

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"] {
    background: #161B22 !important;
    border-right: 1px solid #21262D !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.25rem 1rem !important; }

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

.sb-footer {
    margin-top: 1.25rem; padding-top: 0.85rem;
    border-top: 1px solid #21262D;
    font-size: 0.68rem; color: #4B5563; text-align: center; line-height: 1.7;
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
    z-index: 0;
}
.topbar-left, .topbar-right {
    position: relative;
    z-index: 1;
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
    white-space: nowrap;
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
    min-width: 0;
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
    height: 100%; min-width: 0;
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

/* ══════════════════════════════════════════════════════════
   MOBILE RESPONSIVE FIXES
   - topbar: stack title above badges instead of clipping
   - fault grid: 5 cols -> 2 cols so cards don't get squeezed off-screen
   - step grid: 3 cols -> 1 col so cards stack cleanly
   ══════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem 0.85rem 2rem !important;
    }

    .topbar {
        flex-direction: column;
        align-items: flex-start;
        padding: 1.35rem 1.1rem;
    }
    .topbar::before {
        font-size: 4.5rem;
        right: 0.75rem;
    }
    .topbar-left h1 { font-size: 1.15rem; }
    .topbar-left p  { font-size: 0.8rem; }
    .topbar-right { width: 100%; }

    .status-bar {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.3rem;
    }
    .status-divider { display: none; }

    .fault-grid-wrapper {
        grid-template-columns: repeat(2, 1fr);
    }

    .step-grid-wrapper {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 420px) {
    .fault-grid-wrapper {
        grid-template-columns: 1fr;
    }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  VERSION-SAFE IMAGE HELPER
#  Newer Streamlit versions use `use_container_width`; older
#  versions only support `use_column_width`. This wrapper tries
#  the modern kwarg first and falls back automatically so the
#  app doesn't crash regardless of the Streamlit version that
#  ends up installed on the deployment host.
# ══════════════════════════════════════════════════════════════
def show_image(data, caption=None):
    try:
        st.image(data, caption=caption, use_container_width=True)
    except TypeError:
        st.image(data, caption=caption, use_column_width=True)


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
        Machine Fault Diagnosis System<br>
        CNN-based Vibration Signal Analysis
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MAIN PAGE: FAULT DIAGNOSIS
# ══════════════════════════════════════════════════════════════
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
        show_image(
            uploaded.getvalue(),
            caption=f"{uploaded.name}  |  Resized to {IMG_W}×{IMG_H} for inference"
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
    Machine Fault Diagnosis System &nbsp;·&nbsp; Built with TensorFlow &amp; Streamlit
</div>
""", unsafe_allow_html=True)
