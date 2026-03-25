
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ── Load models ───────────────────────────────
model   = pickle.load(open("model_1.pkl",   "rb"))
scaler  = pickle.load(open("scaler_1.pkl",  "rb"))
encoder = pickle.load(open("ohe_1.pkl",     "rb"))
columns = pickle.load(open("columns_1.pkl", "rb"))

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LoanLens · Credit Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────
#  THEME CSS  — Light · Professional · Blue
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

/* ── TOKENS ── */
:root {
    --blue-900: #0C2340;
    --blue-800: #12325C;
    --blue-700: #1A4A82;
    --blue-600: #1E5FA8;
    --blue-500: #2272CE;
    --blue-400: #4B8FDE;
    --blue-300: #7BB3ED;
    --blue-200: #B8D5F5;
    --blue-100: #DDEEFF;
    --blue-50:  #EEF6FF;
    --blue-25:  #F5FAFF;

    --slate-900: #0F172A;
    --slate-700: #334155;
    --slate-500: #64748B;
    --slate-400: #94A3B8;
    --slate-200: #E2E8F0;
    --slate-100: #F1F5F9;
    --slate-50:  #F8FAFC;
    --white:     #FFFFFF;

    --success:   #0F7B55;
    --success-bg:#ECFDF5;
    --success-b: #A7F3D0;
    --danger:    #B91C1C;
    --danger-bg: #FEF2F2;
    --danger-b:  #FECACA;
    --warn-bg:   #FFFBEB;
    --warn-b:    #FDE68A;
    --warn-text: #92400E;

    --radius-sm: 6px;
    --radius:    10px;
    --radius-lg: 14px;
    --radius-xl: 18px;

    --shadow-sm: 0 1px 3px rgba(15,35,64,0.06), 0 1px 2px rgba(15,35,64,0.04);
    --shadow:    0 4px 12px rgba(15,35,64,0.08), 0 2px 4px rgba(15,35,64,0.05);
    --shadow-lg: 0 10px 32px rgba(15,35,64,0.12), 0 4px 8px rgba(15,35,64,0.06);
}

/* ── RESET ── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stMain"] {
    background: var(--blue-25) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--slate-900) !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebarNav"],
footer { display: none !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--slate-100); }
::-webkit-scrollbar-thumb { background: var(--blue-300); border-radius: 4px; }

/* ── LAYOUT ── */
.block-container {
    padding: 0 2rem 5rem !important;
    max-width: 1240px !important;
    margin: 0 auto !important;
}

[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
}

[data-testid="stHorizontalBlock"] { gap: 1rem !important; }

/* ── TOP NAV BAR ── */
.topbar {
    background: var(--white);
    border-bottom: 1px solid var(--slate-200);
    padding: 0 2.5rem;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: var(--shadow-sm);
    margin-bottom: 0;
}
.topbar-logo {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 18px;
    font-weight: 800;
    color: var(--blue-700);
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 8px;
}
.topbar-logo span { color: var(--blue-500); }
.topbar-right {
    display: flex;
    align-items: center;
    gap: 20px;
    font-size: 12px;
    color: var(--slate-500);
    font-weight: 500;
}
.topbar-badge {
    background: var(--blue-50);
    border: 1px solid var(--blue-200);
    color: var(--blue-600);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, var(--blue-800) 0%, var(--blue-600) 55%, var(--blue-400) 100%);
    border-radius: 0 0 var(--radius-xl) var(--radius-xl);
    padding: 3rem 3rem 3.5rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 2rem;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; right: 120px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.03);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.85);
    margin-bottom: 1.25rem;
}
.hero-dot {
    width: 5px; height: 5px;
    background: #4ADE80;
    border-radius: 50%;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink {
    0%,100% { opacity:1; }
    50%      { opacity:0.35; }
}
.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 38px;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.025em;
    line-height: 1.1;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 15px;
    font-weight: 300;
    color: rgba(255,255,255,0.7);
    line-height: 1.7;
    max-width: 520px;
    margin-bottom: 2rem;
}
.hero-pills {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.hero-pill {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 500;
    color: rgba(255,255,255,0.85);
}
.hero-stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: rgba(255,255,255,0.12);
    border-radius: var(--radius-lg);
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.15);
}
.hero-stat {
    background: rgba(255,255,255,0.08);
    padding: 1.1rem 1.25rem;
    text-align: center;
}
.hero-stat-val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.02em;
}
.hero-stat-label {
    font-size: 11px;
    color: rgba(255,255,255,0.6);
    font-weight: 500;
    letter-spacing: 0.04em;
    margin-top: 2px;
}

/* ── PROGRESS STEPS ── */
.steps {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 1.75rem;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
}
.step-num {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: var(--blue-500);
    color: #fff;
    font-size: 12px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.step-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--blue-700);
    letter-spacing: 0.02em;
}
.step-connector {
    flex: 1;
    height: 1px;
    background: var(--slate-200);
    margin: 0 12px;
}

/* ── SECTION CARD ── */
.sec {
    background: var(--white);
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-xl);
    padding: 1.5rem 1.75rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-sm);
    position: relative;
}
.sec::before {
    content: '';
    position: absolute;
    top: 0; left: 1.75rem; right: 1.75rem;
    height: 2px;
    background: linear-gradient(90deg, var(--blue-500), var(--blue-300));
    border-radius: 0 0 var(--radius) var(--radius);
    opacity: 0;
}
.sec-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--slate-100);
}
.sec-icon {
    width: 34px; height: 34px;
    background: var(--blue-50);
    border: 1px solid var(--blue-200);
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    flex-shrink: 0;
}
.sec-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--blue-700);
    flex: 1;
}
.sec-count {
    font-size: 11px;
    font-weight: 600;
    color: var(--slate-400);
    background: var(--slate-100);
    border-radius: 999px;
    padding: 2px 10px;
}

/* ── INPUTS ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: var(--white) !important;
    border: 1.5px solid var(--slate-200) !important;
    border-radius: var(--radius) !important;
    color: var(--slate-900) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 9px 13px !important;
    height: 44px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    caret-color: var(--blue-500) !important;
}
[data-testid="stNumberInput"] input::placeholder,
[data-testid="stTextInput"] input::placeholder {
    color: var(--slate-400) !important;
    font-weight: 400 !important;
    font-style: italic !important;
}
[data-testid="stNumberInput"] input:hover,
[data-testid="stTextInput"] input:hover {
    border-color: var(--blue-300) !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--blue-500) !important;
    box-shadow: 0 0 0 3px rgba(34,114,206,0.10) !important;
    outline: none !important;
}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--white) !important;
    border: 1.5px solid var(--slate-200) !important;
    border-radius: var(--radius) !important;
    color: var(--slate-900) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    min-height: 44px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--blue-300) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--blue-500) !important;
    box-shadow: 0 0 0 3px rgba(34,114,206,0.10) !important;
}
[data-testid="stSelectbox"] svg { fill: var(--blue-500) !important; }
[data-testid="stSelectboxVirtualDropdown"],
ul[data-testid="stSelectboxVirtualDropdown"] {
    background: var(--white) !important;
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-lg) !important;
}
[data-testid="stSelectbox"] li {
    color: var(--slate-700) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
[data-testid="stSelectbox"] li:hover {
    background: var(--blue-50) !important;
    color: var(--blue-700) !important;
}
/* 🔥 FIX: Dropdown options text visible */
ul[data-testid="stSelectboxVirtualDropdown"] li {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* Hover state */
ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
    background-color: #EEF6FF !important;
    color: #000000 !important;
}

/* Selected value text */
[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #000000 !important;
}

/* ── LABELS ── */
label,
[data-testid="stWidgetLabel"] p,
[data-baseweb="form-control-label"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    color: var(--slate-700) !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
}

/* ── STEPPER BUTTONS ── */
[data-testid="stNumberInput"] button {
    background: var(--slate-50) !important;
    border-color: var(--slate-200) !important;
    color: var(--blue-600) !important;
}
[data-testid="stNumberInput"] button:hover {
    background: var(--blue-50) !important;
    border-color: var(--blue-300) !important;
    color: var(--blue-700) !important;
}

/* ── SUBMIT BUTTON ── */
[data-testid="stFormSubmitButton"] > button {
    background: var(--blue-600) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-lg) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    height: 52px !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(30,95,168,0.30), 0 1px 0 rgba(255,255,255,0.1) inset !important;
    transition: background 0.18s, box-shadow 0.18s, transform 0.12s !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--blue-700) !important;
    box-shadow: 0 6px 24px rgba(30,95,168,0.40), 0 1px 0 rgba(255,255,255,0.12) inset !important;
    transform: translateY(-1px) !important;
}
[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(0) !important;
    background: var(--blue-800) !important;
}

/* ── PROGRESS BAR ── */
[data-testid="stProgress"] > div > div {
    background: var(--slate-100) !important;
    border-radius: 999px !important;
    height: 8px !important;
}
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--blue-500), var(--blue-300)) !important;
    border-radius: 999px !important;
}

/* ── RESULT CARD ── */
.result-wrap {
    background: var(--white);
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-xl);
    overflow: hidden;
    box-shadow: var(--shadow-lg);
}
.result-header {
    padding: 2.25rem 2rem 1.75rem;
    text-align: center;
    border-bottom: 1px solid var(--slate-100);
}
.result-header.approved { background: linear-gradient(160deg, var(--success-bg) 0%, #fff 60%); }
.result-header.rejected { background: linear-gradient(160deg, var(--danger-bg) 0%, #fff 60%); }

.result-verdict {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
}
.result-sub {
    font-size: 13.5px;
    font-weight: 400;
    color: var(--slate-500);
    max-width: 380px;
    margin: 0 auto 1.5rem;
    line-height: 1.65;
}
.result-score {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 64px;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
}
.result-score-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--slate-400);
    margin-top: 6px;
    margin-bottom: 1.5rem;
}
.result-body {
    padding: 1.5rem 2rem 2rem;
}

/* ── METRIC GRID ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-top: 1.25rem;
}
.metric-card {
    background: var(--slate-50);
    border: 1px solid var(--slate-200);
    border-radius: var(--radius-lg);
    padding: 1rem;
    text-align: center;
}
.metric-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--slate-400);
    margin-bottom: 5px;
}
.metric-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--slate-900);
}

/* ── STATUS BADGE ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 5px 16px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.badge-approved { background: var(--success-bg); border: 1px solid var(--success-b); color: var(--success); }
.badge-rejected { background: var(--danger-bg); border: 1px solid var(--danger-b); color: var(--danger); }

/* ── CONFIDENCE BAR LABEL ── */
.conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.conf-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--slate-500);
}
.conf-value {
    font-size: 12px;
    font-weight: 700;
}

/* ── VALIDATION ── */
.val-box {
    background: var(--danger-bg);
    border: 1px solid var(--danger-b);
    border-radius: var(--radius-lg);
    padding: 1rem 1.25rem;
    color: var(--danger);
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 1.25rem;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.val-box ul {
    margin: 6px 0 0;
    padding-left: 1.1rem;
    line-height: 1.9;
}

/* ── DISCLAIMER ── */
.disclaimer {
    background: var(--warn-bg);
    border: 1px solid var(--warn-b);
    border-radius: var(--radius);
    padding: .7rem 1rem;
    font-size: 12px;
    color: var(--warn-text);
    font-weight: 500;
    line-height: 1.55;
    margin-bottom: 1.5rem;
}

/* ── HINT ── */
.hint {
    font-size: 11px;
    color: var(--slate-400);
    font-style: italic;
    margin-top: 3px;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── DIVIDER ── */
.div { height: 1px; background: var(--slate-100); margin: 1rem 0; }

/* ── FOOTER ── */
.foot {
    text-align: center;
    padding: 2rem 0 3rem;
    border-top: 1px solid var(--slate-100);
    margin-top: 3rem;
}
.foot-logo { font-size: 14px; font-weight: 800; color: var(--blue-700); letter-spacing: -0.01em; }
.foot-logo span { color: var(--blue-500); }
.foot-copy { font-size: 11.5px; color: var(--slate-400); margin-top: 4px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  NAV BAR
# ─────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">🏦 &nbsp;Loan<span>Lens</span></div>
  <div class="topbar-right">
    <span>Credit Assessment Portal</span>
    <div class="topbar-badge">● &nbsp;System Online</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────
hero_l, hero_r = st.columns([3, 2])
with hero_l:
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow"><span class="hero-dot"></span>AI-Powered Credit Intelligence</div>
      <div class="hero-title">Instant Loan<br>Eligibility Check</div>
      <div class="hero-sub">
        Fill in your financial details below and receive a real-time
        credit assessment powered by our machine learning model.
        Results in under 3 seconds.
      </div>
      <div class="hero-pills">
        <div class="hero-pill">⚡ Real-time decisioning</div>
        <div class="hero-pill">🔒 Secure & confidential</div>
        <div class="hero-pill">📊 ML-powered scoring</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with hero_r:
    st.markdown("""
    <div style="height:1rem"></div>
    <div class="hero-stat-grid" style="background:var(--white);border:1px solid var(--slate-200);
         border-radius:var(--radius-xl);overflow:hidden;box-shadow:var(--shadow);">
      <div class="hero-stat" style="background:var(--blue-50);padding:1.5rem 1rem;text-align:center;border-right:1px solid var(--slate-200);">
        <div style="font-size:28px;font-weight:800;color:var(--blue-700);font-family:'Plus Jakarta Sans',sans-serif;">95%</div>
        <div style="font-size:11px;color:var(--slate-500);font-weight:600;letter-spacing:0.04em;margin-top:3px;">Model Accuracy</div>
      </div>
      <div class="hero-stat" style="background:var(--blue-50);padding:1.5rem 1rem;text-align:center;border-right:1px solid var(--slate-200);">
        <div style="font-size:28px;font-weight:800;color:var(--blue-700);font-family:'Plus Jakarta Sans',sans-serif;">&lt;3s</div>
        <div style="font-size:11px;color:var(--slate-500);font-weight:600;letter-spacing:0.04em;margin-top:3px;">Decision Time</div>
      </div>
      <div class="hero-stat" style="background:var(--blue-50);padding:1.5rem 1rem;text-align:center;">
        <div style="font-size:28px;font-weight:800;color:var(--blue-700);font-family:'Plus Jakarta Sans',sans-serif;">18+</div>
        <div style="font-size:11px;color:var(--slate-500);font-weight:600;letter-spacing:0.04em;margin-top:3px;">Data Features</div>
      </div>
      <div style="grid-column:1/-1;padding:1.25rem 1.5rem;border-top:1px solid var(--slate-200);">
        <div style="font-size:12px;font-weight:700;color:var(--slate-700);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em;">How it works</div>
        <div style="display:flex;flex-direction:column;gap:7px;">
          <div style="display:flex;align-items:center;gap:10px;font-size:12.5px;color:var(--slate-600);font-weight:500;">
            <div style="width:20px;height:20px;background:var(--blue-500);color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;">1</div>
            Fill your personal & financial details
          </div>
          <div style="display:flex;align-items:center;gap:10px;font-size:12.5px;color:var(--slate-600);font-weight:500;">
            <div style="width:20px;height:20px;background:var(--blue-500);color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;">2</div>
            Our model analyses 18+ credit signals
          </div>
          <div style="display:flex;align-items:center;gap:10px;font-size:12.5px;color:var(--slate-600);font-weight:500;">
            <div style="width:20px;height:20px;background:var(--blue-500);color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;">3</div>
            Receive instant eligibility result
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FORM
# ─────────────────────────────────────────────
SENTINEL = "— Select —"
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

with st.form("loan_form"):

    # ── STEP 1 ────────────────────────────────
    st.markdown("""
    <div class="sec">
      <div class="sec-head">
        <div class="sec-icon">👤</div>
        <div class="sec-title">Personal Information</div>
        <div class="sec-count">Step 1 of 3</div>
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)",
            min_value=None, max_value=None, value=None, step=1,
            placeholder="e.g. 32")
        gender = st.selectbox("Gender", [SENTINEL, "Male", "Female"])
    with c2:
        marital_status = st.selectbox("Marital Status", [SENTINEL, "Single", "Married"])
        dependents = st.number_input("Number of Dependents",
            min_value=None, max_value=None, value=None, step=1,
            placeholder="e.g. 2")
    with c3:
        edu_level = st.selectbox("Education Level", [SENTINEL, "Graduate", "Not Graduate"])
        st.markdown('<p class="hint">Highest qualification attained</p>', unsafe_allow_html=True)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── STEP 2 ────────────────────────────────
    st.markdown("""
    <div class="sec">
      <div class="sec-head">
        <div class="sec-icon">💼</div>
        <div class="sec-title">Financial Profile</div>
        <div class="sec-count">Step 2 of 3</div>
      </div>
    </div>""", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        applicant_income = st.number_input("Monthly Income (₹)",
            min_value=None, max_value=None, value=None, step=1000,
            placeholder="e.g. 55,000")
        coapplicant_income = st.number_input("Co-applicant Monthly Income (₹)",
            min_value=None, max_value=None, value=None, step=1000,
            placeholder="e.g. 20,000  ·  0 if none")
    with c5:
        credit_score = st.number_input("Credit Score (300 – 900)",
            min_value=None, max_value=None, value=None, step=1,
            placeholder="e.g. 720")
        savings = st.number_input("Total Savings (₹)",
            min_value=None, max_value=None, value=None, step=5000,
            placeholder="e.g. 1,50,000")
    with c6:
        dti_ratio = st.number_input("Debt-to-Income Ratio (%)",
            min_value=None, max_value=None, value=None, step=0.1, format="%.1f",
            placeholder="e.g. 28.5")
        existing_loans = st.number_input("Active Loan Count",
            min_value=None, max_value=None, value=None, step=1,
            placeholder="e.g. 1")

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── STEP 3 ────────────────────────────────
    st.markdown("""
    <div class="sec">
      <div class="sec-head">
        <div class="sec-icon">🏦</div>
        <div class="sec-title">Loan Details</div>
        <div class="sec-count">Step 3 of 3</div>
      </div>
    </div>""", unsafe_allow_html=True)

    c7, c8, c9 = st.columns(3)
    with c7:
        loan_amount = st.number_input("Requested Loan Amount (₹)",
            min_value=None, max_value=None, value=None, step=5000,
            placeholder="e.g. 5,00,000")
        loan_term = st.number_input("Loan Tenure (months)",
            min_value=None, max_value=None, value=None, step=1,
            placeholder="e.g. 60")
    with c8:
        loan_purpose  = st.selectbox("Loan Purpose",  [SENTINEL, "Personal", "Car", "Business", "Home", "Education"])
        property_area = st.selectbox("Property Area", [SENTINEL, "Urban", "Semiurban", "Rural"])
    with c9:
        employment_status = st.selectbox("Employment Status",  [SENTINEL, "Salaried", "Self-Employed", "Contract", "Unemployed"])
        emp_category      = st.selectbox("Employer Category",  [SENTINEL, "Private", "Government", "MNC", "Business", "Unemployed"])
        collateral_value  = st.number_input("Collateral / Asset Value (₹)",
            min_value=None, max_value=None, value=None, step=10000,
            placeholder="e.g. 2,00,000  ·  0 if none")

    # ── DISCLAIMER + CTA ──────────────────────
    st.markdown("""
    <div style='height:1.5rem'></div>
    <div class="disclaimer">
      ⚠ &nbsp;<strong>Important:</strong> All fields are mandatory. Ensure all values are
      accurate — inputs are fed directly into the credit model. This result is indicative
      only and does not constitute a formal loan offer or guarantee of approval.
    </div>
    """, unsafe_allow_html=True)

    _, col_cta, _ = st.columns([1, 2, 1])
    with col_cta:
        submit = st.form_submit_button("🔍  Run Credit Assessment")


# ─────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────
def validate(num, sel):
    errs = []
    for k, label in [
        ("age","Age"), ("applicant_income","Monthly Income"),
        ("coapplicant_income","Co-applicant Income"), ("credit_score","Credit Score"),
        ("savings","Total Savings"), ("dti_ratio","DTI Ratio"),
        ("existing_loans","Active Loan Count"), ("loan_amount","Loan Amount"),
        ("loan_term","Loan Tenure"), ("collateral_value","Collateral Value"),
    ]:
        if num.get(k) is None:
            errs.append(f"{label} is required")
    for k, label in [
        ("gender","Gender"), ("marital_status","Marital Status"),
        ("edu_level","Education Level"), ("loan_purpose","Loan Purpose"),
        ("property_area","Property Area"), ("employment_status","Employment Status"),
        ("emp_category","Employer Category"),
    ]:
        if sel.get(k) == SENTINEL or sel.get(k) is None:
            errs.append(f"{label} must be selected")
    if num.get("age") is not None and not (18 <= num["age"] <= 100):
        errs.append("Age must be between 18 and 100")
    if num.get("credit_score") is not None and not (300 <= num["credit_score"] <= 900):
        errs.append("Credit Score must be between 300 and 900")
    if num.get("dti_ratio") is not None and not (0 <= num["dti_ratio"] <= 100):
        errs.append("DTI Ratio must be between 0% and 100%")
    return errs


# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
if submit:
    num_vals = dict(age=age, applicant_income=applicant_income,
        coapplicant_income=coapplicant_income, credit_score=credit_score,
        savings=savings, dti_ratio=dti_ratio, existing_loans=existing_loans,
        loan_amount=loan_amount, loan_term=loan_term, collateral_value=collateral_value)
    sel_vals = dict(gender=gender, marital_status=marital_status, edu_level=edu_level,
        loan_purpose=loan_purpose, property_area=property_area,
        employment_status=employment_status, emp_category=emp_category)
    errors = validate(num_vals, sel_vals)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if errors:
        items = "".join(f"<li>{e}</li>" for e in errors)
        st.markdown(f"""
        <div class="val-box">
          <span style="font-size:18px;flex-shrink:0;">⚠</span>
          <div>
            <strong>Please fix {len(errors)} issue{'s' if len(errors)>1 else ''} before submitting</strong>
            <ul>{items}</ul>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Build dataframe
        input_df = pd.DataFrame([{
            "Applicant_Income":   applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Employment_Status":  employment_status,
            "Age":                age,
            "Marital_Status":     marital_status,
            "Dependents":         dependents if dependents is not None else 0,
            "Credit_Score":       credit_score,
            "Existing_Loans":     existing_loans,
            "DTI_Ratio":          dti_ratio,
            "Savings":            savings,
            "Collateral_Value":   collateral_value,
            "Loan_Amount":        loan_amount,
            "Loan_Term":          loan_term,
            "Loan_Purpose":       loan_purpose,
            "Property_Area":      property_area,
            "Education_Level":    edu_level,
            "Gender":             gender,
            "Employer_Category":  emp_category,
        }])

        input_df["Education_Level"]      = input_df["Education_Level"].map({"Graduate":1,"Not Graduate":0})
        input_df["DTI_Ratio_sq"]         = input_df["DTI_Ratio"] ** 2
        input_df["Credit_Score_sq"]      = input_df["Credit_Score"] ** 2
        input_df["Applicant_Income_log"] = np.log1p(input_df["Applicant_Income"])
        input_df = input_df.drop(columns=["DTI_Ratio","Credit_Score"])

        cat_cols   = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
        encoded    = encoder.transform(input_df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        input_df   = pd.concat([input_df.drop(columns=cat_cols), encoded_df], axis=1)
        input_df   = input_df.reindex(columns=columns, fill_value=0)

        input_scaled = scaler.transform(input_df)
        prediction   = model.predict(input_scaled)
        proba        = model.predict_proba(input_scaled)[0][1]
        pct          = proba * 100

        approved = prediction[0] == 1
        r_class  = "approved" if approved else "rejected"
        r_color  = "var(--success)" if approved else "var(--danger)"
        r_title  = "✅ &nbsp;Loan Approved" if approved else "❌ &nbsp;Application Declined"
        r_sub    = ("Your application meets our eligibility criteria. A lending officer will contact you within 24–48 hours to discuss next steps."
                    if approved else
                    "Your current profile does not meet our lending criteria. Focus on improving your credit score and reducing existing debt obligations.")
        badge_class = "badge-approved" if approved else "badge-rejected"
        badge_text  = "ELIGIBLE" if approved else "NOT ELIGIBLE"

        if pct >= 75:
            conf_label, conf_color = "High Confidence", "var(--success)"
        elif pct >= 50:
            conf_label, conf_color = "Moderate Confidence", "#D97706"
        else:
            conf_label, conf_color = "Low Confidence", "var(--danger)"

        cs_color  = "var(--success)" if credit_score >= 700 else "#D97706" if credit_score >= 600 else "var(--danger)"
        dti_color = "var(--success)" if dti_ratio <= 35 else "#D97706" if dti_ratio <= 50 else "var(--danger)"

        _, col_r, _ = st.columns([1, 3, 1])
        with col_r:
            st.markdown(f"""
            <div class="result-wrap">
              <div class="result-header {r_class}">
                <div class="status-badge {badge_class}">{badge_text}</div>
                <div class="result-verdict" style="color:{r_color};">{r_title}</div>
                <div class="result-sub">{r_sub}</div>
                <div class="result-score" style="color:{r_color};">{pct:.1f}<span style="font-size:26px;font-weight:400;color:var(--slate-400);">%</span></div>
                <div class="result-score-label">Approval Confidence Score</div>
              </div>

              <div class="result-body">
                <div class="conf-row">
                  <span class="conf-label">Confidence Level</span>
                  <span class="conf-value" style="color:{conf_color};">{conf_label}</span>
                </div>
            """, unsafe_allow_html=True)

            st.progress(int(pct))

            st.markdown(f"""
                <div class="div"></div>
                <div style="font-size:12px;font-weight:700;letter-spacing:0.06em;
                            text-transform:uppercase;color:var(--slate-500);
                            margin-bottom:10px;">Application Summary</div>
                <div class="metric-grid">
                  <div class="metric-card">
                    <div class="metric-label">Loan Amount</div>
                    <div class="metric-value">₹{loan_amount:,.0f}</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-label">Credit Score</div>
                    <div class="metric-value" style="color:{cs_color};">{int(credit_score)}</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-label">DTI Ratio</div>
                    <div class="metric-value" style="color:{dti_color};">{dti_ratio:.1f}%</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-label">Monthly Income</div>
                    <div class="metric-value">₹{applicant_income:,.0f}</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-label">Loan Tenure</div>
                    <div class="metric-value">{int(loan_term)} mo</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-label">Total Savings</div>
                    <div class="metric-value">₹{savings:,.0f}</div>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="foot">
  <div class="foot-logo">🏦 &nbsp;Loan<span>Lens</span></div>
  <div class="foot-copy">
    For informational purposes only — not a guarantee of loan approval or a formal financial offer.
    &nbsp;·&nbsp; © 2025 LoanLens Credit Intelligence
  </div>
</div>
""", unsafe_allow_html=True)
