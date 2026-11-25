import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Fraud Shield", page_icon="💳", layout="centered")

# ---------- Load model artifacts ----------
compact_model = joblib.load('model/compact_model.pkl')
compact_features = joblib.load('model/compact_features.pkl')
scaler = joblib.load('model/scaler.pkl')
full_feature_names = joblib.load('model/feature_names.pkl')
samples = joblib.load('model/sample_transactions.pkl')

amount_idx = full_feature_names.index('Amount')
amount_mean = scaler.mean_[amount_idx]
amount_scale = scaler.scale_[amount_idx]

v_features = [f for f in compact_features if f != 'Amount']

# ---------- Session state ----------
st.session_state.setdefault('theme', 'dark')
st.session_state.setdefault('flash', False)
st.session_state.setdefault('genuine_idx', 0)
st.session_state.setdefault('fraud_idx', 0)
st.session_state.setdefault('active_preset', None)
for f in v_features:
    st.session_state.setdefault(f, 0.0)
st.session_state.setdefault('Amount', 100.0)


def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
    st.session_state.flash = True


def load_genuine():
    idx = st.session_state.genuine_idx % len(samples['genuine'])
    sample = samples['genuine'][idx]
    for f in v_features:
        st.session_state[f] = float(sample[f])
    raw_amount = sample['Amount'] * amount_scale + amount_mean
    st.session_state['Amount'] = round(float(raw_amount), 2)
    st.session_state.genuine_idx += 1
    st.session_state.active_preset = 'genuine'


def load_fraud():
    idx = st.session_state.fraud_idx % len(samples['fraud'])
    sample = samples['fraud'][idx]
    for f in v_features:
        st.session_state[f] = float(sample[f])
    raw_amount = sample['Amount'] * amount_scale + amount_mean
    st.session_state['Amount'] = round(float(raw_amount), 2)
    st.session_state.fraud_idx += 1
    st.session_state.active_preset = 'fraud'


# ---------- Theme colors ----------
if st.session_state.theme == 'dark':
    bg = "#111318"
    text_color = "#ffffff"
    sub_color = "#c9ced6"
    card_bg = "#1c1f26"
    card_border = "#3a3f4b"
    accent = "#f4c542"
    caption_color = "#9aa1ad"
else:
    bg = "#f5f6fa"
    text_color = "#111318"
    sub_color = "#3a3f4b"
    card_bg = "#ffffff"
    card_border = "#d7dae0"
    accent = "#b8860b"
    caption_color = "#5a6070"

flash_css = ""
if st.session_state.flash:
    flash_css = """
    <div style="
        position:fixed; top:0; left:0; width:100%; height:100%;
        background:white; z-index:9999; pointer-events:none;
        animation: flicker 0.6s ease-out forwards;">
    </div>
    <style>
    @keyframes flicker {
        0%   { opacity: 0.95; }
        15%  { opacity: 0.1; }
        30%  { opacity: 0.85; }
        45%  { opacity: 0.05; }
        60%  { opacity: 0.6; }
        100% { opacity: 0; }
    }
    </style>
    """
    st.session_state.flash = False

genuine_active = st.session_state.active_preset == 'genuine'
fraud_active = st.session_state.active_preset == 'fraud'

st.markdown(f"""
{flash_css}
<style>
.stApp {{
    background: {bg} !important;
}}
.card {{
    background: {card_bg};
    border-radius: 18px;
    padding: 28px 32px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    border: 1px solid {card_border};
    margin-bottom: 24px;
}}
.card-chip {{
    width: 42px; height: 32px;
    background: linear-gradient(135deg, #d4af37, #f4e5a3);
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 12px;
}}
.card-title {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 1px;
    color: {accent} !important;
}}
.card-sub {{
    font-size: 13px;
    color: {sub_color} !important;
    letter-spacing: 2px;
    margin-top: -4px;
}}

/* Body text */
[data-testid="stAppViewContainer"] * {{
    color: {text_color};
}}

/* Slider labels + values */
[data-testid="stSliderThumbValue"],
[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"] {{
    color: {text_color} !important;
    font-weight: 600 !important;
}}
[data-testid="stWidgetLabel"] p {{
    color: {text_color} !important;
    font-weight: 500;
}}
[data-testid="stCaptionContainer"] {{
    color: {caption_color} !important;
}}

/* ---- BUTTONS: force explicit background + text color ---- */
div.stButton > button {{
    background-color: {card_bg} !important;
    color: {text_color} !important;
    border: 1px solid {card_border} !important;
    border-radius: 10px;
    font-weight: 600;
    padding: 8px 0;
    transition: all 0.35s ease;
}}
div.stButton > button p {{
    color: {text_color} !important;
}}
div.stButton > button[kind="primary"] {{
    background-color: {accent} !important;
    color: #111318 !important;
    border: none !important;
}}
div.stButton > button[kind="primary"] p {{
    color: #111318 !important;
}}

/* ---- NUMBER INPUT: force explicit background + text color ---- */
[data-testid="stNumberInput"] input {{
    background-color: {card_bg} !important;
    color: {text_color} !important;
    border: 1px solid {card_border} !important;
}}
[data-testid="stNumberInput"] button {{
    background-color: {card_bg} !important;
    color: {text_color} !important;
}}

/* Preset button glow/dim states */
.st-key-btn_genuine button {{
    opacity: {'1' if genuine_active or not fraud_active else '0.5'};
    box-shadow: {'0 0 14px 2px rgba(34,197,94,0.6)' if genuine_active else 'none'};
}}
.st-key-btn_fraud button {{
    opacity: {'1' if fraud_active or not genuine_active else '0.5'};
    box-shadow: {'0 0 14px 2px rgba(239,68,68,0.6)' if fraud_active else 'none'};
}}
</style>
""", unsafe_allow_html=True)

# ---------- Theme toggle ----------
top_col1, top_col2 = st.columns([4, 1])
with top_col2:
    st.button("🌙/☀️ Toggle Theme", on_click=toggle_theme, use_container_width=True)

# ---------- Header card ----------
st.markdown("""
<div class="card">
    <div class="card-chip"></div>
    <div class="card-title">💳 FRAUD SHIELD</div>
    <div class="card-sub">AI-POWERED TRANSACTION RISK ANALYSIS</div>
</div>
""", unsafe_allow_html=True)

st.write("Adjust the transaction details below, or load a real example.")

# ---------- Preset buttons ----------
col1, col2 = st.columns(2)
with col1:
    st.button("✅ Genuine Preset", key="btn_genuine", on_click=load_genuine, use_container_width=True)
with col2:
    st.button("🚨 Fraud Preset", key="btn_fraud", on_click=load_fraud, use_container_width=True)

st.caption(f"Genuine examples seen: {st.session_state.genuine_idx} · Fraud examples seen: {st.session_state.fraud_idx}")

st.divider()

# ---------- Input fields ----------
st.markdown("**Transaction Amount**")
st.number_input("Amount ($)", min_value=0.0, step=1.0, key='Amount', label_visibility="collapsed")

st.markdown("**Key Feature Signals** (anonymized transaction components)")
for f in v_features:
    st.slider(f, -10.0, 10.0, step=0.01, key=f)

st.divider()

# ---------- Predict ----------
if st.button("🔍 Analyze Transaction", use_container_width=True, type="primary"):
    scaled_amount = (st.session_state['Amount'] - amount_mean) / amount_scale

    input_row = []
    for f in compact_features:
        input_row.append(scaled_amount if f == 'Amount' else st.session_state[f])

    input_array = np.array(input_row).reshape(1, -1)
    prediction = compact_model.predict(input_array)[0]
    probability = compact_model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error("⚠️ **Fraudulent Transaction Detected**")
    else:
        st.success("✅ **Genuine Transaction**")

    st.metric("Fraud Probability", f"{probability * 100:.2f}%")
    st.progress(float(probability))

st.divider()
st.caption("Model: Random Forest · Trained on top 5 predictive components + transaction amount"),