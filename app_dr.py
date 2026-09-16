import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
import datetime
import os
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="DR Vision AI - MobileNetV2 vs Random Forest",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] { background-color: #0f172a !important; }
[data-testid="stSidebar"] * { color: #f1f5f9 !important; }

.badge {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 5px 13px; border-radius: 20px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.04em; margin-bottom: 14px;
}
.badge-a { background: rgba(59,130,246,.15); color: #60a5fa; border: 1px solid rgba(59,130,246,.35); }
.badge-b { background: rgba(139,92,246,.15); color: #a78bfa; border: 1px solid rgba(139,92,246,.35); }

.diag-box { border-radius: 12px; padding: 16px; text-align: center; margin: 14px 0; }
.diag-label { font-size: 11px; font-weight: 700; text-transform: uppercase;
               letter-spacing: .06em; margin-bottom: 6px; opacity: .8; }
.diag-name  { font-size: 22px; font-weight: 800; margin-bottom: 4px; }
.diag-conf  { font-size: 13px; font-weight: 500; opacity: .85; }

.consensus-agree {
    background: rgba(16,185,129,.12); border: 1px solid #10b981; color: #10b981;
    border-radius: 10px; padding: 12px 18px; margin-bottom: 20px;
    font-size: 14px; font-weight: 600;
}
.consensus-warn {
    background: rgba(245,158,11,.12); border: 1px solid #f59e0b; color: #f59e0b;
    border-radius: 10px; padding: 12px 18px; margin-bottom: 20px;
    font-size: 14px; font-weight: 600;
}

.prob-row { margin-bottom: 10px; }
.prob-labels { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px; }
.bar-track { width: 100%; background: rgba(128,128,128,.2); border-radius: 999px; height: 7px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 999px; }

.kpi-card {
    border-radius: 12px; padding: 16px 14px;
    border: 1px solid rgba(128,128,128,.2); border-top-width: 4px;
}
.kpi-title { font-size: 11px; font-weight: 600; text-transform: uppercase;
              letter-spacing: .05em; opacity: .6; margin-bottom: 6px; }
.kpi-value { font-size: 22px; font-weight: 800; margin-bottom: 2px; }
.kpi-sub   { font-size: 11px; opacity: .55; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
CLASS_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
CLASS_COLORS = ['#10b981', '#facc15', '#f97316', '#ef4444', '#a855f7']

# ── Model Loading ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_cnn_model():
    """Load trained MobileNetV2 Keras model."""
    path = 'models/mobilenetv2_dr.keras'
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    try:
        m = tf.keras.models.load_model(path)
        return m, None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner=False)
def load_rf_model():
    """Load Random Forest classifier from joblib pickle."""
    path = 'models/random_forest_model.pkl'
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    try:
        m = joblib.load(path)
        return m, None
    except Exception as e:
        return None, str(e)

# ── Preprocessing ─────────────────────────────────────────────

def preprocess_cnn(pil_img):
    """224×224 RGB, normalized to [0, 1] by / 255.0"""
    img = pil_img.convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)

def preprocess_rf(pil_img):
    """64×64 RGB, flattened to 1-D feature vector"""
    img = pil_img.convert('RGB').resize((64, 64))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.flatten().reshape(1, -1)  # (1, 12288)

# ── Inference ─────────────────────────────────────────────────

def predict_cnn(model, pil_img):
    arr = preprocess_cnn(pil_img)
    proba = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(proba))
    return idx, float(proba[idx]) * 100, proba

def predict_rf(model, pil_img):
    arr = preprocess_rf(pil_img)
    idx = int(model.predict(arr)[0])
    raw_proba = model.predict_proba(arr)[0]
    classes = getattr(model, 'classes_', list(range(len(raw_proba))))
    full_proba = np.zeros(5, dtype=np.float32)
    for cls, p in zip(classes, raw_proba):
        if 0 <= int(cls) < 5:
            full_proba[int(cls)] = p
    conf = float(full_proba[idx]) * 100 if full_proba[idx] > 0 else float(max(raw_proba)) * 100
    return idx, conf, full_proba

# ── UI Helpers ────────────────────────────────────────────────

def render_diagnosis_box(label, color, pred_name, pred_class, conf):
    bg = color + "1f"
    st.markdown(f"""
        <div class="diag-box" style="background:{bg}; border:1px solid {color};">
            <div class="diag-label" style="color:{color};">{label}</div>
            <div class="diag-name"  style="color:{color};">{pred_name}</div>
            <div style="font-size:13px; font-weight:600; color:{color}; opacity:.75; margin-bottom:8px;">
                Severity Class {pred_class}
            </div>
            <hr style="border:none; border-top:1px dashed {color}55; margin:8px 0;">
            <div class="diag-conf" style="color:{color};">Confidence: <b>{conf:.1f}%</b></div>
        </div>
    """, unsafe_allow_html=True)

def render_prob_bars(probas, pred_idx):
    st.markdown(
        "<div style='font-size:13px; font-weight:700; opacity:.7; margin-bottom:10px;'>"
        "Probability Distribution</div>",
        unsafe_allow_html=True
    )
    for i, (label, prob) in enumerate(zip(CLASS_LABELS, probas)):
        is_top = (i == pred_idx)
        bar_color = CLASS_COLORS[i] if is_top else "rgba(128,128,128,0.35)"
        weight = "700" if is_top else "400"
        pct = float(prob) * 100
        st.markdown(f"""
            <div class="prob-row">
                <div class="prob-labels">
                    <span style="font-weight:{weight};">Class {i}: {label}</span>
                    <span style="font-weight:{weight};">{pct:.1f}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct:.2f}%; background:{bar_color};"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────
with st.spinner("Loading CNN & Random Forest models..."):
    cnn_model, cnn_err = load_cnn_model()
    rf_model,  rf_err  = load_rf_model()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👁️ **DR Vision AI**")
    st.markdown("<p style='color:#94a3b8; font-size:13px; margin-top:-10px;'>MobileNetV2 vs Random Forest</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
        <div style='background:#1e293b; padding:14px; border-radius:12px;
                    margin-bottom:12px; border:1px solid #334155;'>
            <div style='font-size:10px; color:#93c5fd; text-transform:uppercase;
                        font-weight:700; letter-spacing:.05em; margin-bottom:5px;'>Model A · CNN</div>
            <div style='font-size:14px; color:#f8fafc; font-weight:600;'>🧠 MobileNetV2 (Custom Trained)</div>
            <div style='font-size:12px; color:#64748b; margin-top:4px;'>mobilenetv2_dr.keras</div>
            <div style='font-size:12px; color:#64748b; margin-top:2px;'>Input: 224×224 RGB / 255.0</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='background:#1e293b; padding:14px; border-radius:12px;
                    margin-bottom:16px; border:1px solid #334155;'>
            <div style='font-size:10px; color:#c4b5fd; text-transform:uppercase;
                        font-weight:700; letter-spacing:.05em; margin-bottom:5px;'>Model B · Classical ML</div>
            <div style='font-size:14px; color:#f8fafc; font-weight:600;'>🌲 Random Forest</div>
            <div style='font-size:12px; color:#64748b; margin-top:4px;'>random_forest_model.pkl</div>
            <div style='font-size:12px; color:#64748b; margin-top:2px;'>Input: 64×64 RGB → flatten()</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("**Model Status**")
    if cnn_model:
        st.success("✅ MobileNetV2 loaded")
    else:
        st.error(f"❌ MobileNetV2: {cnn_err}")
    if rf_model:
        st.success("✅ Random Forest loaded")
    else:
        st.error(f"❌ RF: {rf_err}")

    st.markdown("---")
    st.caption(f"🕐 {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}")
    st.caption("© 2026 DR Vision AI")

# ── Page Header ───────────────────────────────────────────────
st.markdown("## 👁️ Diabetic Retinopathy — MobileNetV2 vs Random Forest")
st.markdown(
    "Komparasi prediksi **Model A (MobileNetV2 Custom Trained CNN)** dan **Model B (Random Forest)** "
    "secara berdampingan pada citra fundus retina yang sama."
)

# ── KPI Row ───────────────────────────────────────────────────
kpi_cols = st.columns(5)
kpi_data = [
    ("DR Classes",  "5",                            "No DR → Proliferative",  "#3b82f6"),
    ("CNN Input",   "224×224",                      "RGB / 255.0 normalize",  "#06b6d4"),
    ("RF Input",    "64×64 → 1D",                   "12,288 pixel features",  "#8b5cf6"),
    ("CNN Type",    "MobileNetV2 (Custom Trained)", "Deep Learning",          "#10b981"),
    ("RF Type",     "Random Forest",                "Classical ML",           "#f97316"),
]
for col, (title, val, sub, color) in zip(kpi_cols, kpi_data):
    with col:
        st.markdown(f"""
            <div class="kpi-card" style="border-top-color:{color};">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value" style="color:{color};">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Upload & Side-by-Side Prediction ─────────────────────────
st.markdown("### 🔬 Upload Retinal Image & Compare Models")

uploaded_file = st.file_uploader(
    "Upload a fundus retina image (JPG / PNG / JPEG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert('RGB')

    cnn_ok = cnn_model is not None
    rf_ok  = rf_model  is not None
    cnn_pred_idx = cnn_conf = cnn_proba = None
    rf_pred_idx  = rf_conf  = rf_proba  = None

    with st.spinner("Running inference on both models..."):
        if cnn_ok:
            try:
                cnn_pred_idx, cnn_conf, cnn_proba = predict_cnn(cnn_model, pil_img)
            except Exception as ex:
                st.error(f"CNN inference error: {ex}")
                cnn_ok = False
        if rf_ok:
            try:
                rf_pred_idx, rf_conf, rf_proba = predict_rf(rf_model, pil_img)
            except Exception as ex:
                st.error(f"Random Forest inference error: {ex}")
                rf_ok = False

    # Consensus banner
    if cnn_ok and rf_ok:
        if cnn_pred_idx == rf_pred_idx:
            st.markdown(f"""
                <div class="consensus-agree">
                    ✅ <b>Diagnostic Consensus:</b> Both models agree —
                    <u>{CLASS_LABELS[cnn_pred_idx]} (Class {cnn_pred_idx})</u>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="consensus-warn">
                    ⚠️ <b>Discrepancy Detected:</b>
                    CNN → <u>{CLASS_LABELS[cnn_pred_idx]} ({cnn_conf:.1f}%)</u> &nbsp;|&nbsp;
                    RF  → <u>{CLASS_LABELS[rf_pred_idx]}  ({rf_conf:.1f}%)</u>.
                    Manual review recommended.
                </div>
            """, unsafe_allow_html=True)

    # ── Side-by-Side ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    # COL 1 — Model A: MobileNetV2 (Custom Trained)
    with col1:
        st.markdown('<div class="badge badge-a">● Model A · MobileNetV2 (Custom Trained)</div>',
                    unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True, caption="Fundus Input (CNN – 224×224)")

        if cnn_ok:
            render_diagnosis_box(
                label="MobileNetV2 Diagnosis",
                color=CLASS_COLORS[cnn_pred_idx],
                pred_name=CLASS_LABELS[cnn_pred_idx],
                pred_class=cnn_pred_idx,
                conf=cnn_conf
            )
            render_prob_bars(cnn_proba, cnn_pred_idx)
        else:
            st.warning("MobileNetV2 model is not available.")

    # COL 2 — Model B: Random Forest
    with col2:
        st.markdown('<div class="badge badge-b">● Model B · Random Forest</div>',
                    unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True,
                 caption="Fundus Input (RF – 64×64 flatten)")

        if rf_ok:
            render_diagnosis_box(
                label="Random Forest Diagnosis",
                color=CLASS_COLORS[rf_pred_idx],
                pred_name=CLASS_LABELS[rf_pred_idx],
                pred_class=rf_pred_idx,
                conf=rf_conf
            )
            render_prob_bars(rf_proba, rf_pred_idx)
        else:
            st.warning("Random Forest model is not available.")

else:
    st.info("📂 Upload a retinal fundus image above to start the side-by-side comparison.")

# ── Bottom Charts ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
chart_col, metric_col = st.columns(2)

with chart_col:
    st.markdown("#### 📊 Validation Dataset Distribution")
    labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    values = [312, 428, 276, 168, 64]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.6,
        marker_colors=CLASS_COLORS,
        textinfo='percent+label', textfont_size=12
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10), height=290,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(text='<b>1,248</b><br>Images', x=.5, y=.5,
                          font_size=14, showarrow=False)]
    )
    st.plotly_chart(fig, use_container_width=True)

with metric_col:
    st.markdown("#### 🎯 Model Comparison Summary")
    bench_df = pd.DataFrame({
        "Metric":          ["Architecture", "Input Size",    "Preprocessing",      "Model Type",     "Inference"],
        "Model A · CNN":   ["MobileNetV2 (Custom Trained)", "224×224 px", "/ 255.0 normalize", "Deep Learning", "GPU/CPU"],
        "Model B · RF":    ["Random Forest", "64×64 px",    "flatten → 1D",       "Classical ML",   "CPU"],
    })
    st.dataframe(bench_df, use_container_width=True, hide_index=True)
    st.caption(
        "💡 **MobileNetV2** extracts deep spatial feature hierarchies. "
        "**Random Forest** uses raw flattened pixels — simpler but typically less accurate on complex retinal images."
    )

# ── DR Severity Reference ─────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### 📋 DR Severity Scale Reference")
ref_cols = st.columns(5)
ref_data = [
    ("Class 0", "No DR",           "#10b981", "No lesions detected"),
    ("Class 1", "Mild",            "#facc15", "Microaneurysms only"),
    ("Class 2", "Moderate",        "#f97316", "More than mild NPDR"),
    ("Class 3", "Severe",          "#ef4444", "Extensive hemorrhages"),
    ("Class 4", "Proliferative DR","#a855f7", "Neovascularization present"),
]
for col, (cls, name, color, desc) in zip(ref_cols, ref_data):
    with col:
        st.markdown(f"""
            <div style="border-radius:10px; padding:12px; border:1px solid {color}40;
                        background:{color}12; text-align:center;">
                <div style="font-size:11px; font-weight:700; color:{color};
                            text-transform:uppercase; letter-spacing:.05em;">{cls}</div>
                <div style="font-size:15px; font-weight:800; color:{color}; margin:4px 0;">{name}</div>
                <div style="font-size:11px; opacity:.6;">{desc}</div>
            </div>
        """, unsafe_allow_html=True)
