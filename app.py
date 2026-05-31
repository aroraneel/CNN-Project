import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os
import urllib.request
import gzip
import tempfile

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');

html, body { margin: 0; padding: 0; }
[data-testid="stAppViewContainer"] { background: #f5f0e8 !important; min-height: 100vh; }
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
* { box-sizing: border-box; }

.navbar {
    background: #0a3d3d; padding: 0 40px; height: 68px;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 999;
}
.nav-logo { font-family:'Cormorant Garamond',serif; font-size:1.8rem; font-weight:700; color:#e8b84b; letter-spacing:3px; }
.nav-tag { font-family:'Space Mono',monospace; font-size:0.75rem; letter-spacing:2px; color:#e8b84b; border:1px solid #e8b84b66; padding:6px 16px; border-radius:999px; }

.hero { background:linear-gradient(135deg,#0a3d3d 0%,#062828 100%); padding:72px 40px 60px; text-align:center; }
.hero-title { font-family:'Cormorant Garamond',serif; font-size:clamp(3.5rem,8vw,6.5rem); font-weight:700; color:#f5f0e8; margin:0 0 8px; line-height:1.05; }
.hero-title span { color:#e8b84b; }
.hero-sub { font-family:'DM Sans',sans-serif; font-size:clamp(1rem,2.5vw,1.25rem); color:#a8c4c4; margin:16px auto 0; max-width:560px; line-height:1.7; }

.stats-bar { background:#0a3d3d; border-top:1px solid #ffffff11; display:flex; justify-content:center; flex-wrap:wrap; }
.stat-item { padding:20px 36px; text-align:center; border-right:1px solid #ffffff11; flex:1; min-width:110px; }
.stat-item:last-child { border-right:none; }
.stat-val { font-family:'Cormorant Garamond',serif; font-size:2rem; font-weight:700; color:#e8b84b; line-height:1; }
.stat-key { font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:2px; color:#7aaaa8; margin-top:6px; }

.content-wrap { max-width:1300px; margin:0 auto; padding:40px 36px 56px; }
@media(max-width:768px){ .content-wrap{padding:24px 16px 36px;} .hero{padding:48px 20px 40px;} .navbar{padding:0 20px;} .stat-item{padding:14px 16px;} }

.sec-label { font-family:'Space Mono',monospace; font-size:0.75rem; letter-spacing:3px; color:#0a3d3d; margin-bottom:18px; text-transform:uppercase; display:flex; align-items:center; gap:12px; font-weight:700; }
.sec-label::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,#e8b84b88,transparent); }

.card { background:#ffffff; border:1px solid #ddd5c0; border-radius:20px; padding:28px; box-shadow:0 4px 24px rgba(10,61,61,0.07); }

[data-testid="stTabs"] [data-baseweb="tab-list"] { background:transparent !important; border-bottom:2px solid #ddd5c0 !important; gap:0 !important; margin-bottom:24px !important; }
[data-testid="stTabs"] [data-baseweb="tab"] { background:transparent !important; border:none !important; color:#888 !important; font-family:'Space Mono',monospace !important; font-size:0.8rem !important; letter-spacing:2px !important; padding:14px 28px !important; border-bottom:2px solid transparent !important; margin-bottom:-2px !important; font-weight:700 !important; }
[data-testid="stTabs"] [aria-selected="true"] { color:#0a3d3d !important; border-bottom:2px solid #e8b84b !important; }

[data-testid="stButton"] > button { background:#0a3d3d !important; color:#e8b84b !important; border:2px solid #e8b84b66 !important; border-radius:12px !important; padding:16px 28px !important; font-family:'Space Mono',monospace !important; font-size:0.85rem !important; letter-spacing:3px !important; width:100% !important; margin-top:16px !important; font-weight:700 !important; box-shadow:0 4px 16px rgba(10,61,61,0.25) !important; transition:all 0.2s !important; }
[data-testid="stButton"] > button:hover { background:#0d5050 !important; box-shadow:0 8px 28px rgba(10,61,61,0.35) !important; transform:translateY(-2px) !important; }

.pred-banner { background:#0a3d3d; border-radius:18px; padding:28px 32px; margin-bottom:24px; border-left:5px solid #e8b84b; }
.pred-tag { font-family:'Space Mono',monospace; font-size:0.75rem; letter-spacing:3px; color:#7aaaa8; margin-bottom:10px; }
.pred-name { font-family:'Cormorant Garamond',serif; font-size:clamp(2.4rem,5vw,3.5rem); font-weight:700; color:#f5f0e8; line-height:1.1; }
.pred-conf { font-family:'Space Mono',monospace; font-size:1rem; color:#e8b84b; margin-top:10px; font-weight:700; }

.prob-wrap { display:flex; flex-direction:column; gap:10px; }
.prob-row { display:flex; align-items:center; gap:14px; }
.prob-label { font-family:'DM Sans',sans-serif; font-size:0.9rem; color:#444; width:140px; flex-shrink:0; font-weight:500; }
.prob-label.top { color:#0a3d3d; font-weight:700; }
.prob-bar-bg { flex:1; height:10px; background:#ede8de; border-radius:999px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:999px; background:#c8d8c8; }
.prob-bar-fill.top { background:linear-gradient(90deg,#0a3d3d,#e8b84b); }
.prob-pct { font-family:'Space Mono',monospace; font-size:0.78rem; color:#999; width:40px; text-align:right; flex-shrink:0; font-weight:700; }
.prob-pct.top { color:#e8b84b; }

.idle-state { display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:300px; gap:16px; border:2px dashed #ddd5c0; border-radius:18px; background:#faf7f2; }
.idle-icon { font-size:4rem; }
.idle-text { font-family:'Cormorant Garamond',serif; font-size:1.5rem; color:#bbb; font-style:italic; }
.idle-hint { font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:2px; color:#ccc; }

[data-testid="stColorPicker"] label,[data-testid="stSlider"] label,[data-testid="stSelectbox"] label { font-family:'DM Sans',sans-serif !important; font-size:0.9rem !important; color:#333 !important; font-weight:600 !important; }
[data-testid="stSelectbox"] > div > div { background:#faf7f2 !important; border:1px solid #ddd5c0 !important; border-radius:10px !important; font-size:0.95rem !important; color:#222 !important; font-weight:500 !important; }
div[data-baseweb="slider"] > div:first-child { background:linear-gradient(90deg,#0a3d3d,#e8b84b) !important; }

[data-testid="stFileUploader"] section { border:2px dashed #ddd5c0 !important; border-radius:14px !important; background:#faf7f2 !important; padding:24px !important; }
[data-testid="stFileUploaderDropzoneInstructions"] p { font-size:0.95rem !important; color:#777 !important; }

.tips-box { background:#ffffff; border:1px solid #ddd5c0; border-radius:16px; padding:22px 26px; margin-top:20px; }
.tips-title { font-family:'Space Mono',monospace; font-size:0.72rem; letter-spacing:3px; color:#0a3d3d; font-weight:700; margin-bottom:14px; }
.tip-item { font-family:'DM Sans',sans-serif; font-size:0.95rem; color:#555; padding:5px 0; display:flex; gap:10px; align-items:flex-start; line-height:1.5; }
.tip-dot { color:#e8b84b; font-size:1rem; flex-shrink:0; }

[data-testid="stAlert"] { border-radius:12px !important; }
[data-testid="stAlert"] p { font-size:1rem !important; color:#333 !important; }
[data-testid="stSpinner"] p { font-size:1rem !important; color:#0a3d3d !important; }

.footer { background:#0a3d3d; text-align:center; padding:24px; font-family:'Space Mono',monospace; font-size:0.72rem; letter-spacing:3px; color:#e8b84b99; margin-top:24px; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ["T-shirt/Top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]
CLASS_ICONS = ["👕","👖","🧥","👗","🧥","👡","👔","👟","👜","👢"]

# Use system temp dir — works on Windows, Mac, Linux
TEMP_DIR = tempfile.gettempdir()

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model("models/fashion_cnn_model.h5")

@st.cache_resource(show_spinner="Loading sample images...")
def load_samples():
    base = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
    def dl(fname):
        path = os.path.join(TEMP_DIR, f"fmnist_{fname}")
        if not os.path.exists(path):
            urllib.request.urlretrieve(base + fname, path)
        return path
    def load_imgs(path):
        with gzip.open(path) as f:
            f.read(16)
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    def load_lbls(path):
        with gzip.open(path) as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)
    imgs = load_imgs(dl("t10k-images-idx3-ubyte.gz"))
    lbls = load_lbls(dl("t10k-labels-idx1-ubyte.gz"))
    samples = []
    for cls in range(10):
        idx = np.where(lbls == cls)[0][2]
        samples.append((imgs[idx], cls))
    return samples

def preprocess_canvas(canvas_data):
    img = Image.fromarray(canvas_data.astype("uint8"), "RGBA").convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)
    return (np.array(img).astype("float32") / 255.0).reshape(1, 28, 28, 1)

def preprocess_upload(pil_img):
    img = pil_img.convert("L").resize((28, 28), Image.LANCZOS)
    return (np.array(img).astype("float32") / 255.0).reshape(1, 28, 28, 1)

def preprocess_array(arr):
    img = Image.fromarray(arr).resize((28, 28), Image.LANCZOS)
    return (np.array(img).astype("float32") / 255.0).reshape(1, 28, 28, 1)

def run_prediction(tensor):
    model = load_model()
    return model.predict(tensor, verbose=0)[0]

def show_prediction(probs):
    top_idx  = int(np.argmax(probs))
    top_conf = float(probs[top_idx]) * 100
    st.markdown(f"""
    <div class="pred-banner">
        <div class="pred-tag">DETECTED CLASS</div>
        <div class="pred-name">{CLASS_ICONS[top_idx]} {CLASS_NAMES[top_idx]}</div>
        <div class="pred-conf">{top_conf:.1f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sec-label">All Probabilities</div>', unsafe_allow_html=True)
    bars = '<div class="card"><div class="prob-wrap">'
    for i in np.argsort(probs)[::-1]:
        pct = float(probs[i]) * 100
        t   = i == top_idx
        bars += f"""<div class="prob-row">
            <span class="prob-label {'top' if t else ''}">{CLASS_ICONS[i]} {CLASS_NAMES[i]}</span>
            <div class="prob-bar-bg"><div class="prob-bar-fill {'top' if t else ''}" style="width:{pct:.1f}%"></div></div>
            <span class="prob-pct {'top' if t else ''}">{pct:.0f}%</span>
        </div>"""
    bars += '</div></div>'
    st.markdown(bars, unsafe_allow_html=True)

# ── Session state to persist predictions across reruns ────────────────────────
if "probs" not in st.session_state:
    st.session_state.probs = None

# ── NAVBAR ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="nav-logo">Fashion MNIST</div>
    <div class="nav-tag">✦ CNN CLASSIFIER</div>
</div>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1 class="hero-title">Classify <span>Fashion</span><br>with AI</h1>
    <p class="hero-sub">Draw, upload, or pick a sample — our CNN predicts your clothing item instantly</p>
</div>
<div class="stats-bar">
    <div class="stat-item"><div class="stat-val">94.26%</div><div class="stat-key">ACCURACY</div></div>
    <div class="stat-item"><div class="stat-val">60K</div><div class="stat-key">TRAIN IMAGES</div></div>
    <div class="stat-item"><div class="stat-val">10</div><div class="stat-key">CLASSES</div></div>
    <div class="stat-item"><div class="stat-val">5</div><div class="stat-key">CONV LAYERS</div></div>
    <div class="stat-item"><div class="stat-val">0.938</div><div class="stat-key">KAGGLE SCORE</div></div>
</div>
""", unsafe_allow_html=True)

# ── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.markdown('<div class="sec-label">✦ Input Method</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["✏️  DRAW", "📁  UPLOAD", "🖼️  SAMPLES"])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            stroke_color = st.color_picker("Brush Color", "#0a3d3d")
        with c2:
            stroke_width = st.slider("Brush Size", 8, 35, 20)
        with c3:
            mode = st.selectbox("Draw Mode", ["freedraw","line","rect","circle"])
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#000000",
            height=300, width=None,
            drawing_mode=mode,
            key="canvas",
            display_toolbar=True,
        )
        if st.button("⚡  PREDICT DRAWING", key="btn_draw"):
            if canvas_result.image_data is not None and canvas_result.image_data[:,:,:3].sum() > 1000:
                with st.spinner("Classifying..."):
                    st.session_state.probs = run_prediction(preprocess_canvas(canvas_result.image_data))
            else:
                st.warning("Canvas is empty — draw a clothing item first!")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p style="font-family:DM Sans,sans-serif;font-size:1rem;color:#555;margin-bottom:16px;line-height:1.6;">Upload any clothing photo. It will be auto-converted to 28×28 grayscale.</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose an image", type=["png","jpg","jpeg","webp"])
        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption="Uploaded image", use_column_width=True)
            if st.button("⚡  PREDICT UPLOAD", key="btn_upload"):
                with st.spinner("Classifying..."):
                    st.session_state.probs = run_prediction(preprocess_upload(pil_img))
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p style="font-family:DM Sans,sans-serif;font-size:1rem;color:#555;margin-bottom:20px;line-height:1.6;">One real sample per class from the Fashion-MNIST test set. Click any button to classify it.</p>', unsafe_allow_html=True)
        samples = load_samples()
        cols = st.columns(5)
        for i, (img_arr, cls) in enumerate(samples):
            with cols[i % 5]:
                pil = Image.fromarray(img_arr).resize((90, 90), Image.NEAREST)
                st.image(pil, use_column_width=True)
                if st.button(f"{CLASS_ICONS[cls]} {CLASS_NAMES[cls]}", key=f"s{i}", use_container_width=True):
                    with st.spinner("Classifying..."):
                        st.session_state.probs = run_prediction(preprocess_array(img_arr))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="tips-box">
        <div class="tips-title">✦ TIPS FOR BEST RESULTS</div>
        <div class="tip-item"><span class="tip-dot">◆</span><span>Use <strong>thick strokes</strong> — thin lines confuse the model</span></div>
        <div class="tip-item"><span class="tip-dot">◆</span><span>Fill the canvas — don't draw too small</span></div>
        <div class="tip-item"><span class="tip-dot">◆</span><span>Use the <strong>undo ↩</strong> icon in the canvas toolbar to undo strokes</span></div>
        <div class="tip-item"><span class="tip-dot">◆</span><span>Centre your drawing inside the black box</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="sec-label">✦ Prediction Output</div>', unsafe_allow_html=True)
    if st.session_state.probs is not None:
        show_prediction(st.session_state.probs)
    else:
        st.markdown("""
        <div class="idle-state">
            <div class="idle-icon">✦</div>
            <div class="idle-text">Awaiting your input</div>
            <div class="idle-hint">DRAW · UPLOAD · OR PICK A SAMPLE</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    FASHION MNIST CNN &nbsp;✦&nbsp; NEEL ARORA &nbsp;✦&nbsp; BCA UNDERGRADUATE &nbsp;✦&nbsp; 94.26% ACCURACY
</div>
""", unsafe_allow_html=True)