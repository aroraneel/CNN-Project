import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os, urllib.request, gzip, tempfile

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Fashion AI",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body { margin:0; padding:0; }
* { box-sizing:border-box; font-family:'Inter',sans-serif; }

[data-testid="stAppViewContainer"] { background:#f0f4f8 !important; }
[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="stDecoration"],[data-testid="stSidebar"] { display:none !important; }
.block-container { padding:0 !important; max-width:100% !important; }

/* HIDE default streamlit button styles in nav area */
section[data-testid="stMain"] > div:first-child { padding-top:0 !important; }

/* PAGE WRAPPER */
.page { max-width:1100px; margin:0 auto; padding:40px 40px 80px; }

/* CARD */
.card {
  background:#ffffff;
  border:1px solid #e2e8f0;
  border-radius:14px;
  padding:28px;
  box-shadow:0 1px 4px rgba(0,0,0,0.06);
}

/* HEADING */
.page-title {
  font-size:2rem; font-weight:800;
  color:#1a202c; margin:0 0 6px;
  letter-spacing:-0.5px;
}
.page-sub {
  font-size:1rem; color:#718096;
  margin:0 0 32px; line-height:1.6;
}

/* STATS */
.stats {
  display:grid; grid-template-columns:repeat(4,1fr);
  gap:14px; margin-bottom:36px;
}
.stat-card {
  background:#ffffff; border:1px solid #e2e8f0;
  border-radius:12px; padding:20px;
  text-align:center;
  box-shadow:0 1px 3px rgba(0,0,0,0.05);
}
.stat-val {
  font-size:1.6rem; font-weight:800;
  color:#2d3748; line-height:1;
}
.stat-key {
  font-size:0.75rem; font-weight:600;
  color:#a0aec0; margin-top:5px;
  text-transform:uppercase; letter-spacing:0.5px;
}

/* HOME HERO */
.hero-section {
  background:linear-gradient(135deg,#1d4ed8 0%,#2563eb 100%);
  border-radius:16px; padding:48px 40px;
  text-align:center; margin-bottom:32px;
  color:#fff;
}
.hero-title {
  font-size:2.6rem; font-weight:800;
  line-height:1.2; margin:0 0 12px;
  letter-spacing:-0.5px;
}
.hero-sub-text {
  font-size:1.05rem; opacity:0.85;
  max-width:460px; margin:0 auto 28px;
  line-height:1.7;
}

/* CLASSES GRID */
.classes-grid {
  display:grid; grid-template-columns:repeat(5,1fr);
  gap:12px; margin-top:24px;
}
.class-chip {
  background:#ffffff; border:1px solid #e2e8f0;
  border-radius:10px; padding:14px 8px;
  text-align:center;
  box-shadow:0 1px 3px rgba(0,0,0,0.04);
}
.class-chip-icon { font-size:1.5rem; display:block; margin-bottom:5px; }
.class-chip-name {
  font-size:0.8rem; font-weight:600;
  color:#4a5568;
}

/* TABS */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background:#ffffff !important;
  border-bottom:2px solid #e2e8f0 !important;
  gap:0 !important; padding:0 !important;
  margin-bottom:0 !important;
  border-radius:14px 14px 0 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background:transparent !important; border:none !important;
  color:#718096 !important; font-size:0.9rem !important;
  font-weight:600 !important; padding:14px 28px !important;
  border-bottom:2px solid transparent !important;
  margin-bottom:-2px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color:#2563eb !important;
  border-bottom:2px solid #2563eb !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
  padding:24px !important;
  background:#ffffff !important;
  border-radius:0 0 14px 14px !important;
  border:1px solid #e2e8f0 !important;
  border-top:none !important;
}

/* WIDGET LABELS */
label[data-testid="stWidgetLabel"] p,
[data-testid="stColorPicker"] label,
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label {
  font-size:0.85rem !important; font-weight:600 !important;
  color:#4a5568 !important;
}
[data-testid="stSelectbox"] > div > div {
  background:#f7fafc !important; border:1px solid #e2e8f0 !important;
  border-radius:8px !important; color:#2d3748 !important;
  font-size:0.9rem !important; font-weight:500 !important;
}
div[data-baseweb="slider"] > div:first-child {
  background:linear-gradient(90deg,#1d4ed8,#3b82f6) !important;
}
[data-testid="stFileUploader"] section {
  background:#f7fafc !important; border:2px dashed #cbd5e0 !important;
  border-radius:10px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p {
  color:#718096 !important; font-size:0.9rem !important;
}
[data-testid="stFileUploader"] button {
  background:#2563eb !important; color:#fff !important;
  border:none !important; border-radius:8px !important;
  font-weight:600 !important;
}

/* BUTTONS */
[data-testid="stButton"] > button {
  background:#2563eb !important; color:#fff !important;
  border:none !important; border-radius:8px !important;
  padding:12px 28px !important; font-size:0.92rem !important;
  font-weight:700 !important; width:100% !important;
  margin-top:16px !important; transition:all 0.15s !important;
}
[data-testid="stButton"] > button:hover {
  background:#1d4ed8 !important; transform:translateY(-1px) !important;
}

/* ALERT */
[data-testid="stAlert"] { border-radius:8px !important; }
[data-testid="stAlert"] p { font-size:0.9rem !important; color:#2d3748 !important; }
[data-testid="stSpinner"] p { font-size:0.9rem !important; color:#718096 !important; }

/* RESULT */
.result-hero {
  background:linear-gradient(135deg,#1d4ed8 0%,#2563eb 100%);
  border-radius:14px; padding:36px 40px;
  margin-bottom:24px; color:#fff;
  display:flex; align-items:center; gap:32px;
}
.result-icon { font-size:5rem; line-height:1; }
.result-label {
  font-size:0.75rem; font-weight:700;
  opacity:0.7; text-transform:uppercase;
  letter-spacing:1px; margin-bottom:6px;
}
.result-class {
  font-size:3rem; font-weight:800;
  line-height:1.1; letter-spacing:-1px;
  margin-bottom:6px;
}
.result-conf { font-size:1rem; font-weight:600; opacity:0.85; }

/* PROB BARS */
.bars-title {
  font-size:0.78rem; font-weight:700;
  color:#a0aec0; text-transform:uppercase;
  letter-spacing:0.5px; margin-bottom:16px;
}
.prob-row { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
.prob-name { font-size:0.9rem; font-weight:500; color:#718096; width:140px; flex-shrink:0; }
.prob-name.top { color:#2d3748; font-weight:700; }
.prob-track { flex:1; height:8px; background:#edf2f7; border-radius:4px; overflow:hidden; }
.prob-fill { height:100%; background:#e2e8f0; border-radius:4px; }
.prob-fill.top { background:linear-gradient(90deg,#1d4ed8,#3b82f6); }
.prob-pct { font-size:0.82rem; font-weight:600; color:#a0aec0; width:38px; text-align:right; flex-shrink:0; }
.prob-pct.top { color:#2563eb; }

/* IDLE */
.idle {
  display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  min-height:280px; gap:12px;
  border:2px dashed #e2e8f0; border-radius:12px;
  background:#f7fafc;
}
.idle-icon { font-size:3rem; }
.idle-text { font-size:1rem; font-weight:600; color:#a0aec0; }
.idle-sub { font-size:0.82rem; color:#cbd5e0; }

/* TIPS */
.tips {
  background:#f7fafc; border-radius:10px;
  padding:16px 20px; margin-top:16px;
  border:1px solid #e2e8f0;
}
.tips-title {
  font-size:0.78rem; font-weight:700; color:#4a5568;
  text-transform:uppercase; letter-spacing:0.5px; margin-bottom:10px;
}
.tip { font-size:0.88rem; color:#718096; padding:3px 0; line-height:1.5; display:flex; gap:8px; }
.tip-dot { color:#2563eb; font-weight:700; flex-shrink:0; }

/* FOOTER */
.footer {
  text-align:center; padding:24px;
  font-size:0.82rem; color:#a0aec0;
  border-top:1px solid #e2e8f0;
  margin-top:16px;
}
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ["T-shirt/Top","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle Boot"]
CLASS_ICONS = ["👕","👖","🧥","👗","🧥","👡","👔","👟","👜","👢"]
TEMP_DIR    = tempfile.gettempdir()

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return tf.keras.models.load_model("models/fashion_cnn_model.h5")

@st.cache_resource(show_spinner="Loading samples…")
def load_samples():
    base = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
    def dl(fn):
        p = os.path.join(TEMP_DIR, f"fm_{fn}")
        if not os.path.exists(p): urllib.request.urlretrieve(base+fn, p)
        return p
    def li(p):
        with gzip.open(p) as f:
            f.read(16); return np.frombuffer(f.read(),dtype=np.uint8).reshape(-1,28,28)
    def ll(p):
        with gzip.open(p) as f:
            f.read(8); return np.frombuffer(f.read(),dtype=np.uint8)
    I = li(dl("t10k-images-idx3-ubyte.gz"))
    L = ll(dl("t10k-labels-idx1-ubyte.gz"))
    return [(I[np.where(L==c)[0][3]], c) for c in range(10)]

def pre_canvas(d):
    img = Image.fromarray(d.astype("uint8"),"RGBA").convert("L")
    return (np.array(ImageOps.invert(img).resize((28,28),Image.LANCZOS)).astype("float32")/255).reshape(1,28,28,1)
def pre_upload(p):
    return (np.array(p.convert("L").resize((28,28),Image.LANCZOS)).astype("float32")/255).reshape(1,28,28,1)
def pre_array(a):
    return (np.array(Image.fromarray(a).resize((28,28),Image.LANCZOS)).astype("float32")/255).reshape(1,28,28,1)
def do_predict(t):
    return load_model().predict(t,verbose=0)[0]
def render_pred(probs):
    top  = int(np.argmax(probs))
    conf = float(probs[top])*100
    st.markdown(f"""
    <div class="result-hero">
      <div class="result-icon">{CLASS_ICONS[top]}</div>
      <div>
        <div class="result-label">Predicted Class</div>
        <div class="result-class">{CLASS_NAMES[top]}</div>
        <div class="result-conf">Confidence: {conf:.1f}%</div>
      </div>
    </div>
    <div class="card">
      <div class="bars-title">All Class Probabilities</div>
    """, unsafe_allow_html=True)
    html = ""
    for i in np.argsort(probs)[::-1]:
        p  = float(probs[i])*100
        t2 = i==top
        html += f"""<div class="prob-row">
          <span class="prob-name {'top' if t2 else ''}">{CLASS_ICONS[i]} {CLASS_NAMES[i]}</span>
          <div class="prob-track"><div class="prob-fill {'top' if t2 else ''}" style="width:{p:.1f}%"></div></div>
          <span class="prob-pct {'top' if t2 else ''}">{p:.0f}%</span>
        </div>"""
    st.markdown(html + "</div>", unsafe_allow_html=True)

# SESSION STATE
if "page"  not in st.session_state: st.session_state.page  = "home"
if "probs" not in st.session_state: st.session_state.probs = None

# ═══════════════════
# NAV — pure Streamlit
# ═══════════════════
nav_l, nav_mid, nav_r = st.columns([3, 6, 3])
with nav_l:
    st.markdown('<p style="font-size:1.1rem;font-weight:800;color:#2d3748;padding:14px 0 0 0;margin:0;">Fashion AI</p>', unsafe_allow_html=True)
with nav_r:
    nb1, nb2 = st.columns(2)
    with nb1:
        if st.button("Home", key="nb_home"):
            st.session_state.page = "home"
            st.rerun()
    with nb2:
        if st.button("Classify", key="nb_cls"):
            st.session_state.page = "classify"
            st.rerun()

st.markdown('<hr style="margin:0 0 0 0;border:none;border-top:1px solid #e2e8f0;">', unsafe_allow_html=True)

# ═══════════════════
# HOME
# ═══════════════════
if st.session_state.page == "home":
    st.markdown('<div class="page">', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero-section">
      <h1 class="hero-title">What clothing is this?</h1>
      <p class="hero-sub-text">An AI model that identifies clothing from sketches, photos, or samples using a 5-layer CNN trained on 60,000 images.</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3 = st.columns([2,1,2])
    with c2:
        if st.button("Try it now →", key="try_btn"):
            st.session_state.page = "classify"
            st.rerun()

    st.markdown("""
    <div class="stats">
      <div class="stat-card"><div class="stat-val">94.26%</div><div class="stat-key">Accuracy</div></div>
      <div class="stat-card"><div class="stat-val">60,000</div><div class="stat-key">Train Images</div></div>
      <div class="stat-card"><div class="stat-val">10</div><div class="stat-key">Classes</div></div>
      <div class="stat-card"><div class="stat-val">0.938</div><div class="stat-key">Kaggle Score</div></div>
    </div>
    <p style="font-size:1rem;font-weight:700;color:#2d3748;margin:0 0 14px;">10 Clothing Classes</p>
    <div class="classes-grid">
    """, unsafe_allow_html=True)
    for icon,name in zip(CLASS_ICONS,CLASS_NAMES):
        st.markdown(f'<div class="class-chip"><span class="class-chip-icon">{icon}</span><span class="class-chip-name">{name}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════
# CLASSIFY
# ═══════════════════
elif st.session_state.page == "classify":
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Classify a Clothing Item</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Choose an input method below, then click Classify.</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["✏️  Draw", "📁  Upload", "🖼️  Samples"])

    with tab1:
        c1,c2,c3 = st.columns(3)
        with c1: color = st.color_picker("Brush Color","#ffffff")
        with c2: size  = st.slider("Brush Size",8,40,22)
        with c3: mode  = st.selectbox("Mode",["freedraw","line","rect","circle"])
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=size, stroke_color=color,
            background_color="#000000",
            height=440, width=680,
            drawing_mode=mode, key="canvas",
            display_toolbar=True,
        )
        if st.button("Classify Drawing →", key="btn_draw"):
            if canvas.image_data is not None and canvas.image_data[:,:,:3].sum() > 800:
                with st.spinner("Running model…"):
                    st.session_state.probs = do_predict(pre_canvas(canvas.image_data))
                st.session_state.page = "result"
                st.rerun()
            else:
                st.warning("Canvas is empty — draw something first.")
        st.markdown("""
        <div class="tips">
          <div class="tips-title">Tips for best results</div>
          <div class="tip"><span class="tip-dot">→</span>Use thick strokes — thin lines reduce accuracy</div>
          <div class="tip"><span class="tip-dot">→</span>Fill the canvas, don't draw too small</div>
          <div class="tip"><span class="tip-dot">→</span>Use ↩ in the toolbar to undo strokes</div>
          <div class="tip"><span class="tip-dot">→</span>Centre your drawing in the canvas</div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<p style="font-size:0.95rem;color:#718096;margin-bottom:16px;">Upload any clothing photo — auto-converted to 28×28 grayscale.</p>', unsafe_allow_html=True)
        f = st.file_uploader("Upload", type=["png","jpg","jpeg","webp"], label_visibility="collapsed")
        if f:
            img = Image.open(f)
            st.image(img, width=300)
            if st.button("Classify Image →", key="btn_up"):
                with st.spinner("Running model…"):
                    st.session_state.probs = do_predict(pre_upload(img))
                st.session_state.page = "result"
                st.rerun()

    with tab3:
        st.markdown('<p style="font-size:0.95rem;color:#718096;margin-bottom:20px;">Real images from the test set — click any to classify.</p>', unsafe_allow_html=True)
        samples = load_samples()
        cols = st.columns(5)
        for i,(arr,cls) in enumerate(samples):
            with cols[i%5]:
                st.image(Image.fromarray(arr).resize((90,90),Image.NEAREST), use_column_width=True)
                if st.button(f"{CLASS_ICONS[cls]} {CLASS_NAMES[cls]}", key=f"s{i}", use_container_width=True):
                    with st.spinner("Running model…"):
                        st.session_state.probs = do_predict(pre_array(arr))
                    st.session_state.page = "result"
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════
# RESULT
# ═══════════════════
elif st.session_state.page == "result":
    st.markdown('<div class="page">', unsafe_allow_html=True)

    if st.session_state.probs is not None:
        render_pred(st.session_state.probs)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,1,4])
    with c1:
        if st.button("← Try Again", key="try_again"):
            st.session_state.page = "classify"
            st.rerun()
    with c2:
        if st.button("Home", key="go_home"):
            st.session_state.page = "home"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown('<div class="footer">Fashion AI · CNN · Fashion-MNIST · 94.26% Accuracy</div>', unsafe_allow_html=True)