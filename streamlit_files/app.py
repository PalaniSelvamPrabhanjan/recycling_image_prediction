import io
import numpy as np
import streamlit as st
from PIL import Image

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoScan · Waste Classifier",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Disposal guide ────────────────────────────────────────────────────────────
DISPOSAL_INFO = {
    "battery": {
        "icon": "🔋",
        "action": "Hazardous Disposal",
        "color": "#FF4500",
        "bg": "#fff5f0",
        "tip": "Never throw batteries in the bin. Take them to a dedicated battery drop-off point at supermarkets or electronics stores.",
        "recycle": False,
    },
    "biological": {
        "icon": "🍃",
        "action": "Compost",
        "color": "#2E7D32",
        "bg": "#f1f8e9",
        "tip": "Organic/biological waste can go in your compost bin or green waste collection. Keep it out of landfill to reduce methane emissions.",
        "recycle": True,
    },
    "cardboard": {
        "icon": "📦",
        "action": "Recycle",
        "color": "#1565C0",
        "bg": "#e3f2fd",
        "tip": "Flatten boxes to save space. Remove tape and staples where possible. Wet cardboard should go in general waste.",
        "recycle": True,
    },
    "clothes": {
        "icon": "👕",
        "action": "Donate / Textile Bin",
        "color": "#6A1B9A",
        "bg": "#f3e5f5",
        "tip": "If still wearable, donate to a charity shop. Otherwise use a textile recycling bank — clothes should never go to landfill.",
        "recycle": True,
    },
    "glass": {
        "icon": "🫙",
        "action": "Recycle",
        "color": "#00838F",
        "bg": "#e0f7fa",
        "tip": "Rinse jars and bottles before recycling. Separate by colour if your local scheme requires it. Broken glass goes in general waste, wrapped safely.",
        "recycle": True,
    },
    "metal": {
        "icon": "🥫",
        "action": "Recycle",
        "color": "#37474F",
        "bg": "#eceff1",
        "tip": "Rinse food tins and aluminium cans. Scrunch small pieces of foil into a ball so they don't fall through sorting machinery.",
        "recycle": True,
    },
    "paper": {
        "icon": "📄",
        "action": "Recycle",
        "color": "#1565C0",
        "bg": "#e3f2fd",
        "tip": "Newspapers, magazines and office paper are easily recycled. Greasy or soiled paper (e.g. pizza boxes) should go in general waste.",
        "recycle": True,
    },
    "plastic": {
        "icon": "♳",
        "action": "Check & Recycle",
        "color": "#E65100",
        "bg": "#fff3e0",
        "tip": "Check the recycling number on the bottom. Types 1 (PET) and 2 (HDPE) are widely accepted. Rinse containers and remove lids where instructed.",
        "recycle": True,
    },
    "shoes": {
        "icon": "👟",
        "action": "Donate / Shoe Bank",
        "color": "#6A1B9A",
        "bg": "#f3e5f5",
        "tip": "Wearable shoes can be donated to charity. Many shoe brands and retailers also run take-back schemes for recycling old footwear.",
        "recycle": True,
    },
    "trash": {
        "icon": "🗑️",
        "action": "General Waste",
        "color": "#757575",
        "bg": "#fafafa",
        "tip": "This item appears to be general waste. Try to reduce single-use items where possible and look for sustainable alternatives in future.",
        "recycle": False,
    },
}

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #f7f9f4;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 760px; }

/* ── Fix file uploader filename color ── */
[data-testid="stFileUploaderFileName"],
.stFileUploaderFileName,
div[class*="stFileUploaderFileName"] {
    color: #1a2e1a !important;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: #e8f5e9;
    color: #2E7D32;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 1rem;
    border: 1px solid #c8e6c9;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #1a2e1a;
    line-height: 1.1;
    margin: 0 0 0.6rem;
}
.hero h1 span { color: #4CAF50; }
.hero p {
    color: #5a6b5a;
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Upload zone ── */
.upload-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #2E7D32;
    margin-bottom: 0.5rem;
    display: block;
}

/* ── Result card ── */
.result-card {
    border-radius: 20px;
    padding: 2.2rem;
    margin-top: 1.5rem;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    min-height: 320px;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-header {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1rem;
}
.result-icon {
    font-size: 3.5rem;
    line-height: 1;
    flex-shrink: 0;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.55;
    margin-bottom: 0.1rem;
}
.result-class {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1rem, 3vw, 1.6rem);
    font-weight: 800;
    line-height: 1.1;
    text-transform: capitalize;
    word-break: break-word;
    overflow-wrap: break-word;
}
.action-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    margin-top: 0.4rem;
}
.tip-box {
    background: rgba(255,255,255,0.7);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #3a4a3a;
    margin-top: 0.8rem;
    border: 1px solid rgba(0,0,0,0.06);
}
.tip-box strong { color: #1a2e1a; }

/* ── Confidence bar ── */
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    font-weight: 500;
    color: #5a6b5a;
    margin-bottom: 0.25rem;
    margin-top: 1rem;
}
.conf-track {
    height: 8px;
    background: rgba(0,0,0,0.07);
    border-radius: 999px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

/* ── How it works ── */
.how-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 2rem 0 0;
}
.how-card {
    background: white;
    border-radius: 16px;
    padding: 1.3rem 1rem;
    text-align: center;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.how-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #4CAF50;
    margin-bottom: 0.3rem;
}
.how-text {
    font-size: 0.82rem;
    color: #5a6b5a;
    line-height: 1.5;
}

/* ── Footer ── */
.eco-footer {
    text-align: center;
    margin-top: 3rem;
    font-size: 0.78rem;
    color: #8a9a8a;
}
</style>
""", unsafe_allow_html=True)


# ─── Model helpers ──────────────────────────────────────────────────────────────
def load_interpreter(model_path: str):
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        return Interpreter(model_path=model_path)
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        return Interpreter(model_path=model_path)


@st.cache_resource
def get_interpreter():
    interpreter = load_interpreter("final_model.tflite")
    interpreter.allocate_tensors()
    return interpreter


@st.cache_data
def load_labels(path="labels(1).txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


def prepare_image(pil_img: Image.Image, input_shape):
    img = pil_img.convert("RGB")
    if len(input_shape) == 4 and input_shape[-1] == 3:
        h, w = int(input_shape[1]), int(input_shape[2])
        nhwc = True
    elif len(input_shape) == 4 and input_shape[1] == 3:
        h, w = int(input_shape[2]), int(input_shape[3])
        nhwc = False
    else:
        h, w = 256, 256
        nhwc = True
    img = img.resize((w, h))
    arr = np.array(img)
    if nhwc:
        arr = arr.reshape(1, h, w, 3)
    else:
        arr = arr.transpose(2, 0, 1).reshape(1, 3, h, w)
    return arr


def set_input(interpreter, img_arr: np.ndarray, mode: str):
    info = interpreter.get_input_details()[0]
    idx = info["index"]
    dtype = info["dtype"]
    scale, zero = info.get("quantization", (0.0, 0))
    if dtype == np.float32:
        x = img_arr.astype(np.float32)
        if mode == "0..1":
            x = x / 255.0
        elif mode == "resnet_preprocess":
            from tensorflow.keras.applications.resnet50 import preprocess_input  # type: ignore
            x = preprocess_input(x)
    elif dtype == np.uint8:
        x = img_arr.astype(np.float32)
        if scale and scale > 0:
            x = (x / scale + zero).round()
        x = np.clip(x, 0, 255).astype(np.uint8)
    elif dtype == np.int8:
        x = img_arr.astype(np.float32)
        if scale and scale > 0:
            x = (x / scale + zero).round()
        x = np.clip(x, -128, 127).astype(np.int8)
    else:
        x = img_arr.astype(dtype)
    interpreter.set_tensor(idx, x)


def predict(pil_img: Image.Image, mode: str, show_debug: bool):
    interpreter = get_interpreter()
    labels = load_labels()
    in_info = interpreter.get_input_details()[0]
    out_info = interpreter.get_output_details()[0]
    if show_debug:
        with st.sidebar:
            st.subheader("🛠 Debug Info")
            st.write("Input shape:", in_info["shape"])
            st.write("Input dtype:", in_info["dtype"])
            st.write("Input quant:", in_info.get("quantization", None))
            st.write("Output shape:", out_info["shape"])
            st.write("Output dtype:", out_info["dtype"])
            st.write("Output quant:", out_info.get("quantization", None))
    img_arr = prepare_image(pil_img, in_info["shape"])
    set_input(interpreter, img_arr, mode)
    interpreter.invoke()
    out = interpreter.get_tensor(out_info["index"])
    logits = np.squeeze(out).astype(np.float32)
    out_dtype = out_info["dtype"]
    out_scale, out_zero = out_info.get("quantization", (0.0, 0))
    if out_dtype in (np.uint8, np.int8) and out_scale and out_scale > 0:
        logits = (logits - out_zero) * out_scale
    probs = softmax(logits)
    topk = min(5, len(probs))
    idxs = np.argsort(probs)[::-1][:topk]
    results = [(labels[i] if i < len(labels) else f"class_{i}", float(probs[i])) for i in idxs]
    return results[0][0], results[0][1]


# ─── Sidebar settings ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    mode_label = st.selectbox(
        "Input scaling",
        ["0..255", "0..1", "resnet_preprocess"],
        index=0,
        help="How pixel values are normalised before being fed to the model. Match this to your training pipeline."
    )
    show_debug = st.checkbox("Show tensor debug info", value=False)
    st.markdown("---")
    st.markdown("**Recognisable items**")
    for key, val in DISPOSAL_INFO.items():
        st.markdown(f"{val['icon']} {key.capitalize()}")

mode_map = {"0..255": "0..255", "0..1": "0..1", "resnet_preprocess": "resnet_preprocess"}
mode = mode_map[mode_label]

# ─── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">♻️ AI-Powered Waste Guide</div>
    <h1>Snap it.<br><span>Sort it right.</span></h1>
    <p>Upload a photo of any item and we'll instantly tell you how to dispose of it responsibly.</p>
</div>
""", unsafe_allow_html=True)

# ─── Upload ─────────────────────────────────────────────────────────────────────
st.markdown('<span class="upload-label">📸 Upload your item</span>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="collapsed",
)

# ─── Results ────────────────────────────────────────────────────────────────────
if uploaded:
    pil_img = Image.open(io.BytesIO(uploaded.read()))

    col_img, col_res = st.columns([1, 1.4], gap="large")

    with col_img:
        # Convert image to base64 for inline HTML display with controlled height
        import base64
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(f"""
        <div style="height:100%; min-height:320px;">
            <img src="data:image/png;base64,{b64}"
                 style="width:100%; height:100%; min-height:320px; object-fit:cover;
                        border-radius:16px; display:block;" />
        </div>
        """, unsafe_allow_html=True)

    with col_res:
        with st.spinner("Analysing..."):
            pred_label, pred_conf = predict(pil_img, mode, show_debug)

        info = DISPOSAL_INFO.get(pred_label.lower(), {
            "icon": "❓", "action": "Unknown", "color": "#888",
            "bg": "#f5f5f5", "tip": "We couldn't find disposal info for this item.", "recycle": False
        })

        action_emoji = "✅" if info["recycle"] else "⚠️"
        action_bg = f"{info['color']}18"

        # Result card
        st.markdown(f"""
        <div class="result-card" style="background:{info['bg']};">
            <div class="result-header">
                <div class="result-icon">{info['icon']}</div>
                <div>
                    <div class="result-label" style="color:{info['color']};">Detected item</div>
                    <div class="result-class" style="color:{info['color']};">{pred_label.capitalize()}</div>
                    <div class="action-pill" style="background:{action_bg}; color:{info['color']};">
                        {action_emoji} {info['action']}
                    </div>
                </div>
            </div>
            <div class="tip-box">
                💡 <strong>How to dispose:</strong> {info['tip']}
            </div>

        </div>
        """, unsafe_allow_html=True)

else:
    # How it works
    st.markdown("""
    <div class="how-grid">
        <div class="how-card">
            <div class="how-num">01</div>
            <div class="how-text">📸 Upload a photo of any household item or waste</div>
        </div>
        <div class="how-card">
            <div class="how-num">02</div>
            <div class="how-text">🤖 Our AI model identifies what the item is</div>
        </div>
        <div class="how-card">
            <div class="how-num">03</div>
            <div class="how-text">♻️ Get instant disposal advice to help the planet</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:2rem; padding: 1.5rem; background:white; border-radius:16px; border:1px dashed #c8e6c9;">
        <div style="font-size:2.5rem; margin-bottom:0.5rem;">📂</div>
        <div style="font-family:'Syne',sans-serif; font-weight:700; color:#1a2e1a; margin-bottom:0.25rem;">Drop your image above</div>
        <div style="font-size:0.85rem; color:#5a6b5a;">Supports PNG, JPG, JPEG, WEBP</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="eco-footer">
    EcoScan · Helping you make smarter disposal decisions, one item at a time 🌱
</div>
""", unsafe_allow_html=True)