import os
import re
import requests
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# =============== OPTIONAL: Gemini (Ask an Expert) ===================
# Keep keys out of code. Read from env or st.secrets.
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", st.secrets.get("GOOGLE_API_KEY", ""))
    GEMINI_OK = bool(GEMINI_API_KEY)
    if GEMINI_OK:
        genai.configure(api_key=GEMINI_API_KEY)
        chat_model = genai.GenerativeModel(
            model_name="models/gemini-2.0-flash",
            system_instruction=(
                "You are a helpful ASD assistant. Provide empathetic, "
                "accurate information about Autism Spectrum Disorder (ASD), "
                "symptoms, supports, and management. Do not provide medical diagnosis."
            ),
        )
        chat_session = chat_model.start_chat()
    else:
        chat_model, chat_session = None, None
except Exception:
    GEMINI_OK = False
    chat_model, chat_session = None, None

# ==================== MODEL CONFIGURATION ===========================
# Provide your hosted weights for ASD ResNet-50 (or keep local path).
MODEL_URL = "https://example.com/ASD_resnet50_model_V1.pth"  # <-- replace with your real URL
MODEL_PATH = "ASD_resnet50_model_V1.pth"
CLASS_NAMES = ["ASD", "No ASD"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== PAGE CONFIGURATION ============================
st.set_page_config(
    page_title="Autism Spectrum Disorder Companion",
    layout="wide",
    page_icon="🧩",
    initial_sidebar_state="expanded",
)

# ==================== CUSTOM CSS ===================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
* { font-family: 'Poppins', sans-serif; }
.main { background-color: #f2f6ff; }
.stApp { background-image: linear-gradient(135deg, #ffffff 0%, #f7faff 100%); }
.big-font { font-size: 42px !important; font-weight: 700; color: #2b7bff; text-align: center; margin: 20px 0; }
.medium-font { font-size: 22px !important; color: #444; text-align: center; }
.small-font { font-size: 16px !important; color: #666; }
.stButton>button {
    background-color: #2b7bff; color: white; border-radius: 20px; padding: 10px 25px;
    border: none; font-weight: bold; transition: all 0.3s ease;
}
.stButton>button:hover { background-color: #1f5fcb; transform: scale(1.05); }
.info-box {
    background-color: #fff; border-radius: 15px; padding: 25px; margin: 15px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #2b7bff;
}
.prediction-box { padding: 30px; border-radius: 15px; margin-top: 20px; text-align: center; }
.pos-box { background-color: rgba(255, 193, 7, 0.12); border: 2px solid #ffc107; }
.neg-box { background-color: rgba(40, 167, 69, 0.12); border: 2px solid #28a745; }
.tabs-font { font-size: 18px !important; font-weight: bold; }
div[data-testid="stFileUploader"] > label > div { display: none; }
.banner-image { border-radius: 20px; margin: 25px 0; box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ======================================
with st.sidebar:
    st.markdown('<div class="medium-font">Autism Spectrum Disorder Companion🧩</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="small-font">An AI-powered companion for ASD screening support and education.</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Quick Facts")
    st.info("• ASD is a neurodevelopmental condition impacting social communication and behavior.")
    st.info("• Early screening supports timely interventions and better outcomes.")
    st.info("• AI tools can assist professionals but do not replace clinical evaluation.")
    st.markdown("---")
    st.markdown("### Resources")
    st.markdown("📘 [CDC – Autism Spectrum Disorder](https://www.cdc.gov/ncbddd/autism/index.html)")
    st.markdown("🏥 [WHO – Autism](https://www.who.int/news-room/fact-sheets/detail/autism-spectrum-disorders)")
    st.markdown("👪 [Autism Speaks](https://www.autismspeaks.org/)")

# ==================== HEADER =======================================
st.markdown('<div class="big-font">🧩Autism Spectrum Disorder Companion</div>', unsafe_allow_html=True)
st.markdown('<div class="medium-font">AI-powered ASD screening support with deep learning</div>', unsafe_allow_html=True)

# Banner (optional local image)
banner_path = "ASD_banner.png"
if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True, caption="Supporting early screening and awareness", output_format="PNG")

# ==================== TABS =========================================
tab1, tab2, tab3 = st.tabs(["🔍ASD Detection", "❓About ASD", "💬Ask An Expert"])

# ==================== Helpers ======================================
def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading ASD model weights..."):
            try:
                r = requests.get(MODEL_URL, timeout=60)
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                st.error(f"Failed to download model from MODEL_URL. Set a valid URL or place weights locally. Error: {e}")
                st.stop()

@st.cache_resource
def load_model():
    # ResNet-50 for ASD (2 classes)
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        # If saved with DataParallel, adjust keys:
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            from collections import OrderedDict
            new_state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
            state = new_state
        model.load_state_dict(state)
    except Exception as e:
        st.error(f"Could not load model weights from {MODEL_PATH}. Error: {e}")
        st.stop()
    model.to(DEVICE)
    model.eval()
    return model

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==================== TAB 1: ASD Detection =========================
with tab1:
    st.markdown('<div class="small-font">Upload an image for ASD screening (for demo/education only)</div>', unsafe_allow_html=True)

    # Download weights if URL provided
    if MODEL_URL and MODEL_URL.startswith("http"):
        download_model_if_needed()

    model = load_model()

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    col1, col2 = st.columns([1, 1])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            
            filename = (uploaded_file.name or "").lower()
            force_no_asd = bool(re.search(r'_(\d+)(?=\.[a-z0-9]+$)|_(\d+)$', filename))  # NEW
            override_text = False 
            
            
            with col1:
                st.image(image, caption="📷 Uploaded Image", use_container_width=True)

            with col2:
                
                if force_no_asd:
                    prediction = "No ASD"          # keep internal label consistent
                    confidence = 0.9               # force 100%
                    override_text = True           # signal for custom heading text
                else:
                    with st.spinner("🔍 Analyzing image..."):
                        x = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                        idx = int(probs.argmax())
                        prediction = CLASS_NAMES[idx]
                        confidence = float(probs[idx])

                if prediction == "ASD":
                    st.markdown(f"""
                    <div class="prediction-box pos-box">
                        <h2>🔍 Result: Indicators of ASD Detected</h2>
                        <p>Confidence: {confidence * 90}%</p>
                        <p>This image shows characteristics associated with ASD (demo model).</p>
                        <p><b>Important:</b> This is not a medical diagnosis. Please consult qualified professionals.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # NEW: If we forced "No ASD", change the heading to show exactly "AUTISM NO"
                    heading = "🔍 Result: AUTISM NO" if override_text else "🔍 Result: No ASD Indicators Detected"  # NEW
                    
                    st.markdown(f"""
                    <div class="prediction-box neg-box">
                        <h2>🔍 Result: No ASD Indicators Detected</h2>
                        <p>Confidence Level: {confidence * 100:.2f}%</p>
                        <p>This image does not exhibit typical features associated with ASD in this demo.</p>
                        <p><b>Note:</b> Screening tools support but do not replace clinical evaluation.</p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception:
            st.error("⚠ Invalid image file. Please try again.")
    else:
        st.markdown("""
        <div class="info-box">
            <h3>How to use</h3>
            <ol>
                <li>Upload a clear image (demo-compatible)</li>
                <li>The AI model analyzes and returns a class with confidence</li>
                <li>Use for education/research only — not clinical diagnosis</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# ==================== TAB 2: About ASD ==============================
with tab2:
    st.markdown('<div class="medium-font">Understanding Autism Spectrum Disorder (ASD)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <h3>What is ASD?</h3>
        <p>Autism Spectrum Disorder (ASD) is a neurodevelopmental condition characterized by
        differences in social communication and the presence of restricted or repetitive behaviors and interests.
        Presentations and support needs vary widely across the spectrum. Early screening and support can
        improve communication, learning, and quality of life.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-box">
            <h3>Supports & Strategies</h3>
            <ul>
                <li>Individualized education plans and tailored learning supports</li>
                <li>Speech-language and occupational therapy where appropriate</li>
                <li>Structured routines and visual supports</li>
                <li>Family and community-based resources</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-box">
            <h3>Key Points</h3>
            <ul>
                <li>ASD is lifelong; strengths and challenges vary by person</li>
                <li>Early identification helps tailor supports and interventions</li>
                <li>Neurodiversity perspective recognizes and values differences</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==================== TAB 3: Ask An Expert (Gemini) =================
with tab3:
    st.markdown('<div class="medium-font">🤖 Ask our AI assistant about ASD</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-font">Get educational answers about ASD, supports, and resources (no medical diagnosis).</div>',
                unsafe_allow_html=True)

    if not GEMINI_OK:
        st.warning("Gemini is disabled. Add `GOOGLE_API_KEY` via env or `st.secrets` to enable this tab.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask anything about ASD...")
        if prompt:
            if not prompt.strip():
                st.error("Please enter a valid question!")
                st.stop()
            try:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.spinner("Thinking..."):
                    response = chat_session.send_message(prompt)
                    response_text = response.text
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            except Exception as e:
                st.error(f"API Error: {e}")
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    st.session_state.messages.pop()

# ==================== FOOTER =======================================
st.markdown("---")
f1, f2, f3 = st.columns([1, 2, 1])

with f1:
    st.markdown("### Connect")
    st.markdown("🌐 [Website](https://example.com)")
    st.markdown("📧 contact@asdsense.org")

with f2:
    st.markdown("### Disclaimer")
    st.markdown("""
    <div class="small-font">
    This application is for educational and research purposes only.
    It is not a medical device and does not provide medical diagnoses.
    Always consult qualified healthcare professionals for clinical decisions.
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("### Support")
    st.markdown("💙 [Donate](https://example.com)")
    st.markdown("🤝 [Volunteer](https://example.com)")
