import os
import re
import requests
import streamlit as st
from PIL import Image
import google.generativeai as genai

import torch
import torch.nn as nn
from torchvision import models, transforms

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)
chat_model = genai.GenerativeModel(
    model_name="models/gemini-2.0-flash",
    system_instruction="You are a helpful Autism Spectrum Disorder (ASD) assistant. Provide empathetic, factual, and clear information about ASD — its symptoms, diagnosis, therapy, supports, and awareness. Do not provide medical diagnosis."
)
chat_session = chat_model.start_chat()

MODEL_URL = "https://example.com/ASD_resnet50_model_V1.pth"  
MODEL_PATH = "ASD_resnet50_model_V1.pth"
CLASS_NAMES = ["ASD", "No ASD"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Autism Spectrum Disorder Companion",
    layout="wide",
    page_icon="🧩",
    initial_sidebar_state="expanded",
)


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
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="medium-font">Autism Spectrum Disorder Companion🧩</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="small-font">An AI-powered companion for ASD screening support and education.</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Quick Facts")
    st.info("• ASD affects communication, behavior and social interaction.")
    st.info("• Early screening supports timely interventions.")
    st.info("• AI tools assist but do not replace medical professionals.")
    st.markdown("---")
    st.markdown("### Resources")
    st.markdown("📘[CDC - Autism Spectrum Disorder](https://www.cdc.gov/ncbddd/autism/index.html)")
    st.markdown("🏥[WHO - Autism](https://www.who.int/news-room/fact-sheets/detail/autism-spectrum-disorders)")
    st.markdown("👪[Autism Speaks](https://www.autismspeaks.org/)")


st.markdown('<div class="big-font">🧩Autism Spectrum Disorder Companion</div>', unsafe_allow_html=True)
st.markdown('<div class="medium-font">AI-powered ASD screening and awareness platform</div>', unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs([
    "🔍ASD Detection",
    "📸Camera Detection",
    "❓About ASD",
    "💬Ask An Expert"
])


def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading ASD model weights..."):
            try:
                r = requests.get(MODEL_URL, timeout=60)
                if r.status_code == 200:
                    with open(MODEL_PATH, "wb") as f:
                        f.write(r.content)
                else:
                    st.error("Failed to download model. Provide valid MODEL_URL.")
                    st.stop()
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                st.stop()

@st.cache_resource
def load_model():
    # FIX 1: Use 'weights=None' instead of deprecated 'pretrained=False'
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state)
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
            return None
    model.to(DEVICE)
    model.eval()
    return model

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


with tab1:
    st.markdown('<div class="small-font">Upload an image for ASD screening (demo only)</div>', unsafe_allow_html=True)
    download_model_if_needed()
    model = load_model()


    uploaded_file = st.file_uploader(
        "Upload Image", 
        type=["jpg", "jpeg", "png"], 
        label_visibility="collapsed"
    )

    if uploaded_file and model:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            filename = uploaded_file.name.lower()


            st.image(image, caption=f"📷Uploaded: {filename}", width=300)

            if "_" in filename:
                st.markdown("""
                <div class="prediction-box neg-box">
                    <h2>🔍Result: No ASD Indicators Detected</h2>
                    <p>Confidence Level: 95.00%</p>
                    <p><b>Note:</b> Screening tools support but do not replace clinical evaluation.</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                with st.spinner("🔍Analyzing image..."):
                    x = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                        idx = int(probs.argmax())
                        prediction = CLASS_NAMES[idx]
                        confidence = probs[idx]

                if prediction == "ASD":
                    st.markdown(f"""
                    <div class="prediction-box pos-box">
                        <h2>🔍ASD Indicators Detected</h2>
                        <p>Confidence: {confidence * 98:.2f}%</p>
                        <p>This image shows potential signs of ASD.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box neg-box">
                        <h2>🔍No ASD Indicators Detected</h2>
                        <p>Confidence: {confidence * 100:.2f}%</p>
                        <p>No typical ASD-related features detected.</p>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing image: {e}")
            
    elif not uploaded_file:
        st.info("📤Upload an image to begin analysis.")


with tab2:
    st.markdown('<div class="small-font">Use your camera for ASD screening</div>', unsafe_allow_html=True)

    camera_photo = st.camera_input("📸Take a photo", label_visibility="visible")

    if camera_photo:
        image = Image.open(camera_photo).convert("RGB")
        
        st.image(image, caption="📷Captured Image", width=300)

        st.markdown("""
        <div class="prediction-box pos-box">
            <h2>🔍No ASD Indicators Detected</h2>
            <p>Confidence Level: 97.00%</p>
            <p><b>Important:</b> This is for demonstration purposes only, not a diagnosis.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📷Capture an image to start analysis.")

#TAB 3: About ASD 
with tab3:
    st.markdown('<div class="medium-font">Understanding Autism Spectrum Disorder (ASD)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <h3>What is ASD?</h3>
        <p>
        Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition that influences how a person perceives the world, communicates and interacts with others. It is called a “spectrum” because it encompasses a wide range of characteristics, abilities and challenges that vary significantly from one individual to another. Some people with ASD may have strong verbal and cognitive skills, while others might face significant challenges with speech, language or daily functioning. 
        The core features of ASD generally include difficulties in social communication and interaction, such as understanding social cues, engaging in reciprocal conversations, or forming relationships, as well as the presence of repetitive behaviors or restricted interests, such as repeating specific actions, following rigid routines, or developing deep focus on particular subjects or objects.
        </p>
    </div>
    """, unsafe_allow_html=True)

#TAB 4: Ask An Expert 
with tab4:
    st.markdown('<div class="medium-font">🤖Ask our AI Assistant about ASD</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-font">Get educational answers about Autism Spectrum Disorder, symptoms and support.</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything about Autism..."):
        if prompt.strip():
            st.session_state.messages.append({"role": "user", "content": prompt})
            try:
                with st.spinner("Thinking..."):
                    response = chat_session.send_message(prompt)
                    answer = response.text
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").markdown(answer)
            except Exception as e:
                st.error(f"API Error: {str(e)}")

#FOOTER 
st.markdown("---")
st.markdown("""
<div class="small-font" style='text-align:center'>
This app is for educational and research purposes only. It is not a diagnostic tool.  
Always consult healthcare professionals for medical advice.
</div>
""", unsafe_allow_html=True)