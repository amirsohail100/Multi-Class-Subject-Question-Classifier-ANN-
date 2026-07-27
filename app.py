import streamlit as st
import numpy as np
import pickle
import time
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EduRoute AI - Subject Classifier",
    page_icon="📚",
    layout="centered"
)

# --- PREMIUM CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    /* Text Area Styling & Visibility Fix */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        font-size: 16px;
        background-color: #ffffff !important;
        color: #212529 !important; /* Saaf kale rang mein text dikhega */
    }
    .stTextArea textarea:focus {
        border-color: #4a00e0;
        box-shadow: 0 0 10px rgba(74, 0, 224, 0.15);
    }
    .main-title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1e1b4b;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #475569;
        text-align: center;
        margin-bottom: 30px;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title & Subtitle
st.markdown("<h1 class='main-title'>📚 EduRoute AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Intelligent Subject Classifier & Question Router powered by GRU Neural Network</p>", unsafe_allow_html=True)

# --- 1. MODEL & ASSETS LOADING ---
@st.cache_resource
def load_assets():
    try:
        # Agar aapne tensorflow use kiya hai (Aap apne saved files ke naam yahan check kar lena)
        from tensorflow.keras.models import load_model
        
        # ⚠️ Apne files ke sahi naam se inhe replace kar lena agar alag hain:
        model = load_model('model.h5') 
        tokenizer = joblib.load('tokenizer.pkl')  # Tokenizer ko joblib se load karna zyada efficient hai
        label_encoder = joblib.load('label_encoder.pkl')  # Label Encoder ko bhi joblib se load karna zyada efficient hai
            
        return model, tokenizer, label_encoder, True
    except Exception as e:
        # Fallback agar file load na ho paye toh interface crash na ho
        return None, None, None, False

model, tokenizer, label_encoder, is_loaded = load_assets()

# Accuracy & Engine Status Badge
col_b1, col_b2 = st.columns([3, 1])
with col_b1:
    if is_loaded:
        st.success("⚡ **Deep Learning Engine:** GRU Production Model Active.")
    else:
        st.warning("⚠️ **Model Files Not Found:** Place your model.h5 & pickle files in the same directory.")
with col_b2:
    st.metric(label="Model Accuracy", value="95.0%")

st.markdown("---")

# --- 2. USER INPUT AREA ---
st.markdown("### 📥 Enter or Paste your Question:")
user_question = st.text_area(
    "",
    height=140,
    placeholder="Example: What is the derivative of sin(x) with respect to x? / Define Newton's third law..."
)

# Basic text statistics
if user_question.strip():
    st.caption(f"🔤 Characters: **{len(user_question)}** | 📝 Words: **{len(user_question.split())}**")

st.markdown("<br>", unsafe_allow_html=True)

# --- 3. INFERENCE PIPELINE ---
if st.button("🚀 Route & Classify Question", type="primary", use_container_width=True):
    if not user_question.strip():
        st.error("⚠️ Kripya analyze karne ke liye pehle koi question type ya paste karein!")
    elif not is_loaded:
        st.error("🚫 Cannot classify: Model files missing. Please check backend assets.")
    else:
        with st.spinner("🧠 Analyzing linguistic features and routing sequence through GRU layers..."):
            time.sleep(0.8) # Premium feel coding delay
            
            # --- PREPROCESSING WORKFLOW (Wahi sequence jo aapne banaya tha) ---
            # 1. Lowercase
            cleaned_text = user_question.lower()
            
            # 2. Tokenize & Pad Sequence
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            # Tokenizer standard texts_to_sequences leta hai (Aapka max_len check kar lena, maine 100 default rakha hai)
            sequences = tokenizer.texts_to_sequences([cleaned_text])
            padded_seq = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
            
            # 3. Prediction
            prediction_probs = model.predict(padded_seq, verbose=0)[0]
            predicted_class_idx = np.argmax(prediction_probs)
            confidence_score = prediction_probs[predicted_class_idx] * 100
            
            # 4. Decode Label
            predicted_subject = label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # --- 4. DISPLAY RESULTS ---
            st.markdown("### 🎯 AI Routing Report")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Predicted Subject", value=f"📘 {predicted_subject.upper()}")
            with res_col2:
                st.metric(label="Routing Confidence", value=f"{confidence_score:.2f}%")
            
            # Interactive Progress Bar for confidence
            st.progress(float(prediction_probs[predicted_class_idx]))
            
            st.balloons()
            st.success(f"✅ Question ko successfully **{predicted_subject}** department mein route kar diya gaya hai.")

# Footer
st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 13px;'>EduRoute AI • GRU Text Classification Pipeline • Portfolio Project</p>", unsafe_allow_html=True)