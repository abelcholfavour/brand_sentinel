import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Brand Sentinel | Crisis Monitor",
    page_icon="🛡️",
    layout="centered"
)

# --- 2. SESSION STATE FOR CLEAR BUTTON ---
# This ensures the text box can be wiped clean
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

def clear_text_callback():
    st.session_state.user_input = ""
    if 'input_widget' in st.session_state:
        st.session_state['input_widget'] = ""

# --- 3. CLEANING PIPELINE ---
def master_sentinel_cleaner(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    emoji_map = {
        '❤️': ' love ', '😍': ' love ', '🙂': ' happy ', '😊': ' happy ',
        '🙁': ' sad ', '😢': ' sad ', '😡': ' angry ', '🙄':' mad ','🔥': ' awesome ',
        '🔋': ' battery ', '📱': ' phone ', '💀': ' dead ', '👎': ' bad ', '👍': ' good '
    }
    for emoji, word in emoji_map.items():
        text = text.replace(emoji, word)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r'\brt\b', '', text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# --- 4. RESOURCE LOADING ---
@st.cache_resource
def load_sentinel_resources():
    # Loading the model and pre-trained assets
    model = tf.keras.models.load_model('sentinel_rnn_model.h5') 
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pkl', 'rb') as handle:
        le = pickle.load(handle)
    return model, tokenizer, le

try:
    sentinel_rnn, tokenizer, le = load_sentinel_resources()
    max_sequence_len = 80
except Exception as e:
    st.error("⚠️ Model files missing! Ensure .h5 and .pkl files are in GitHub.")
    st.stop()

# --- 5. INTERFACE DESIGN ---
# Logo logic
try:
    st.image("logo.png", width=100) 
except:
    st.image("https://img.icons8.com/fluency/96/shield-with-crown.png", width=80)

st.title("Brand Sentinel")
st.markdown("### *Real-time Crisis Detection & Sentiment Analysis*")
st.caption(f"Engine: TensorFlow {tf.__version__} | RNN Architecture")
st.info("Monitoring brand reputation for high-traffic events like SXSW.")

st.divider()

# --- 6. INPUT AREA ---
user_text = st.text_area(
    "✍️ Enter Customer Comment/Tweet:", 
    value=st.session_state.user_input,
    key="input_widget",
    placeholder="Type a comment to test sentiment..."
)

# Buttons for interaction
col1, col2 = st.columns([1,1])
with col1:
    btn_run = st.button("🚀 Run Sentinel Scan", use_container_width=True)
with col2:
    btn_clear = st.button("🗑️ Clear & Reset", on_click=clear_text_callback, use_container_width=True)

# --- 7. PREDICTION & RESULTS ---
if btn_run:
    if user_text.strip():
        # Update session state to keep text visible during this specific run
        st.session_state.user_input = user_text
        
        # Clean and Tokenize
        cleaned = master_sentinel_cleaner(user_text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_sequence_len)
        
        # Get raw probabilities from model
        probs = sentinel_rnn.predict(padded, verbose=0)[0]
        
        # Find the highest probability and its corresponding label
        best_index = np.argmax(probs)
        verdict = le.classes_[best_index]
        confidence = probs[best_index]

        st.subheader("Analysis Results")
        
        # Logic is now tied to the 'verdict' word (Positive/Negative/Neutral)
        if verdict == "Negative":
            st.error(f"### 🚨 BRAND ALERT: {verdict}")
            st.write(f"**Confidence Level:** {confidence:.1%}")
            st.progress(float(confidence))
            st.warning("Action Required: Direct to Crisis Response Team immediately.")
            
        elif verdict == "Positive":
            st.success(f"### ✅ CLEAR: {verdict}")
            st.write(f"**Confidence Level:** {confidence:.1%}")
            st.progress(float(confidence))
            st.balloons()
            
        else: # Neutral
            st.info(f"### ℹ️ NEUTRAL: {verdict}")
            st.write(f"**Confidence Level:** {confidence:.1%}")
            st.progress(float(confidence))
            st.write("Recommendation: Monitor for escalation.")

    else:
        st.warning("Please enter a comment to analyze.")

# --- 8. FOOTER ---
st.divider()
st.caption("Developed by the Brand Sentinel Team | 81% Recall Target Achieved")
