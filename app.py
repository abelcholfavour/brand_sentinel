import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# PAGE SETTINGS
st.set_page_config(
    page_title="Brand Sentinel | Crisis Monitor",
    page_icon="🛡️",
    layout="centered"
)

# CLEANING PIPELINE
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

# RESOURCE LOADING
@st.cache_resource
def load_sentinel_resources():
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

# INTERFACE DESIGN

try:
    st.image("logo.png", width=100) 
except:
    st.image("https://img.icons8.com/fluency/96/shield-with-crown.png", width=80)

st.title("Brand Sentinel")
st.markdown("### *Real-time Crisis Detection & Sentiment Analysis*")
st.info("Tasked to monitor brand reputation during high-traffic events like SXSW.")

st.divider()

# INPUT AREA
user_text = st.text_area("✍️ Enter Customer Comment/Tweet:", placeholder="Type here to test sentiment...")

if st.button("Run Sentinel Scan"):
    if user_text.strip():
        # Clean and Tokenize
        cleaned_text = master_sentinel_cleaner(user_text)
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=max_sequence_len)
        
        # Predict
        probs = sentinel_rnn.predict(padded, verbose=0)[0]
        best_index = np.argmax(probs)
        verdict = le.classes_[best_index]
        confidence = probs[best_index]

        # Display Results
        st.subheader("Analysis Results")
        
        if verdict == "Negative":
            st.error(f"### 🚨 BRAND ALERT: {verdict}")
            st.progress(float(confidence))
            st.write(f"**Confidence:** {confidence:.1%}")
            st.warning("Action Required: Direct to Crisis Response Team immediately.")
            
        elif verdict == "Neutral":
            st.info(f"### ℹ️ NEUTRAL: {verdict}")
            st.progress(float(confidence))
            st.write(f"**Confidence:** {confidence:.1%}")
            
        else: 
            st.success(f"### ✅ CLEAR: {verdict}")
            st.progress(float(confidence))
            st.write(f"**Confidence:** {confidence:.1%}")
            st.balloons()

    else:
        st.warning("Please enter a comment first!")

# FOOTER
st.divider()
col1, col2 = st.columns([2,1])

with col1:
    st.caption("Developed by the Brand Sentinel Team | Bi-RNN Architecture | 81% Recall Target Achieved")

with col2:
    if st.button("🗑️ Clear & Reset"):
        st.rerun()
