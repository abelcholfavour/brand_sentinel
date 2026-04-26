import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# page settings
st.set_page_config(
    page_title="Brand Sentinel | Crisis Monitor",
    page_icon="🛡️",
    layout="centered"
)

# cleaning pipeline
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

# Model and  Tokenizer
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
except:
    st.error("⚠️ Model files not found! Please ensure 'sentinel_rnn_model.h5', 'tokenizer.pkl', and 'label_encoder.pkl' are in the GitHub repo.")
    st.stop()

# Interface Design
st.image("https://img.icons8.com/fluency/96/shield-with-crown.png", width=80)
st.title("Brand Sentinel")
st.markdown("### *Real-time Crisis Detection & Sentiment Analysis*")
st.info("Tasked to monitor brand reputation during high-traffic events like SXSW.")

st.divider()

# Input area
user_text = st.text_area("✍️ Enter Customer Comment/Tweet:", placeholder="Type here to test sentiment...")

if st.button("Run Sentinel Scan"):
    if user_text:
        # Cleans
        cleaned_text = master_sentinel_cleaner(user_text)
        
        # Transforms
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=max_sequence_len)
        
        # Predicts
        probs = sentinel_rnn.predict(padded, verbose=0)[0]
        best_index = np.argmax(probs)
        verdict = le.classes_[best_index]
        neg_prob = probs[0] 

        # Display Results
        st.subheader("Analysis Results")
        
        # Determine Alert Level
        if neg_prob > 0.35:
            st.error(f"### 🚨 BRAND ALERT: {verdict}")
            st.progress(float(neg_prob))
            st.write(f"**Negative Risk Level:** {neg_prob:.1%}")
            st.warning("Action Required: Direct this to the Crisis Response Team immediately.")
        elif neg_prob > 0.20:
            st.warning(f"### ⚠️ WATCH: {verdict}")
            st.progress(float(neg_prob))
            st.write(f"**Negative Risk Level:** {neg_prob:.1%}")
            st.info("Recommendation: Monitor this thread for potential sarcasm or escalation.")
        else:
            st.success(f"### ✅ CLEAR: {verdict}")
            st.write(f"**Negative Risk Level:** {neg_prob:.1%}")
            st.balloons()

    else:
        st.sidebar.warning("Please enter a comment to analyze.")

# --- 5. FOOTER ---
st.divider()
st.caption("Developed by the Brand Sentinel Team | Bidirectional RNN Architecture | 81% Recall Target Achieved")
