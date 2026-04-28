import streamlit as st
import pandas as pd
import numpy as np
import re
import html
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="Brand Sentinel | Crisis Monitor",
    page_icon="🛡️",
    layout="centered"
)

@st.cache_resource
def setup_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

@st.cache_resource
def load_sentinel_resources():
    model = tf.keras.models.load_model('sentinel_rnn_model_v3.h5') 
    with open('tokenizer_v3.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder_v3.pkl', 'rb') as handle:
        le = pickle.load(handle)
    return model, tokenizer, le

stop_words, lemmatizer = setup_resources()

try:
    sentinel_rnn, tokenizer, le = load_sentinel_resources()
    max_sequence_len = 80 
except Exception as e:
    st.error("⚠️ v3 Model files missing! Ensure .h5 and .pkl (v3) files are in your GitHub.")
    st.stop()

def universal_purity_pipeline(text):
    text = html.unescape(str(text))
    text = re.sub(r'^RT\s+@\w+:\s*', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.lower()
    
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'s", " is", text)
    
    slang_map = {"u ": "you ", " r ": " are ", "gr8": "great", "lol": "laughing", "2": "to"}
    for slang, formal in slang_map.items():
        text = text.replace(slang, formal)
    
    emoji_map = {
        '😡': ' angry ', '😊': ' happy ', '😒': ' annoyed ', '📱': ' phone ', 
        '🔋': ' battery ', '❤️': ' love ', '😍': ' love ', '🔥': ' awesome ', 
        '💀': ' dead ', '👎': ' bad ', '👍': ' good '
    }
    for emo, word in emoji_map.items():
        text = text.replace(emo, word)
    
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words).strip()

# This creates a centered image
col_1, col_2, col_3 = st.columns([1, 1, 1])
with col_2:
    try:
        st.image("logo.png", width=120) 
    except:
        st.image("https://img.icons8.com/fluency/96/shield-with-crown.png", width=100)

st.markdown("<h1 style='text-align: center;'>Brand Sentinel</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>Real-time Crisis Detection & Sentiment Analysis</p>", unsafe_allow_html=True)
st.info("Tasked to monitor brand reputation during high-traffic events like SXSW.")

st.divider()


if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

user_text = st.text_area(
    "✍️ Enter Customer Comment/Tweet:", 
    placeholder="Type here to test sentiment...",
    height=150,
    key=f"user_input_{st.session_state.reset_counter}"
)

# Buttons
col_btn1, col_btn2 = st.columns([1, 4])

with col_btn1:
    run_scan = st.button("Scan Comment")

with col_btn2:
    if st.button("Clear Comment"):

        st.session_state.reset_counter += 1
        st.rerun()

if run_scan:
    if user_text.strip():
        cleaned_text = universal_purity_pipeline(user_text)
        
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=max_sequence_len)
        
        probs = sentinel_rnn.predict(padded, verbose=0)[0]
        classes = list(le.classes_)

        neg_idx = classes.index('Negative')
        neg_score = probs[neg_idx]
        
        best_idx = np.argmax(probs)
        verdict = classes[best_idx]
st.write(f"DEBUG - Raw Scores: {dict(zip(classes, probs))}")
        confidence = probs[best_idx]

        st.divider()
        st.subheader("Analysis Results")
        
        if neg_score > 0.60:
            st.error(f"### 🚨 BRAND ALERT: {verdict}")
            st.progress(float(neg_score))
            st.write(f"**Negative Risk Score:** {neg_score:.1%}")
            st.warning("Action Required: Direct to Crisis Response Team immediately.")
        elif verdict == "Neutral":
            st.info(f"### ℹ️ NEUTRAL: {verdict}")
            st.progress(float(confidence))
            st.write(f"**Confidence:** {confidence:.1%}")
        else: 
            st.success(f"### ✅ CLEAR: {verdict}")
            st.progress(float(confidence))
            st.write(f"**Confidence:** {confidence:.1%}")
            if verdict == "Positive":
                st.balloons()
    else:
        st.warning("Please enter a comment first!")

st.divider()
st.caption("Developed by the Brand Sentinel Group Seven | RNN Model")
