import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. PAGE SETTINGS
st.set_page_config(page_title="Brand Sentinel", page_icon="🛡️")

# 2. SESSION STATE (For the Clear Button)
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

def clear_text():
    st.session_state.user_input = ""
    if 'input_widget' in st.session_state:
        st.session_state['input_widget'] = ""

# 3. CLEANER (Exactly as in your notebook)
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

# 4. LOAD RESOURCES
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('sentinel_rnn_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tok = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, tok, le

model, tokenizer, le = load_assets()

# 5. INTERFACE
st.title("🛡️ Brand Sentinel")
user_text = st.text_area("Enter Comment:", value=st.session_state.user_input, key="input_widget")

col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("🚀 Scan")
with col2:
    st.button("🗑️ Clear", on_click=clear_text)

# 6. THE PREDICTION ENGINE
if run_btn and user_text:
    # CLEAN
    cleaned = master_sentinel_cleaner(user_text)
    
    # TOKENIZE (Ensuring it matches Notebook exactly)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=80) 
    
    # PREDICT
    probs = model.predict(padded, verbose=0)
    best_idx = np.argmax(probs, axis=1)[0]
    verdict = le.classes_[best_idx]
    
    # Extract specific Negative Score (Assuming Negative is Index 0 like your notebook code)
    neg_score = probs[0][0] 

    # 7. DISPLAY LOGIC (Mirroring your master_sentinel_test)
    st.divider()
    
    # If the score is high OR the verdict is Negative, trigger the alert
    if neg_score > 0.35 or verdict == "Negative":
        st.error(f"### 🚩 ACTION: Brand Defense Suggested")
        st.write(f"**Verdict:** {verdict}")
        st.write(f"**Negative Risk Score:** {neg_score:.1%}")
        st.progress(float(neg_score))
    else:
        st.success(f"### ✅ STATUS: No immediate crisis detected")
        st.write(f"**Verdict:** {verdict}")
        st.write(f"**Negative Risk Score:** {neg_score:.1%}")
        if verdict == "Positive":
            st.balloons()
