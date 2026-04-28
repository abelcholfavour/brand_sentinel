# BRAND SENTINEL

Real-time social media sentiment analysis system for brand crisis detection and management.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Technical Dictionary](#technical-dictionary)
- [Team](#team)

## links
**Tableau**

https://public.tableau.com/views/Brand_sentinel/Story1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

**App**

https://brandsentinel-sm72bfykdtgwhbba44hl6h.streamlit.app/

## Overview

Brand Sentinel is an intelligent sentiment analysis system designed to protect brand reputation by detecting negative sentiment in social media conversations in real-time. The system is particularly valuable during major tech events (like SXSW) where online reactions can quickly impact brand perception.

### The Problem

Social media managers face overwhelming data volumes:
- ~60% of posts are neutral noise
- ~6% contain critical issues (bugs, complaints)
- Manual monitoring risks missing important problems that can escalate

### The Solution

Brand Sentinel automates sentiment detection through a three-tier model architecture:
1. **Fast filtering** (Naive Bayes)
2. **Risk detection** (XGBoost)
3. **Context-aware decision making** (RNN/LSTM)

The system achieves **80% negative recall**, catching most brand risks before they become crises.

## Features

- **Real-time Sentiment Analysis**: Classifies posts as Positive, Negative, or Neutral
- **Multi-Model Ensemble**: Three complementary models working together
- **Crisis Detection**: Automatically flags high-risk content with confidence scores
- **Robust Text Processing**: 13-step data cleaning pipeline (the "Purity Pipeline")
- **Handles Imbalanced Data**: Uses SMOTE and class weighting to improve minority class detection
- **Production-Ready**: Exports trained model (.h5), tokenizer, and label encoder

## Dataset

### Sources
- **SXSW Tweets**: 9,093 records
- **Apple Sentiment Tweets**: ~3,900 records
- **Total**: 12,740 labeled records

### Distribution
The dataset exhibits class imbalance:
- Neutral: ~60%
- Positive: ~34%
- Negative: ~6% (the critical class we prioritize)

## Installation

### Prerequisites
- Python 3.11+
- TensorFlow 2.x
- NLTK data files

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/brand-sentinel.git
cd brand-sentinel

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('omw-1.4')"
```

### Required Libraries

```python
# Core
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0

# NLP
nltk>=3.7
imbalanced-learn>=0.10.0

# Deep Learning
tensorflow>=2.10.0

# ML Models
xgboost>=1.7.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
```

## Model Architecture

### The Three-Model Ensemble

| Model | Role | Strength |
|-------|------|----------|
| **Naive Bayes** | Fast Filter | Quick baseline classification |
| **XGBoost** | Risk Detector | Identifies risky keywords and patterns |
| **RNN (LSTM)** | Final Decision | Deep context understanding, sarcasm detection |

### RNN Architecture Details

```
Input Layer (Tokenized Text)
    ↓
Embedding Layer (100D word vectors)
    ↓
SpatialDropout1D (0.2)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dropout (0.5)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Output Layer (3 classes, Softmax)
```

**Key Features:**
- **Bidirectional LSTM**: Reads text forward and backward for context
- **Embedding Layer**: Converts words into 100D semantic space
- **SpatialDropout**: Prevents overfitting on specific patterns
- **Class Weighting**: Boosts focus on rare Negative class
- **Early Stopping**: Prevents overfitting during training

## Performance

### Model Comparison

| Model | Accuracy | Negative Recall | Primary Use Case |
|-------|----------|----------------|------------------|
| Naive Bayes | 67.24% | 59% | Fast initial filtering |
| XGBoost | 61.14% | 72% | Keyword-based risk detection |
| **Sentinel RNN** | **62.36%** | **80%** | **Final crisis detection** |

### Why Negative Recall Matters

For brand protection, **false negatives are more costly than false positives**:
- Missing a crisis = Brand damage
- Over-flagging = Manual review (acceptable trade-off)

The RNN's **80% negative recall** means it catches 4 out of 5 real brand risks.

### Confusion Matrix Insights

The RNN excels at:
- Detecting subtle negativity and sarcasm
- Understanding context (e.g., "great" used sarcastically)
- Handling emoji-heavy and informal language

## Technical Dictionary

### NLP & Preprocessing Concepts

- **Sentiment**: Emotion in a tweet (Positive / Negative / Neutral)
- **Tokenization**: Splitting text into words
- **Lemmatization**: Reducing words to base form (e.g., "running" → "run")
- **Stopwords**: Common unimportant words removed (e.g., "the", "a", "is")
- **Imbalance**: Uneven class distribution (many Neutral, few Negative)
- **SMOTE**: Creates synthetic samples for rare classes

### Deep Learning Concepts

- **Embedding Layer**: Converts words into numerical meaning space
- **LSTM**: Long Short-Term Memory - remembers context and word order
- **Bidirectional Layer**: Reads text forward and backward
- **SpatialDropout**: Reduces overfitting on specific patterns
- **Softmax**: Converts model outputs into probabilities
- **Categorical Crossentropy**: Loss function for multi-class classification
- **Adam Optimizer**: Adaptive learning rate optimization

### Key Functions

- **pad_sequences**: Makes all text inputs the same length
- **LabelEncoder**: Converts text labels (Positive/Negative/Neutral) into numbers (0/1/2)
- **compute_class_weight**: Boosts model focus on rare Negative class
- **EarlyStopping**: Stops training before overfitting
- **ReLU**: Activation function for learning complex patterns

## Model Export Files

The system exports three files (the "Three Siblings"):

1. **sentinel_rnn_model_v3.h5**: The trained neural network brain
2. **tokenizer_v3.pkl**: Vocabulary and text-to-number conversion
3. **label_encoder_v3.pkl**: Maps numeric predictions back to labels

All three are required for deployment.

## Team

**Group 7 Development Team:**

1. Patience Chepkosgei
2. Carolyne Githenduka
3. Augustine Magani
4. Marcus Kaula
5. Abel Aleu Chol Garang

## Development Process

The project followed a phased development approach:

- **Phase 0**: Project Setup & Problem Definition
- **Phase 1**: Data Integration (SXSW + Apple datasets merging strategy)
- **Phase 2**: Purity Pipeline (13-step data cleaning process)
- **Phase 3**: Neural Architecture (RNN / LSTM model design decisions)
- **Phase 4**: Model Evaluation & Testing (confusion matrix, recall analysis)

## Ethics & Privacy

- Personal information is removed from all tweets
- Only message content is analyzed
- Class balancing techniques protect minority classes from bias
- Low-quality and spam data is filtered out

## Use Cases

**Primary Users**: Social media managers, brand protection teams, crisis management professionals

**Applications**:
- Real-time event monitoring (SXSW, product launches, conferences)
- Early crisis detection and response
- Customer sentiment tracking
- Competitive intelligence
- Product feedback analysis

## Technical Stack

- **Deep Learning**: TensorFlow / Keras
- **ML Models**: Scikit-learn, XGBoost
- **NLP**: NLTK
- **Data Handling**: Pandas, NumPy
- **Class Balancing**: Imbalanced-learn (SMOTE)
- **Visualization**: Matplotlib, Seaborn

## Example Predictions

```
Input: "OMG my iphone is 🔥 but the battery is 💀... HELP!!"
Output: Negative (Risk: 99.8%) 🚩

Input: "Best app ever ❤️ #SXSW"
Output: Positive (Risk: 3.3%) ✅

Input: "Oh great, another update that breaks my phone. Simply wonderful. 🙄"
Output: Negative (Risk: 60.2%) 🚩 (Sarcasm detected)
```

## Future Enhancements

- Multi-language support
- Real-time streaming integration (Twitter API)
- Aspect-based sentiment analysis (battery, design, price)
- Trend detection and anomaly alerts
- Interactive dashboard for visualization
