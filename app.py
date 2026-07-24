import random
import re
from typing import List, Tuple

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# NLTK imports & data downloads (idempotent & safe)
# ---------------------------------------------------------------------------
import nltk

for resource in ["wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()


def sanitize_text(text: str) -> str:
    """Ensure string is safe from UTF-16 surrogate codepoints that break UTF-8 encoders."""
    if not text or not isinstance(text, str):
        return text or ""
    try:
        text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    except Exception:
        pass
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


def clean_text(text: str) -> str:
    """Preprocess text for conversational NLP:
    1. Lowercase & strip extra spaces
    2. Retain all conversational words (no aggressive stop-word removal!)
    3. Remove punctuation / special characters
    4. Lemmatize tokens
    """
    if not text:
        return ""
    text = sanitize_text(text)
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 0]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 1. LOAD DATASET (Huge Synthetic Dataset with Inline Fallback)
# ---------------------------------------------------------------------------
try:
    from intents_huge import INTENTS
except ImportError:
    INTENTS = {
        "greeting": {
            "patterns": ["hi", "hello", "hey", "good morning", "sup", "what's up"],
            "responses": ["Hey there! 👋 Welcome! How can I help you today?"],
        },
        "how_are_you": {
            "patterns": ["how are you", "how's it going", "how are you doing"],
            "responses": ["I'm doing great, thank you for asking! 😊 How are you?"],
        },
        "identity": {
            "patterns": ["who are you", "what is your name", "what are you"],
            "responses": ["I'm Azale! 🤖 An intent-classification AI chatbot."],
        },
    }

# Build flat training arrays
INTENT_LABELS: List[str] = []
TRAINING_PATTERNS: List[str] = []
for intent_name, intent_data in INTENTS.items():
    for pattern in intent_data["patterns"]:
        TRAINING_PATTERNS.append(pattern)
        INTENT_LABELS.append(intent_name)

# ---------------------------------------------------------------------------
# 2.  ML PIPELINE (cached)
# ---------------------------------------------------------------------------
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@st.cache_resource(show_spinner="Training conversational intent engine on huge dataset...")
def train_pipeline() -> Pipeline:
    """Build and train TF-IDF (1-3 grams) + Calibrated LinearSVC pipeline."""
    cleaned_patterns = [clean_text(p) for p in TRAINING_PATTERNS]

    pipeline = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "classifier",
                CalibratedClassifierCV(
                    LinearSVC(random_state=42, dual="auto"), cv=3
                ),
            ),
        ]
    )
    pipeline.fit(cleaned_patterns, INTENT_LABELS)
    return pipeline


pipeline = train_pipeline()

# ---------------------------------------------------------------------------
# 3. PREDICTION ENGINE
# ---------------------------------------------------------------------------


def predict_intent(message: str, threshold: float = 0.25) -> Tuple[str, float, str]:
    """Predict intent for user message and return (intent, confidence, response)."""
    cleaned = clean_text(message)

    if not cleaned:
        fallback_responses = [
            "I didn't catch any words there! Type a question, greeting, or topic to start chatting. 😊",
            "Hmm, try typing a full question or greeting like 'how are you' or 'what is Python'!",
        ]
        return ("unknown", 0.0, random.choice(fallback_responses))

    probs: np.ndarray = pipeline.predict_proba([cleaned])[0]
    confidence = float(np.max(probs))
    predicted_idx = int(np.argmax(probs))
    predicted_intent = pipeline.classes_[predicted_idx]

    if confidence < threshold:
        fallback_responses = [
            "That's an interesting question! Ask me about Python, Machine Learning, Data Science, Docker, Math, or general questions! 💭",
            "I'm not completely sure about that specific phrase yet. Try asking 'what is Machine Learning', 'tell me a joke', or 'how are you'!",
            "Could you rephrase that slightly? You can ask about code, algorithms, tech, science, or general topics!",
        ]
        return ("unknown", confidence, random.choice(fallback_responses))

    response = sanitize_text(random.choice(INTENTS[predicted_intent]["responses"]))
    return (predicted_intent, confidence, response)


# ---------------------------------------------------------------------------
# 4. STREAMLIT UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Azale Conversational AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.25

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1rem;">
            <span style="font-size: 3rem;">🤖</span>
            <h2 style="margin: 0;">Azale Chatbot</h2>
            <p style="color: #888; font-size: 0.9rem;">Conversational ML & NLP</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### 📊 Engine Stats")
    col1, col2 = st.columns(2)
    col1.metric("Intents", len(INTENTS))
    col2.metric("Patterns", len(TRAINING_PATTERNS))

    st.metric(
        "Classifier Engine",
        "Calibrated LinearSVC",
        delta="TF-IDF (1-3 grams)",
    )

    st.divider()

    st.markdown("### ⚙️ Settings")
    confidence_val = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.8,
        value=st.session_state.confidence_threshold,
        step=0.05,
        help="Minimum confidence required for intent matching. Lower value allows flexible daily conversation.",
    )
    st.session_state.confidence_threshold = confidence_val

    if st.button("🧹 Clear Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #666; text-align: center;">
            Conversational ML + NLP Engine<br>
            Streamlit · scikit-learn · NLTK
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Main Chat Area ───────────────────────────────────────────────

st.title("💬 Azale — Conversational AI")
st.caption(
    f"A conversational chatbot trained on **{len(INTENTS)} intents** and **{len(TRAINING_PATTERNS)} patterns**. "
    "Ask about programming, ML, AI, cloud, math, science, databases, jokes, or general small talk!"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": sanitize_text(
                "👋 Hey! I'm **Azale**, your conversational companion. "
                "Ask me *how are you*, *what is Python*, *what is Machine Learning*, or *tell me a joke*!"
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        safe_content = sanitize_text(msg["content"])
        if msg["role"] == "assistant" and "intent" in msg and msg["intent"] != "unknown":
            st.markdown(safe_content)
            confidence_pct = msg["confidence"] * 100
            st.markdown(
                f"<span style='font-size:0.75rem;color:#888;'>🎯 "
                f"Topic: <b>{msg['intent'].replace('_', ' ').title()}</b> · {confidence_pct:.1f}% confidence</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(safe_content)

if prompt := st.chat_input("Talk to Azale..."):
    safe_prompt = sanitize_text(prompt)
    st.session_state.messages.append({"role": "user", "content": safe_prompt})
    with st.chat_message("user"):
        st.markdown(safe_prompt)

    intent, confidence, response = predict_intent(safe_prompt, threshold=st.session_state.confidence_threshold)

    safe_response = sanitize_text(response)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": safe_response,
            "intent": intent,
            "confidence": confidence,
        }
    )
    with st.chat_message("assistant"):
        st.markdown(safe_response)
        if intent != "unknown":
            confidence_pct = confidence * 100
            st.markdown(
                f"<span style='font-size:0.75rem;color:#888;'>🎯 "
                f"Topic: <b>{intent.replace('_', ' ').title()}</b> · {confidence_pct:.1f}% confidence</span>",
                unsafe_allow_html=True,
            )
