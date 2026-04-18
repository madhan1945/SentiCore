"""
app.py
------
Flask REST API for Deep Learning Sentiment Analysis.

Endpoints:
  GET  /            -> serves frontend/index.html
  POST /predict     -> { "text": "..." }  ->  { "sentiment": "...", "confidence": 0.xx }
  GET  /accuracy    -> { "accuracy": "xx.xx" }
  GET  /health      -> { "status": "ok" }

Start:
    python app.py
"""

import os
import re
import pickle
import logging

import nltk
from nltk.corpus import stopwords
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Suppress TF info/warning logs ─────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Absolute paths — works no matter where you cd from ────────
BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))        # .../backend
PROJECT_DIR  = os.path.dirname(BACKEND_DIR)                      # .../sentiment-analysis-dl
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")             # .../frontend

MODEL_PATH   = os.path.join(BACKEND_DIR, "model.h5")
TOK_PATH     = os.path.join(BACKEND_DIR, "tokenizer.pkl")
ACC_PATH     = os.path.join(BACKEND_DIR, "model_accuracy.txt")

MAX_LEN = 200   # must match train_model.py

# ── NLTK stopwords ────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Flask — static_folder points to frontend/, served at "/" ──
app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=""          # serve static files at root URL
)
CORS(app)

# ── Load model & tokenizer ────────────────────────────────────
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] model.h5 not found at:\n  {MODEL_PATH}\n"
            "Run  python train_model.py  first.\n"
        )
    if not os.path.exists(TOK_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] tokenizer.pkl not found at:\n  {TOK_PATH}\n"
            "Run  python train_model.py  first.\n"
        )

    log.info(f"Frontend dir : {FRONTEND_DIR}")
    log.info("Loading model ...")
    mdl = load_model(MODEL_PATH)

    log.info("Loading tokenizer ...")
    with open(TOK_PATH, "rb") as f:
        tok = pickle.load(f)

    acc = "N/A"
    if os.path.exists(ACC_PATH):
        with open(ACC_PATH) as f:
            acc = f.read().strip()

    log.info(f"Model test accuracy: {acc}%")
    return mdl, tok, acc


model, tokenizer, MODEL_ACCURACY = load_artifacts()

# ── Text cleaning ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>",    " ", text)   # strip HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters
    text = re.sub(r"\s+",      " ", text).strip()
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/accuracy", methods=["GET"])
def accuracy():
    return jsonify({"accuracy": MODEL_ACCURACY})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain a 'text' field."}), 400

    raw_text = str(data["text"]).strip()
    if not raw_text:
        return jsonify({"error": "Text field must not be empty."}), 400

    # Preprocess
    cleaned  = clean_text(raw_text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")

    # Predict
    prob = float(model.predict(padded, verbose=0)[0][0])

    sentiment  = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else 1.0 - prob

    log.info(f"'{raw_text[:60]}' -> {sentiment} ({confidence:.2%})")

    return jsonify({
        "sentiment":  sentiment,
        "confidence": round(confidence, 4),
        "raw_score":  round(prob, 4),
    })


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  SentiCore — Sentiment Analysis API")
    print(f"  Frontend : {FRONTEND_DIR}")
    print("  URL      : http://127.0.0.1:5002")
    print("=" * 55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5002)
