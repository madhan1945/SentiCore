# 🧠 SentiCore — Deep Learning Sentiment Analysis Web App

> Bidirectional LSTM neural network trained on 50,000 IMDb movie reviews.  
> Full-stack: **Python Flask** backend + **Bootstrap 5** frontend.

---

## 📌 Project Overview

**SentiCore** is a university-level NLP deep learning project that:

- Trains a **Bidirectional LSTM** model on the Keras IMDb dataset (50 K reviews).
- Exposes a **REST API** via Flask to classify text as **Positive** or **Negative**.
- Serves a **modern dark-themed web UI** with confidence bars, examples, and pipeline visualisation.

### Model Architecture

```
Input Sequence (len=200)
        │
  Embedding Layer (vocab=20 000, dim=128)
        │
  SpatialDropout1D (0.3)
        │
  Bidirectional LSTM (64 units, dropout=0.2)
        │
  Dense (64, relu) → Dropout (0.4)
        │
  Dense (1, sigmoid)   ← Positive if ≥ 0.5
```

---

## 🗂 Project Structure

```
sentiment-analysis-dl/
│
├── backend/
│   ├── app.py              ← Flask REST API
│   ├── train_model.py      ← LSTM training script
│   ├── model.h5            ← Saved model (generated after training)
│   ├── tokenizer.pkl       ← Saved tokenizer (generated after training)
│   ├── model_accuracy.txt  ← Test accuracy (generated after training)
│   └── requirements.txt
│
├── frontend/
│   ├── index.html          ← Main UI
│   ├── style.css           ← Dark industrial theme
│   └── script.js           ← API calls + rendering
│
├── dataset/                ← IMDb auto-downloaded via Keras
├── README.md
└── run_project.bat         ← Windows one-click launcher
```

---

## ⚙️ Installation

### Prerequisites
- Python **3.10 – 3.11** (recommended for TensorFlow compatibility)
- Node.js not required — pure Python backend

### 1 — Clone / unzip the project

```powershell
cd C:\Projects
# unzip or place the folder here
cd sentiment-analysis-dl
```

### 2 — Create & activate virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3 — Install dependencies

```powershell
cd backend
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```powershell
# Make sure you are in the backend/ folder with venv active
python train_model.py
```

This will:
1. Download the IMDb dataset via Keras (~17 MB).
2. Clean, tokenise, and pad all 25 000 training reviews.
3. Train the Bidirectional LSTM (EarlyStopping enabled — typically 4–7 epochs).
4. Save `model.h5`, `tokenizer.pkl`, and `model_accuracy.txt` in `backend/`.

**Expected accuracy: ~87–90 %** on the test set.

---

## 🚀 Start the Server

```powershell
# Still in backend/ with venv active
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## 🔌 REST API

### `POST /predict`

**Request**
```json
{
  "text": "I love this movie. The acting was superb!"
}
```

**Response**
```json
{
  "sentiment":  "Positive",
  "confidence": 0.9712,
  "raw_score":  0.9712
}
```

### `GET /accuracy`

```json
{ "accuracy": "88.54" }
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## 📸 Screenshots

After running the app, open `http://127.0.0.1:5000` to see:

- **Hero section** with model stats
- **Text input** with live character counter
- **Result card** with animated confidence bar
- **Example chips** (click any to auto-analyse)
- **Pipeline diagram** showing the 5-step process

---

## 🛠️ Technologies

| Layer      | Technology                          |
|------------|-------------------------------------|
| Model      | TensorFlow / Keras 2.16             |
| API        | Flask 3 + flask-cors                |
| NLP        | NLTK (stopword removal)             |
| Dataset    | Keras IMDb (25 K train, 25 K test)  |
| Frontend   | HTML5, CSS3, Bootstrap 5, Vanilla JS|
| Fonts      | Syne + DM Sans (Google Fonts)       |

---

## 👨‍💻 Author

**Madhan** — [@madhan1945](https://github.com/madhan1945)
