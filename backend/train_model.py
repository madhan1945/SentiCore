"""
train_model.py
--------------
Downloads the IMDb dataset, cleans text, tokenizes, pads sequences,
trains an LSTM-based sentiment analysis model, and saves:
  - model.h5
  - tokenizer.pkl
  - model_accuracy.txt

Run once before starting app.py:
    python train_model.py
"""

import os
import re
import pickle
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import nltk
from nltk.corpus import stopwords

import tensorflow as tf
from keras.datasets import imdb as keras_imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Hyperparameters
VOCAB_SIZE = 20000
MAX_LEN = 200
EMBED_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 128
EPOCHS = 10
TEST_SPLIT = 0.2

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SAVE_DIR, "model.h5")
TOK_PATH = os.path.join(SAVE_DIR, "tokenizer.pkl")
ACC_PATH = os.path.join(SAVE_DIR, "model_accuracy.txt")


print("Downloading NLTK resources...")
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


print("Loading IMDb dataset from Keras...")
(x_train_raw, y_train), (x_test_raw, y_test) = keras_imdb.load_data(num_words=VOCAB_SIZE)

word_index = keras_imdb.get_word_index()
index_word = {v + 3: k for k, v in word_index.items()}
index_word.update({0: "<PAD>", 1: "<START>", 2: "<UNK>", 3: "<UNUSED>"})


def decode_review(seq):
    return " ".join(index_word.get(i, "?") for i in seq)


print("Cleaning reviews...")
x_train_text = [clean_text(decode_review(seq)) for seq in x_train_raw]
x_test_text = [clean_text(decode_review(seq)) for seq in x_test_raw]


print("Tokenizing text...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train_text)

x_train_seq = tokenizer.texts_to_sequences(x_train_text)
x_test_seq = tokenizer.texts_to_sequences(x_test_text)

x_train = pad_sequences(x_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
x_test = pad_sequences(x_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

y_train = np.array(y_train, dtype="float32")
y_test = np.array(y_test, dtype="float32")

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)


print("Building model...")
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(LSTM_UNITS, dropout=0.2)),
    Dense(64, activation="relu"),
    Dropout(0.4),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
]


print("Training model...")
history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=TEST_SPLIT,
    callbacks=callbacks,
    verbose=1
)


print("Evaluating model...")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print("Test Accuracy:", accuracy * 100)
print("Test Loss:", loss)


model.save(MODEL_PATH)
print("Model saved:", MODEL_PATH)

with open(TOK_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved:", TOK_PATH)

with open(ACC_PATH, "w") as f:
    f.write(f"{accuracy * 100:.2f}")

print("Accuracy saved:", ACC_PATH)

print("\nTraining complete! Run: python app.py\n")