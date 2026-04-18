# Dataset Notes

The IMDb dataset is **automatically downloaded** by `train_model.py` via the
`tensorflow.keras.datasets.imdb` module.

- 25,000 training reviews (balanced: 12,500 positive, 12,500 negative)
- 25,000 test reviews (same balance)
- Vocabulary: top 20,000 words

No manual download is required.
