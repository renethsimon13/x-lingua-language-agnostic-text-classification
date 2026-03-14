# 🌐 X-lingua — Language-Agnostic Text Classification

> Multilingual text classification using sentence embeddings — train on one language, evaluate on another.

`x-lingua` is a zero-shot cross-lingual text classifier that leverages the `quora-distilbert-multilingual` sentence transformer to produce language-agnostic embeddings, then trains a lightweight feedforward classifier on top. The key insight: train on English, test on Hindi (or any other supported language) — no translation needed.

---

## How It Works

1. **Embeddings** — Text is encoded using `quora-distilbert-multilingual` (via `sentence-transformers`), which maps sentences from any supported language into a shared multilingual embedding space.

2. **Classifier** — A simple single-layer feedforward network with dropout sits on top of the frozen embeddings and learns to map them to category labels.

3. **Cross-lingual Evaluation** — The model is trained on English text but evaluated on Hindi text. Since both languages share the same embedding space, the classifier generalizes across languages without any multilingual training data.

```
Text (any language)
      ↓
quora-distilbert-multilingual
      ↓
 Dense Embedding (768-dim)
      ↓
  Dropout → Linear → Softmax
      ↓
  Predicted Label
```

---

## Results

Trained on English, evaluated on Hindi:

```
              precision    recall  f1-score   support

           0       1.00      0.93      0.96       728
           1       0.45      0.93      0.61        46

    accuracy                           0.93       774
   macro avg       0.72      0.93      0.78       774
weighted avg       0.96      0.93      0.94       774
```

**93% accuracy** on cross-lingual evaluation with no translation or multilingual fine-tuning.

---

## Dataset

This project uses a multilingual CSV dataset with English, Hindi, German, and French text samples.

Expected format:

```
Message,Category
"Hello, how are you?",greeting
"Bonjour, comment ça va?",greeting
...
```

Update the path in the notebook:

```python
data_df = pd.read_csv("path/to/data-en-hi-de-fr.csv")
```

---

## Requirements

- Python 3.8+
- PyTorch
- sentence-transformers
- scikit-learn
- pandas

Install all dependencies:

```bash
pip install torch sentence-transformers scikit-learn pandas
```

---

## Usage

Run `multilingual_classification.ipynb` end-to-end to:

- Load and preprocess the multilingual dataset
- Encode text using the multilingual sentence transformer
- Train the classifier on English embeddings
- Evaluate cross-lingually on Hindi text
- Print a full classification report

---

## Project Structure

```
x-lingua/
│
├── multilingual_classification.ipynb   # Main notebook
├── README.md
└── .gitignore
```

---

## .gitignore

```gitignore
# Data
*.csv

# Jupyter
.ipynb_checkpoints/

# Python
__pycache__/
*.pyc

# Virtual environment
venv/
env/
```

---

## Notes

- The model uses frozen embeddings from `quora-distilbert-multilingual` — no fine-tuning of the transformer. Only the classification head is trained.
- Batch shuffling is handled by a custom `Batcher` class that randomizes indices each epoch.
- Swap the encoder for any other `sentence-transformers` multilingual model to experiment with different embedding spaces.
