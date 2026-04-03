# IMDB Sentiment Analysis — PyTorch

A binary sentiment classifier trained on the IMDB movie review dataset, built from scratch in PyTorch. Classifies reviews as **positive** or **negative** using a bag-of-words architecture with learned word embeddings.

---

## Pipeline

```
raw text → tokenization → word IDs → embeddings → mean pool → ReLU → dropout → sigmoid → positive/negative
```

---

## Architecture

```python
SentimentModel(
  embeddings: nn.Embedding(74878, 64)   # learned word vectors
  fc1:        nn.Linear(64, 32)         # hidden layer
  dropout:    nn.Dropout(p=0.3)         # regularization
  fc2:        nn.Linear(32, 1)          # binary output
  sigmoid:    squash to [0, 1]
)
```

### Key Design Decisions

- **Vocabulary**: 74,878 tokens built from the training split only (to prevent data leakage). Includes `<pad>` and `<unk>` special tokens.
- **Encoding**: Reviews are tokenized with `BasicTokenizer`, truncated/padded to 256 tokens.
- **Mean Pooling**: Word embeddings are averaged across the sequence dimension (`dim=1`) to produce a single fixed-size review vector — the "bag-of-words" step.
- **Hidden Layer + ReLU**: Adds non-linearity, allowing the model to learn more complex feature combinations than a single linear layer.
- **Dropout**: Applied after ReLU to reduce overfitting.
- **Sigmoid output**: Produces a probability; threshold at 0.5 for binary prediction.

---

## Training

| Setting | Value |
|---|---|
| Loss function | `BCELoss` |
| Optimizer | `Adam` |
| Learning rate | `5e-4` |
| Batch size | `32` |
| Epochs | `10` |
| Max sequence length | `256` |
| Embedding dim | `64` |

---

## Results

| Epoch | Train Loss | Test Accuracy | Test Loss |
|---|---|---|---|
| 1 | 0.564 | 83.66% | 0.401 |
| 2 | 0.316 | 85.34% | 0.337 |
| 3 | 0.236 | 87.84% | 0.299 |
| 4 | 0.183 | **88.34%** | **0.305** |
| 5 | 0.148 | 87.91% | 0.329 |
| 10 | 0.049 | 85.01% | 0.506 |

**Best test accuracy: 88.34% (epoch 4)**

The model shows classic overfitting after epoch 4-5 — training loss continues to drop while test loss rises. Early stopping at epoch 4 gives the best generalisation.

---

## Limitations

This is a bag-of-words model, meaning **word order is ignored**. Averaging word embeddings loses sequential context — for example, "not good" and "good" produce similar representations because "good" dominates the mean. This is the primary ceiling on accuracy.

To improve further, the next step would be an Transformer-based architecture that models word order and context.

---

## Usage

```python
review = "This film was an absolute masterpiece."
tensor = encode_review(review, vocab)
tensor = torch.unsqueeze(tensor, 0).to(device)

with torch.no_grad():
    pred = model(tensor).item()
    sentiment = "Positive" if pred >= 0.5 else "Negative"
    print(f"Sentiment: {sentiment} ({pred:.2f})")
```
