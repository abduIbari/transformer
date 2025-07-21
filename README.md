# 🧠 English-to-German Neural Machine Translation

This project implements a **Transformer-based sequence-to-sequence model** for translating sentences from **English to German** using PyTorch. The model was trained on a subset of the **Europarl v7 parallel corpus** and evaluated using **BLEU score** on a held-out validation set.

---

## 📂 Dataset

We use the **Europarl v7** English-German parallel dataset:  
📎 https://www.statmt.org/europarl/v7/de-en.tgz

- 2,000 sentence pairs were extracted for training and validation.
- Sentences were tokenized and padded to a maximum sequence length of 60 tokens.
- Special tokens `<pad>`, `<unk>`, `<sos>`, and `<eos>` were added to handle padding and sequence boundaries.

---

## 🔧 Model Architecture

The architecture is a custom implementation of the **original Transformer** as proposed in Vaswani et al. (2017), with the following configuration:

| Component              | Value              |
|------------------------|--------------------|
| Embedding Dimension    | 256                |
| Encoder/Decoder Layers | 4                  |
| Attention Heads        | 8                  |
| Feedforward Dimension  | 1024               |
| Dropout Rate           | 0.3                |
| Max Sequence Length    | 60 (src & tgt)     |
| Optimizer              | Adam               |
| Learning Rate          | 1e-4               |
| Weight Decay           | 1e-3               |
| Batch Size             | 16                 |
| Label Smoothing        | 0.1                |

Additional Features:
- **Positional encoding** for capturing sequence order.
- **Multi-head attention**, **residual connections**, and **layer normalization**.
- **Early stopping** and **learning rate scheduling** for efficient training.
- **Gradient clipping** to prevent exploding gradients.

---

## 📈 Training

- Model trained using **CrossEntropyLoss** with label smoothing.
- Trained for **30 epochs** with early stopping after **no improvement for 7 epochs**.
- Train/validation split: 80% / 20%.

---


