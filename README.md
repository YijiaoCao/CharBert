# CharBert
# Custom Character-level BERT Implementation From Scratch (2 pretraining tasks and 5 downstream tasks) in PyTorch

This repository contains a **from-scratch BERT implementation** in PyTorch, supporting multiple NLP tasks and **parameter-efficient fine-tuning with LoRA**. It is designed for experimentation, research, and learning purposes.

## Features
- **Core BERT architecture**
  - Multi-head self-attention with residual connections
  - Feed-forward network with GELU activation
  - Layer normalization and dropout
- **Token, position, and sentence embeddings**
- **LoRA (Low-Rank Adaptation)**
  - Optional LoRA layers for efficient fine-tuning
  - Freeze transformer parameters and train only LoRA modules
- **Supported tasks**
  - Masked Language Modeling (MLM)
  - Next Sentence Prediction (NSP)
  - Theme / text classification
  - Sentence similarity scoring
  - Question answering (start-end span prediction)
  - Named Entity Recognition (NER)
  - Relation prediction
- **Training and fine-tuning pipelines**
  - Pre-training and full fine-tuning
  - Gradient clipping and learning rate scheduling
  - Loss monitoring for training and validation splits

## Installation
```bash
git clone https://github.com/YijiaoCao/Custom-BERT-PyTorch.git
cd Custom-BERT-PyTorch
pip install -r requirements.txt
```

## Usage
- Set hyperparameters in hyp() function.
- Prepare datasets for different tasks (MLM, NSP, QA, etc.).
- Train / fine-tune models:
```python
from train import pretrain, full_fine_tune, lora_fine_tune

# Initialize models
models()

# Pretrain BERT
pretrain()

# Full fine-tuning on downstream tasks
full_fine_tune()

# LoRA fine-tuning
lora_fine_tune()
```

Save checkpoint:
```python
checkpoint()
```

## Repository Structure
```bash
.
├── models/                 # BERT architecture and task heads
├── data/                   # Datasets for different NLP tasks
├── notebooks/              # Optional Jupyter notebooks for experimentation
├── train.py                # Training and fine-tuning pipelines
├── requirements.txt        # Python dependencies
├── README.md
└── LICENSE
```
