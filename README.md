# NanoGPT: Minimal GPT Implementation for Tiny Shakespeare

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

## Overview

This repository contains a minimal implementation of a GPT-style transformer model (NanoGPT) trained on the Tiny Shakespeare dataset. The project demonstrates the core concepts of modern deep learning for language modeling, including:

- Token and positional embeddings
- Multi-head causal self-attention
- Transformer blocks with residual connections
- Layer normalization and dropout
- Autoregressive text generation

The code is organized as a Jupyter Notebook for clarity and reproducibility.

---

## Features

- **Dataset:** Tiny Shakespeare (public domain, ~1MB)
- **Tokenizer:** GPT-2 BPE via [tiktoken](https://github.com/openai/tiktoken)
- **Model:** Custom transformer (NanoGPT) with configurable depth, width, and attention heads
- **Training:** AdamW optimizer, learning rate warmup and cosine decay, gradient clipping
- **Visualization:** Training/validation loss curves, learning rate schedule
- **Text Generation:** Temperature and top-k sampling, prompt-based generation
- **Checkpointing:** Save/load model weights (excluded from git tracking)

---

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Jupyter Notebook
- tiktoken
- numpy
- matplotlib

### Installation

```bash
pip install torch tiktoken numpy matplotlib jupyter
```

### Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DPHeshanRanasinghe/Hello-World-of-modern-Deep-Learning.git
   cd Hello-World-of-modern-Deep-Learning
   ```
2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook model.ipynb
   ```
3. **Run the notebook cells sequentially:**
   - Data download and preprocessing
   - Model definition
   - Training loop
   - Visualization
   - Text generation

### Checkpoints
- Model checkpoints (`*.pt`) are **not tracked by git** and are excluded via `.gitignore`.
- To save/load checkpoints, use the provided code in the notebook.

---

## Project Structure

```
├── model.ipynb           # Main Jupyter Notebook
├── .gitignore            # Ignore large files and checkpoints
├── README.md             # Project documentation
├── input.txt             # Tiny Shakespeare dataset (downloaded automatically)
├── train.bin, val.bin    # Tokenized data (generated)
└── nanogpt_checkpoint.pt # Model checkpoint (local only)
```

---

## Model Configuration

- **Embedding dimension:** 256 (default)
- **Transformer layers:** 4
- **Attention heads:** 4
- **Context length:** 128 tokens
- **Batch size:** 16
- **Dropout:** 0.2
- **Optimizer:** AdamW

You can adjust these parameters in the notebook for different hardware or experiments.

---

## Results & Examples

After training, generate Shakespeare-style text by providing a prompt (e.g., `ROMEO:` or `To be or not to be`).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) for the Tiny Shakespeare dataset and inspiration
- [OpenAI tiktoken](https://github.com/openai/tiktoken) for fast BPE tokenization
- [PyTorch](https://pytorch.org/) for deep learning framework

---

## Author

**Heshan Ranasinghe**  
Electronic and Telecommunication Engineering Undergraduate

- Email: hranasinghe505@gmail.com
- GitHub: [@DPHeshanRanasinghe](https://github.com/DPHeshanRanasinghe)
- LinkedIn: [Heshan Ranasinghe](https://www.linkedin.com/in/heshan-ranasinghe-988b00290)
