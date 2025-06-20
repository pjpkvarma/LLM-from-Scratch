# Building a Language Model from Scratch

This project demonstrates how to build a GPT-style transformer language model from scratch using PyTorch. The model is trained on the first four books of the Harry Potter series using character-level tokenization via the `tiktoken` library.

## Dataset

The model is trained on the following texts:
- Harry Potter and the Sorcerer's Stone
- Harry Potter and the Chamber of Secrets
- Harry Potter and the Prisoner of Azkaban
- Harry Potter and the Goblet of Fire

The books were combined into a single corpus and tokenized using GPT-2's encoding.

## Model Overview

The architecture is inspired by GPT models and includes:
- Token and positional embeddings
- Multi-head self-attention with causal masking
- Feedforward layers with GELU activations
- Layer normalization and residual connections
- Dropout for regularization

## Training

The model was trained using:
- Cross-entropy loss
- AdamW optimizer
- A context length of 1024 tokens
- A vocabulary size of 50,257 (GPT-2 BPE tokenizer)
- Batched input sequences generated using a sliding window

## Results

After several epochs, the model begins to generate coherent sequences in the style of the training text. Fine-tuning and scaling up can lead to further improvements.

## How to Use

- Clone this repository
- Install required dependencies (`torch`, `tiktoken`, etc.)
- Run the training script or notebook
- Use the generation utility to sample text from a trained checkpoint

## Future Work

- Add model checkpoint saving/loading
- Extend training to more books or other corpora
- Add attention visualization tools
- Integrate a simple CLI or web UI for text generation
