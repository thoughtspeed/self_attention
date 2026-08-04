# PyTorch Self-Attention

## Overview

This folder contains a minimal PyTorch implementation of self-attention based on the core idea from the Into AI article. It shows how a sequence of token embeddings can be transformed into queries, keys, and values, scored, and combined to produce a richer representation.

## What the Code Does

The implementation uses a small `SelfAttention` module that:

- projects the input into queries, keys, and values,
- computes attention scores with $QK^T$,
- scales the scores by $\sqrt{d_k}$,
- applies softmax to obtain attention weights,
- multiplies the weights by values to form the output.

## Project Structure

```text
pytorch/
├── self_attention.py
└── README.md
```

## How to Run

From this folder, run:

```bash
python self_attention.py
```

You should see:

- the input embedding tensor,
- the attention weights,
- the attention output tensor.

## Notes

This version is intentionally simple and educational. It is meant to help you understand the mechanics of self-attention before moving to more advanced transformer components like multi-head attention or full encoder/decoder stacks.

## Next Steps

You can extend this example by:

- adding positional encodings,
- implementing multi-head attention,
- comparing it with a NumPy version,
- using real text embeddings.
