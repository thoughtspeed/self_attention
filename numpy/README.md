<img width="342" height="53" alt="image" src="https://github.com/user-attachments/assets/528c3ce1-4bbf-4876-8bfb-3f934978e051" /># Self-Attention in NumPy

## Overview

Self-attention is a core idea behind modern transformer models. It lets each token in a sequence look at the other tokens and decide which ones matter most for its representation. That is why self-attention is so useful for language, vision, and sequence modeling tasks.

This project provides a small, readable NumPy implementation that demonstrates the mechanics of self-attention without the extra complexity of a full deep learning framework.

## Project Structure

```text
self_attention/
├── self_attention.py
└── README.md
```

## How It Works

The implementation follows these steps:

1. Create embeddings for the input sequence.
2. Project the input into three learned spaces:
   - Queries $Q$
   - Keys $K$
   - Values $V$
3. Compute attention scores using the scaled dot product:

   ![Alt text](images/scaled_dot_product.jpg)

   $$
   \text{score}(q_i, k_j) = \frac{q_i \cdot k_j}{\sqrt{d_k}}
   $$

4. Apply softmax to convert scores into attention weights.
5. Compute the weighted sum of values:

   $$
   \text{output}_i = \sum_j \alpha_{ij} v_j
   $$

Where $\alpha_{ij}$ are the attention weights.

## Quick Start

Run the demo from the project folder:

```bash
python self_attention.py
```

You should see:
- the input sequence,
- the attention weights,
- the resulting attention output.

## Example Output

```text
Input sequence:
[[ 0.496  0.768  0.088  1.013]
 [ 1.120  0.139  0.166  0.184]
 [-1.137  1.059  0.283  0.772]]

Attention weights:
[[0.333 0.333 0.333]
 [0.333 0.333 0.333]
 [0.333 0.333 0.333]]

Attention output:
[[...]
 [...]
 [...]]
```

The exact numbers will vary because the weights are initialized randomly.

## Why This Matters

This example is useful for developers because it:

- makes the attention mechanism easy to inspect,
- highlights the core math behind transformers,
- provides a clean starting point for experimentation.

## Next Steps

Ideas for extending the project:

- add multi-head attention,
- compare the NumPy version with a PyTorch implementation,
- visualize attention weights as a heatmap,
- test on real text or sequence data.
