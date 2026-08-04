import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """A simple self-attention layer implemented in PyTorch."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Learnable projection matrices for queries, keys, and values.
        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Compute self-attention for a batch of token embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Shape: [batch_size, sequence_length, embedding_dim]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The attention output and the attention weights.
        """
        # Project input embeddings into Q, K, and V spaces.
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Compute attention scores by matching queries against keys.
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale the scores to keep softmax numerically stable.
        attn_scores = attn_scores / (self.embedding_dim**0.5)

        # Convert scores into attention weights.
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Use the attention weights to combine the value vectors.
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


if __name__ == "__main__":
    torch.manual_seed(0)

    # Example input: 2 sentences, each with 4 tokens, each token has 4 features.
    batch_size = 2
    sequence_length = 4
    embedding_dim = 4

    input_embeddings = torch.randn(batch_size, sequence_length, embedding_dim)

    # Create and run the self-attention layer.
    self_attention = SelfAttention(embedding_dim=embedding_dim)
    output, attn_weights = self_attention(input_embeddings)

    # Print a compact demo of the mechanism.
    torch.set_printoptions(precision=3, sci_mode=False)
    print("Input embeddings shape:", input_embeddings.shape)
    print("Input embeddings:\n", input_embeddings)
    print("\nAttention weights shape:", attn_weights.shape)
    print("Attention weights:\n", attn_weights)
    print("\nOutput shape:", output.shape)
    print("Output:\n", output)

