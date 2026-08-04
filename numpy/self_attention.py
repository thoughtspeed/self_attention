import numpy as np


class SelfAttention:
    """A simple self-attention layer implemented with NumPy."""

    def __init__(self, input_dim: int, output_dim: int | None = None):
        """Initialize learnable weight matrices for Q, K, and V."""
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Scale the random initialization so the values stay numerically stable.
        scale = 1.0 / np.sqrt(input_dim)
        self.Wq = np.random.randn(input_dim, output_dim) * scale
        self.Wk = np.random.randn(input_dim, output_dim) * scale
        self.Wv = np.random.randn(input_dim, output_dim) * scale

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax over the requested axis."""
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def forward(self, x: np.ndarray):
        """
        Compute self-attention for an input sequence.

        Parameters
        ----------
        x : np.ndarray
            Shape (seq_len, input_dim).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The attention output and the attention weights.
        """
        # Project the input into query, key, and value spaces.
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # Compute scaled dot-product attention scores.
        scores = (Q @ K.T) / np.sqrt(K.shape[1])

        # Convert scores to attention weights with softmax.
        weights = self._softmax(scores, axis=-1)

        # Weighted sum of values produces the attention output.
        output = weights @ V
        return output, weights


if __name__ == "__main__":
    # Create a small random input sequence: 3 tokens, each with 4 features.
    np.random.seed(42)
    x = np.random.randn(3, 4)

    # Instantiate the attention module and run a forward pass.
    attention = SelfAttention(input_dim=4)
    output, weights = attention.forward(x)

    # Print the input sequence, attention weights, and output.
    np.set_printoptions(precision=3, suppress=True)
    print("Input sequence:")
    print(x)
    print("\nAttention weights:")
    print(weights)
    print("\nAttention output:")
    print(output)
