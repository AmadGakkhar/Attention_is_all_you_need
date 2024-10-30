import torch
import torch.nn as nn
import math

## Input Embeddings Module.
## Converts original sentence to a vector of 512 dims
## Word -> Input ID (position in vocab) -> Vector 512 dims
## Embeddings layer is learnable


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        Parameters:
        - vocab_size: Size of the vocabulary
        - d_model:   The dimensionality of the embedding space
        """
        self.d_model = d_model
        self.vocab_size = vocab_size

        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass of the embeddings layer.

        Parameters:
        - x: The input tensor (Batch_size, Sequence_length) of type torch.long

        Returns:
        - The embedded tensor (Batch_size, Sequence_length, d_model) of type torch.float
        """
        return self.emb(x) * math.sqrt(self.d_model)


## Positional Encoding Module
## Adds positional information to the input embeddings
## Positional info is added as sinusoidal wave


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        # Create a Matrix of shape (max_len, d_model)
        pe = torch.zeros(self.seq_len, self.d_model)
        # Create a position vector of shape (seq_len, 1)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)

        # Create a div vector of shape (seq_len, 1)
        # Calculate the div_term, which is used to scale the sinusoidal functions
        # This term is a tensor of shape (d_model/2,), where each element is
        # calculated as exp(position * -log(10000) / d_model). It allows the
        # positional encoding to have varying wavelengths for each dimension.

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )

        # Apply Sinusoidal function to the position vector
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape the pe matrix to (1, seq_len, d_model) for multiple sentences
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass of the Positional Encoding layer.

        Parameters:
        - x: The input tensor (Batch_size, Sequence_length, d_model) of type torch.float

        Returns:
        - The embedded tensor (Batch_size, Sequence_length, d_model) of type torch.float
        after adding the positional encoding and dropout.
        """
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


## Normalization Layer
## Normalizes the input embeddings


class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplicative Term of shape (1,1)
        self.beta = nn.Parameter(torch.zeros(1))  # Aditive Term of shape (1,1)

    def forward(self, x):
        # mean is the mean of the input embeddings along the last dimension
        # of shape (Batch_size, Sequence_length, 1)
        mean = x.mean(dim=-1, keepdim=True)
        # std is the standard deviation of the input embeddings along the last dimension
        # of shape (Batch_size, Sequence_length, 1)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.beta
