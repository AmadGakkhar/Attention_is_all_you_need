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


## FeedForward Class
## Applies a feedforward neural network to the input embeddings


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # self.linear1 is a fully connected layer with an input size of d_model
        # and an output size of d_ff. It applies a linear transformation to the
        # input and produces an output of shape (Batch_size, Sequence_length, d_ff).
        # The number of neurons in this layer is d_ff.
        # The output of this layer is passed through a ReLU activation function
        self.linear1 = nn.Linear(d_model, d_ff)

        # self.linear2 is a fully connected layer with an input size of d_ff
        # and an output size of d_model. It applies a linear transformation to the
        # input and produces an output of shape (Batch_size, Sequence_length, d_model).
        # The number of neurons in this layer is d_model.
        # The output of this layer is passed through a Dropout layer with a dropout rate of dropout.
        self.linear2 = nn.Linear(d_ff, d_model)

        # self.dropout is a Dropout layer with a dropout rate of dropout.
        # It randomly sets dropout% of the output of the previous layer to 0.
        # The output of this layer is passed through a ReLU activation function.
        self.dropout = nn.Dropout(dropout)

        # dummy example
        # x = torch.randn(2, 3, d_model)  # (Batch_size, Sequence_length, d_model)
        # out1 = self.linear1(x)
        # print(out1.shape)  # (Batch_size, Sequence_length, d_ff)
        # out2 = self.linear2(out1)
        # print(out2.shape)  # (Batch_size, Sequence_length, d_model)

        def forward(self, x):
            return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        d_k = q.shape[-1]
        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        ## k goes from (batch, h, seq_len, d_k) --> (batch, h, d_k, seq_len)
        ## attention shape --> (batch, h, seq_len, seq_len)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        # Apply softmax to the attention scores. The softmax operation is
        # applied along the last dimension (dim=-1) of the tensor.
        # The softmax function takes the input tensor and returns a tensor of
        # the same shape, where all the values are in the range [0, 1] and the
        # sum of all the values is 1.
        # The dim=-1 argument means that the softmax operation is applied
        # along the last dimension of the tensor.
        # The reason for applying softmax is to ensure that the attention
        # weights are normalized and the model can focus on the most important
        # parts of the input sequence.

        attention_score = attention_score.softmax(
            dim=-1
        )  # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_score = dropout(attention_score)

        ##shape of attention_score @ v --> (batch, h, seq_len, d_k)
        return (attention_score @ v), attention_score

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  ## (batch , seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  ## (batch , seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  ## (batch , seq_len, d_model) --> (batch, seq_len, d_model)

        ## (batch, seq_len, d_model) --> (batch, seq_len, h, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k)

        ## (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        x, attention_score = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        x = (
            x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        )  # Shape of x is (batch, seq_len, d_model)
        return self.w_o(x)
