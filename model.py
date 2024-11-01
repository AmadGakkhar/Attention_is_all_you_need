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

        # After calculating the attention scores, the shape of x is
        # (batch, h, seq_len, d_k). This is because the attention scores
        # are calculated for each head (h), and the output of the attention
        # mechanism is a weighted sum of the value vector (d_k).
        #
        # To prepare the output for the final linear layer, we need to
        # reshape x to (batch, seq_len, d_model). We do this by first
        # transposing the tensor so that the sequence length is the second
        # dimension, and then reshaping the tensor to (batch, seq_len, d_model).
        # The '-1' in the view function means that the size of the last dimension
        # is inferred from the other dimensions.
        #
        # The contiguous function is used to make sure that the tensor is
        # contiguous in memory, which can improve performance.
        # The contiguous function is used to make sure that the tensor is
        # contiguous in memory. This can improve performance, because many
        # operations in PyTorch are optimized for contiguous tensors.
        #
        # When you call view on a tensor, it returns a new tensor that shares
        # the same underlying data as the original tensor, but with a different
        # size and stride. However, the resulting tensor may not be contiguous
        # in memory, because it may have "holes" or "gaps" in the memory.
        #
        # For example, if you have a tensor of shape (3, 4, 5) and you call
        # view to reshape it to (6, 10), the resulting tensor will have a
        # stride of (5, 1) and a size of (6, 10), but it will not be contiguous
        # in memory. This is because the original tensor has a stride of (4, 5)
        # and a size of (3, 4, 5), so the memory layout of the original tensor
        # is not compatible with the memory layout of the reshaped tensor.
        #
        # The contiguous function takes a tensor as input and returns a new
        # tensor that has the same size and stride as the original tensor, but
        # is contiguous in memory. This is done by allocating new memory for
        # the tensor and copying the data from the original tensor to the new
        # tensor.
        #
        # The contiguous function is useful when you need to make sure that a
        # tensor is contiguous in memory, but you don't want to allocate new
        # memory for the tensor. This can be the case when you are working with
        # large tensors and you want to avoid allocating new memory.
        #
        # The contiguous function is also useful when you are working with
        # tensors that have a complex memory layout, such as tensors that have
        # been reshaped or tensors that have been sliced. In these cases, the
        # contiguous function can be used to make sure that the tensor is
        # contiguous in memory, even if the original tensor is not contiguous.
        #


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        """
        Parameters:
        - dropout: The dropout rate to be used in the Dropout layer

        Initializes the ResidualConnection module.
        The ResidualConnection module is a class that is used to create a residual connection
        between two layers. It applies a dropout to the output of the previous layer and then
        adds the output of the previous layer to the output of the current layer.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        """
        Parameters:
        - x: The input to the residual connection
        - sublayer: The sublayer to be used in the residual connection

        Returns:
        - The output of the residual connection

        Applies a residual connection to the sublayer. The residual connection
        is a module that is used to connect the output of two layers. It applies
        a dropout to the output of the previous layer and then adds the output
        of the previous layer to the output of the current layer.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    # In the EncoderBlock class, a lambda function is used in the first layer
    # of residual_connection to wrap the self.self_attention_block function,
    # but not in the second layer.

    # The reason for this difference is that self.self_attention_block expects
    # four arguments (q, k, v, and mask), but self.feed_forward_block expects
    # only one argument (x).


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, trg_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, trg_mask)
        )
        x = self.residual_connection[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x


class DecoderModule(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
            return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: DecoderModule,
        projection: ProjectionLayer,
        src_embed: InputEmbeddings,
        trg_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        trg_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projection = projection
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, trg, trg_mask):
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg, encoder_output, src_mask, trg_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_covab_size: int,
    trg_vocab_size: int,
    src_seq_len: int,
    trg_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    dff: int = 2044,
):

    src_embed = InputEmbeddings(d_model, src_covab_size)
    trg_embed = InputEmbeddings(d_model, trg_vocab_size)
    src_pos = PositionalEncoding(d_model, dropout, src_seq_len)
    trg_pos = PositionalEncoding(d_model, dropout, trg_seq_len)
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, dff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    decoder_blocks = []

    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, dff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = DecoderModule(nn.ModuleList(decoder_blocks))
    projection = ProjectionLayer(d_model, trg_vocab_size)

    transformer = Transformer(
        encoder, decoder, projection, src_embed, trg_embed, src_pos, trg_pos, projection
    )

    # initialize Parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
