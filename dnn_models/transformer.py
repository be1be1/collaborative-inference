# from torch import Tensor
import torch.nn.functional as f
import torch
from torch import nn

def scaled_dot_product_attention(query, key, value):
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, query, key, value):
        temp = self.q(query).bmm(self.k(key).transpose(1, 2))
        scale = self.q(query).size(-1) ** 0.5
        softmax = f.softmax(temp / scale, dim=-1)
        return softmax.bmm(self.v(value))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_k, dim_v) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_v, dim_in)

    def forward(self, query, key, value):
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors):
        # Assume that the "value" tensor is given last, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[-1] + self.dropout(self.sublayer(*tensors)))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            nn.Sequential(
                nn.Linear(dim_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, dim_model)),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src):
        src = self.attention(src, src, src)
        return self.feed_forward(src)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.pos = torch.arange(16, dtype=torch.float).reshape(1, -1, 1)
        self.dim = torch.arange(512, dtype=torch.float).reshape(1, 1, -1)
        self.tpp = (self.pos / 1e4)
        self.dml = self.dim.long() % 2 == 0


    def forward(self, src):
        dimension = src.size(2)
        # print("seq_len", seq_len)
        # print("dimension", dimension)
        # pos = torch.arange(16, dtype=torch.float).reshape(1, -1, 1)
        # dim = torch.arange(512, dtype=torch.float).reshape(1, 1, -1)
        phase = self.tpp ** torch.div(self.dim, dimension, rounding_mode='floor')
        src += torch.where(self.dml, torch.sin(phase), torch.cos(phase))
        for layer in self.layers:
            src = layer(src)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            nn.Sequential(
                nn.Linear(dim_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, dim_model)),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, tgt, memory):
        tgt = self.attention_1(tgt, tgt, tgt)
        tgt = self.attention_2(memory, memory, tgt)
        return self.feed_forward(tgt)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(dim_model, dim_model)
        self.pos = torch.arange(16, dtype=torch.float).reshape(1, -1, 1)
        self.dim = torch.arange(512, dtype=torch.float).reshape(1, 1, -1)
        self.tpp = (self.pos / 1e4)
        self.dml = self.dim.long() % 2 == 0


    def forward(self, tgt, memory):
        dimension = tgt.size(2)
        phase = self.tpp ** torch.div(self.dim, dimension, rounding_mode='floor')
        tgt += torch.where(self.dml, torch.sin(phase), torch.cos(phase))
        for layer in self.layers:
            tgt = layer(tgt, memory)

        return torch.softmax(self.linear(tgt), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, src, tgt):
        return self.decoder(tgt, self.encoder(src))


