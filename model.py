from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class BgeConfig:
    word_size: int = 250_002
    position_size: int = 8192
    layer_norm_eps: float = 1e-5

    embed_dim: int = 1024
    hidden_size: int = 4096
    dropout_prob: float = 0.1

    num_of_attn_layers: int = 24

    num_heads: int = 16

    attn_dim: int = 1024
    attn_output_dim: int = 1024
    attn_layer_norm_eps: float = 1e-5
    ffn_layer_norm_eps: float = 1e-5


class BgeM3Embedding(nn.Module):
    def __init__(self, config: BgeConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_of_words = config.word_size
        self.embed_dim = config.embed_dim
        self.position_embed_dim = config.position_size
        self.layer_norm_eps = config.layer_norm_eps

        # word embedding
        self.word_embedding = nn.Embedding(self.num_of_words, self.embed_dim)
        # position embedding
        self.position_embedding = nn.Embedding(self.position_embed_dim, self.embed_dim)
        # token type embedding
        self.token_type_embedding = nn.Embedding(1, self.embed_dim)

        # LayerNorm
        self.LayerNorm = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps, bias=True)


    def forward(self, input_ids, token_type_ids=None):
        # word embedding
        word_embed = self.word_embedding(input_ids)

        # position embedding
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        position_embed = self.position_embedding(position_ids)

        # token type embedding
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embed = self.token_type_embedding(token_type_ids)

        # sum
        embed = word_embed + position_embed + token_type_embed

        # LayerNorm
        embed = self.LayerNorm(embed)

        return embed


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: BgeConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embed_dim = config.embed_dim
        self.attn_dim = config.attn_dim
        self.attn_output_dim = config.attn_output_dim
        self.attn_layer_norm_eps = config.attn_layer_norm_eps
        self.num_heads = config.num_heads

        # Attention
        self.q = nn.Linear(self.embed_dim, self.attn_dim)
        self.k = nn.Linear(self.embed_dim, self.attn_dim)
        self.v = nn.Linear(self.embed_dim, self.attn_dim)
        self.o = nn.Linear(self.attn_dim, self.attn_output_dim)
        self.LayerNorm = nn.LayerNorm(self.attn_output_dim, eps=self.attn_layer_norm_eps, bias=True)


    def forward(self, x):
        num_heads = self.num_heads
        attn_dim = self.attn_dim
        batch_size, seq_length, _ = x.size()

        head_dim = attn_dim // num_heads

        # Linear projections
        # q, k, v: (batch_size, num_heads, seq_length, head_dim)
        q = self.q(x).view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
        k = self.k(x).view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
        v = self.v(x).view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
        attn = torch.nn.functional.softmax(attn, dim=-1)

        # Attention output
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        out = self.o(out)

        # LayerNorm
        out = self.LayerNorm(out)

        return out


class FFN(nn.Module):
    def __init__(self, config: BgeConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attn_output_dim = config.attn_output_dim
        self.hidden_size = config.hidden_size
        self.ffn_layer_norm_eps = config.ffn_layer_norm_eps
        self.dropout_prob = config.dropout_prob

        self.fc1 = nn.Linear(self.attn_output_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.attn_output_dim)

        self.gelu = torch.nn.functional.gelu
        self.output_dropout = nn.Dropout(self.dropout_prob)

        self.LayerNorm = nn.LayerNorm(self.attn_output_dim, eps=self.ffn_layer_norm_eps, bias=True)


    def forward(self, x):
        # FFN
        out = torch.nn.functional.relu(self.fc1(x))
        out = self.gelu(out)
        out = self.fc2(out)

        # Residual connection
        out = out + x

        # Dropout
        out = self.output_dropout(out)

        # LayerNorm
        out = self.LayerNorm(out)

        return out


class BgeAttention(nn.Module):
    def __init__(self, config: BgeConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Attention
        self.attention = MultiHeadSelfAttention(config)

        # FFN
        self.ffn = FFN(config)


    def forward(self, x):
        # Attention
        out = self.attention(x)

        # FFN
        out = self.ffn(out)

        return out


class BgeM3(nn.Module):
    def __init__(self, config: BgeConfig):
        super().__init__()

        # Save the config
        self.config = config

        self.num_of_attn_layers = config.num_of_attn_layers

        # Embedding
        self.embedding = BgeM3Embedding(config)

        # Attention
        self.attentions = nn.ModuleList([
            BgeAttention(config) for _ in range(self.num_of_attn_layers)
        ])

        #TODO dense retriever, sparse retriever, colbert retriever


    def forward(self, input_ids, token_type_ids=None):
        # Embedding
        embed = self.embedding(input_ids, token_type_ids)

        out = embed

        # Attention
        for attention in self.attentions:
            out = attention(out)

        # pooling (explicit pooling)
        dense_retrieval_vec = out[:, 0, :]

        #TODO colbert retrieval, sparse retrieval
        sparse_retrieval_vec = None
        colbert_retrieval_vec = None

        return dense_retrieval_vec, sparse_retrieval_vec, colbert_retrieval_vec
