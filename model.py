from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class BgeConfig:
    word_size: int = 250_002
    position_size: int = 8192
    layer_norm_eps: float = 1e-5

    embed_dim: int = 1024
    hidden_size: int = 1024

    attn_dim: int = 1024
    attn_output_dim: int = 1024
    attn_layer_norm_eps: float = 1e-5


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


class MultiHeadAttention(nn.Module):
    def __init__(self, config: BgeConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_size = config.hidden_size
        self.attn_dim = config.attn_dim
        self.attn_output_dim = config.attn_output_dim
        self.attn_layer_norm_eps = config.attn_layer_norm_eps

        self.q = nn.Linear(self.hidden_size, self.attn_dim)
        self.k = nn.Linear(self.hidden_size, self.attn_dim)
        self.v = nn.Linear(self.hidden_size, self.attn_dim)
        self.o = nn.Linear(self.attn_dim, self.attn_output_dim)

class BgeM3(nn.Module):
    def __init__(self, config: BgeConfig):
        super().__init__()

        # Save the config
        self.config = config

        # Embedding
        self.embedding = BgeM3Embedding(config)


    def forward(self, input_ids, token_type_ids=None):
        # Embedding
        embed = self.embedding(input_ids, token_type_ids)

        out = embed
        return out
