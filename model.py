from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class BgeConfig:
    embed_dim: int = 1024
    word_size: int = 250_002
    position_size: int = 8192


class BgeM3Embedding(nn.Module):
    def __init__(self, config: BgeConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_of_words = config.word_size
        self.embed_dim = config.embed_dim
        self.position_embed_dim = config.position_size

        # word embedding
        self.word_embedding = nn.Embedding(self.num_of_words, self.embed_dim)
        # position embedding
        self.position_embedding = nn.Embedding(self.position_embed_dim, self.embed_dim)
        # token type embedding
        self.token_type_embedding = nn.Embedding(2, self.embed_dim)

        # LayerNorm
        self.LayerNorm = nn.LayerNorm(self.embed_dim)


    def forward(self, input_ids, token_type_ids=None):
        pass


class BgeM3(nn.Module):
    def __init__(self, config: BgeConfig):
        super().__init__()

        # Save the config
        self.config = config

        # Embedding
        self.embedding = BgeM3Embedding(config)
