from dataclasses import dataclass


@dataclass
class BgeTrainerConfig:
    epochs: int = 10
    batch_size: int = 8


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


def get_config_for_tiny() -> BgeConfig:
    return BgeConfig(
        word_size=250_002,
        position_size=8192,
        layer_norm_eps=1e-5,
        embed_dim=1024,
        hidden_size=4096,
        dropout_prob=0.1,
        num_of_attn_layers=1,
        num_heads=16,
        attn_dim=1024,
        attn_output_dim=1024,
        attn_layer_norm_eps=1e-5,
        ffn_layer_norm_eps=1e-5
    )


def get_base_config() -> BgeConfig:
    return BgeConfig(
        word_size=250_002,
        position_size=8192,
        layer_norm_eps=1e-5,
        embed_dim=1024,
        hidden_size=4096,
        dropout_prob=0.1,
        num_of_attn_layers=24,
        num_heads=16,
        attn_dim=1024,
        attn_output_dim=1024,
        attn_layer_norm_eps=1e-5,
        ffn_layer_norm_eps=1e-5
    )
