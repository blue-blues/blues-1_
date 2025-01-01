import dataclasses
import torch

from model import *


@dataclasses.dataclass
class config:
    vocab_size: int = 30000
    max_position_embeddings: int = 256
    num_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    hidden_size: int = 192
    embedding_multiplier_scale: int = 4
    tot_num_experts: int = 4
    chosen_num_experts: int = 1
    noise_std: float = 0.05
    lambadada: float = 0.5
    head_dim: int = 48  # hidden_size / num_attention_heads
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    use_scale: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout = 0.1
    # batch_size = 128



#     model = model.to(device)
