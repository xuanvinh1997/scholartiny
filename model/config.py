"""Explicit model configuration; no silent architecture fallbacks."""
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

@dataclass
class ModelConfig:
    vocab_size: int = 32768
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    intermediate_size: int = 2816
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    layer_pattern: str = 'MMMA'
    ssm_backend: str = 'mamba2'  # mamba2: upstream CUDA; reference: slow, for tests
    state_size: int = 64
    ssm_expand: int = 2
    ssm_head_dim: int = 64
    conv_kernel: int = 4
    ssm_chunk_size: int = 256
    conv_bias: bool = True
    qk_norm: bool = True
    rope_theta: float = 100000.0
    norm_eps: float = 1e-6
    max_seq_len: int = 8192
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False
    loss_chunk_size: int = 256

    def __post_init__(self):
        positive = ['vocab_size','hidden_size','num_hidden_layers','intermediate_size',
                    'num_attention_heads','num_key_value_heads','state_size','ssm_expand',
                    'ssm_head_dim','conv_kernel','max_seq_len','loss_chunk_size']
        if any(getattr(self, k) <= 0 for k in positive):
            raise ValueError('Sizes must be positive')
        if not self.layer_pattern or set(self.layer_pattern) - {'M', 'A'}:
            raise ValueError('layer_pattern contains only M and A')
        if self.num_hidden_layers % len(self.layer_pattern):
            raise ValueError('Layer count must be a multiple of pattern length')
        if self.hidden_size % self.num_attention_heads:
            raise ValueError('hidden_size must divide attention heads')
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError('attention heads must divide KV heads')
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError('RoPE head dimension must be even')
        if self.hidden_size*self.ssm_expand % self.ssm_head_dim:
            raise ValueError('Expanded SSM width must divide ssm_head_dim')
        if self.ssm_backend not in ('reference','mamba2'):
            raise ValueError('Choose reference or mamba2 explicitly')

    @property
    def layers(self):
        return self.layer_pattern * (self.num_hidden_layers // len(self.layer_pattern))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**yaml.safe_load(Path(path).read_text(encoding='utf-8')))
