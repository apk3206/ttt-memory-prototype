from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    hidden_size: int = 512
    intermediate_size: int = 1536
    num_layers: int = 8
    num_heads: int = 8
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.1
    mixer: str = "attn"  # attn | ttt_linear
    ttt_mini_batch: int = 16
    ttt_base_lr: float = 1.0


@dataclass
class TrainConfig:
    seq_len: int = 256
    batch_size: int = 4
    grad_accum: int = 2
    epochs: int = 10
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0
    dropout: float = 0.1
    eval_batches: int = 64
    seed: int = 42
    mode: str = "pretrain"  # pretrain | meta
    ttt_lr: float = 1e-2
    ttt_steps: int = 1
    amp: str = "auto"
    optimizer: str = "adamw"  # adamw | adam
    adapt_prefixes: tuple[str, ...] = ("blocks.",)
