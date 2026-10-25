"""Paper architecture and the two-A100 execution preset."""
from dataclasses import asdict, dataclass
import os
from pathlib import Path


@dataclass
class Config:
    root: str = '/root/data/WorkSpace'
    data_dir: str = ''
    encoder_dir: str = ''
    output_dir: str = ''
    seed: int = 42
    epochs: int = 40
    # 0 selects the largest measured batch within the memory budget at startup.
    batch_size: int = 0
    max_auto_batch: int = 512
    memory_fraction: float = 0.88
    eval_batch_size: int = 0
    encoder_lr: float = 3e-5
    head_lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.10
    grad_clip: float = 1.0
    workers: int = 4
    preprocessing_workers: int = 8
    max_chars: int = 128
    overlap_chars: int = 32
    max_encoder_tokens: int = 512
    max_words: int = 4
    max_word_length: int = 8
    min_segment_frequency: int = 2
    word_dim: int = 256
    boundary_dim: int = 128
    gru_hidden: int = 384
    gru_layers: int = 2
    gru_dropout: float = 0.2
    attention_heads: int = 8
    gp_head_size: int = 64
    threshold: float = 0.0
    save_every_epochs: int = 5
    save_every_steps: int = 100
    amp: str = 'bf16'
    schema: str = 'leman-paper-char-gp-v1'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_environment(cls):
        cfg = cls()
        env = {
            'root': 'ROOT', 'data_dir': 'DATA', 'encoder_dir': 'ENCODER',
            'output_dir': 'OUTPUT', 'seed': 'SEED', 'epochs': 'EPOCHS',
            'batch_size': 'BATCH_SIZE', 'max_auto_batch': 'MAX_AUTO_BATCH',
            'eval_batch_size': 'EVAL_BATCH_SIZE', 'memory_fraction': 'MEMORY_FRACTION',
            'encoder_lr': 'ENCODER_LR', 'head_lr': 'HEAD_LR', 'workers': 'WORKERS',
            'preprocessing_workers': 'PREPROCESS_WORKERS', 'save_every_steps': 'SAVE_STEPS',
        }
        for name, suffix in env.items():
            value = os.environ.get('LEMAN_GP_' + suffix)
            if value is not None:
                setattr(cfg, name, type(getattr(cfg, name))(value))
        cfg.root = str(Path(cfg.root).resolve())
        cfg.data_dir = cfg.data_dir or str(Path(cfg.root) / 'data')
        cfg.encoder_dir = cfg.encoder_dir or str(Path(cfg.root) / 'model')
        cfg.output_dir = cfg.output_dir or str(Path(cfg.root) / 'output' / f'leman_paper_a100_seed{cfg.seed}')
        if not 0.5 <= cfg.memory_fraction <= 0.95:
            raise ValueError('MEMORY_FRACTION must be in [0.5, 0.95].')
        if cfg.batch_size < 0 or cfg.max_auto_batch < 16 or cfg.epochs < 1:
            raise ValueError('Invalid batch size or epoch count.')
        if cfg.workers < 0 or cfg.preprocessing_workers < 1 or cfg.save_every_steps < 1:
            raise ValueError('Worker count / checkpoint interval is invalid.')
        return cfg
