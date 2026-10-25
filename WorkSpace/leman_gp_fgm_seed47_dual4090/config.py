"""Three-epoch FGM fine-tuning of the completed v4 best model."""
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path


@dataclass
class Config:
    root: str = '/root/data/WorkSpace'
    data_dir: str = ''
    encoder_dir: str = ''
    output_dir: str = ''
    warm_start: str = ''
    warm_start_sha256: str = '55b803b48f2aeee3fbfda60422ac3918ab6d87603cdd2f5039b7312f1ec300e4'
    seed: int = 47
    epochs: int = 3
    batch_size: int = 32
    max_auto_batch: int = 64
    memory_fraction: float = 0.86
    eval_batch_size: int = 16
    encoder_lr: float = 3e-6
    reused_head_lr: float = 1e-5
    head_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0
    ema_decay: float = 0.995
    workers: int = 2
    preprocessing_workers: int = 8
    max_chars: int = 192
    overlap_chars: int = 64
    max_encoder_tokens: int = 512
    max_span: int = 64
    max_words: int = 4
    max_word_length: int = 40
    min_segment_frequency: int = 2
    word_dim: int = 256
    boundary_dim: int = 128
    char_dim: int = 96
    gru_hidden: int = 384
    gru_layers: int = 2
    gru_dropout: float = 0.2
    attention_heads: int = 8
    gp_head_size: int = 64
    dropout: float = 0.15
    lexicon_dropout: float = 0.10
    lexicon_word_dropout: float = 0.10
    boundary_weight: float = 0.10
    gp_positive_weight: float = 1.10
    rare_type_exponent: float = 0.25
    rare_type_cap: float = 1.50
    gp_type_weights: list[float] = field(default_factory=list)
    fgm_epsilon: float = 0.5
    fgm_clean_weight: float = 0.5
    experiment: str = 'leman-fgm-seed47-dual4090'
    threshold: float = 0.0
    calibration_floor: float = -4.0
    calibration_step: float = 0.25
    calibration_min_support: int = 200
    save_every_epochs: int = 5
    save_every_steps: int = 250
    amp: str = 'bf16'
    schema: str = 'leman-f1-v2'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_environment(cls):
        cfg = cls()
        names = {
            'root': 'ROOT', 'data_dir': 'DATA', 'encoder_dir': 'ENCODER',
            'output_dir': 'OUTPUT', 'warm_start': 'WARM_START', 'seed': 'SEED',
            'epochs': 'EPOCHS', 'batch_size': 'BATCH_SIZE',
            'eval_batch_size': 'EVAL_BATCH_SIZE', 'max_auto_batch': 'MAX_AUTO_BATCH',
            'memory_fraction': 'MEMORY_FRACTION', 'encoder_lr': 'ENCODER_LR',
            'reused_head_lr': 'REUSED_HEAD_LR', 'head_lr': 'HEAD_LR',
            'workers': 'WORKERS', 'preprocessing_workers': 'PREPROCESS_WORKERS',
            'save_every_steps': 'SAVE_STEPS',
        }
        for name, suffix in names.items():
            value = os.environ.get('LEMAN_GP_' + suffix)
            if value is not None:
                setattr(cfg, name, type(getattr(cfg, name))(value))
        cfg.root = str(Path(cfg.root).resolve())
        cfg.data_dir = cfg.data_dir or str(Path(cfg.root) / 'data')
        cfg.encoder_dir = cfg.encoder_dir or str(Path(cfg.root) / 'model')
        cfg.output_dir = cfg.output_dir or str(Path(cfg.root) / 'output' / f'leman_fgm_v10_seed{cfg.seed}')
        cfg.warm_start = cfg.warm_start or str(Path(cfg.root) / 'output' / 'leman_finetune_v4_seed42' / 'best.pt')
        if cfg.batch_size < 0 or cfg.max_auto_batch < 4 or cfg.epochs < 1:
            raise ValueError('Use batch_size=0 for automatic GPU sizing or a positive batch size.')
        if cfg.fgm_epsilon <= 0 or not 0 < cfg.fgm_clean_weight < 1:
            raise ValueError('Use positive FGM epsilon and a clean weight strictly between zero and one.')
        return cfg
