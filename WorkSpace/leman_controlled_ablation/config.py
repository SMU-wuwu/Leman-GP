"""Fixed AgCNER controlled matrix; public encoder -> A -> B -> C."""
from dataclasses import dataclass
from pathlib import Path
from parent_config import Config as ParentConfig

VARIANTS = ('F', 'no_lexicon', 'no_attention', 'dot_attention', 'crf',
            'max_words3', 'max_words5', 'encoder_gp')
PHASES = {
    'A': dict(epochs=20, encoder_lr=3e-5, head_lr=1e-4,
              gp_positive_weight=1.0, rare_type_exponent=0.0),
    'B': dict(epochs=5, encoder_lr=1e-6, head_lr=1e-5,
              gp_positive_weight=1.1, rare_type_exponent=0.25),
    'C': dict(epochs=3, encoder_lr=3e-6, head_lr=1e-5,
              gp_positive_weight=1.1, rare_type_exponent=0.25),
}


@dataclass
class Config(ParentConfig):
    variant: str = 'F'
    phase: str = 'A'
    seed: int = 42
    warm_start_sha256: str = ''
    schema: str = 'leman-controlled-ablation-v1'
    experiment: str = 'AgCNER controlled architecture matrix'
    resource_dir: str = ''
    qa: bool = False

    @classmethod
    def make(cls, root, variant, seed, phase):
        if variant not in VARIANTS or phase not in PHASES:
            raise ValueError('Unknown variant or phase')
        cfg = cls(root=str(Path(root).resolve()), variant=variant, seed=seed, phase=phase)
        for name, value in PHASES[phase].items():
            setattr(cfg, name, value)
        cfg.reused_head_lr = cfg.head_lr
        cfg.data_dir = str(Path(cfg.root) / 'data')
        cfg.encoder_dir = str(Path(cfg.root) / 'model')
        base = Path(cfg.root) / 'output/leman_controlled_ablation_v1'
        cfg.resource_dir = str(base / 'resources')
        cfg.output_dir = str(base / f'{variant}_seed{seed}' / phase)
        cfg.warm_start = ('' if phase == 'A' else
                          str(Path(cfg.output_dir).parent / chr(ord(phase)-1) / 'best.pt'))
        cfg.max_words = {'max_words3': 3, 'max_words5': 5}.get(variant, 4)
        if variant == 'encoder_gp':
            cfg.boundary_weight = 0.0
        return cfg
