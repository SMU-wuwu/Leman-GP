"""Five-epoch paired interventions, conditional on the completed F42 parent."""
from dataclasses import dataclass
from pathlib import Path
from parent_config import Config as Parent

VARIANTS = ('F', 'attention_norm_only', 'uniform_attention')
PHASES = {'S': {}}
PARENTS = {
 'A': '8ff4aa5c6ea2f691df2d29b5a5ca3acba8aff50cf7059c4ef94461c57044e21b',
 'B': 'f6eb2c4b472fd5e90063a6968054681ff2a52cdd4c155c34599fe3578f485fe2',
 'C': '145224e85c808ddc84cda9177e7cd4a6731b004edbc713b03008bb0695fe01cd'}

@dataclass
class Config(Parent):
    variant: str = 'F'
    phase: str = 'S'
    seed: int = 42
    schema: str = 'leman-controlled-ablation-v1'  # cache identity, not this experiment's identity
    experiment: str = 'leman-attention-mechanism-v2'
    resource_dir: str = ''
    qa: bool = False
    use_fgm: bool = False
    two_clean_passes: bool = False
    probe_sentences: int = 128
    parent_phase: str = 'C'
    protocol: str = 'conditional intervention plus fixed-budget adaptation; not from-scratch ablation'

    @classmethod
    def make(cls, root, variant, seed, phase='S'):
        if variant not in VARIANTS or phase != 'S': raise ValueError('Unknown short experiment')
        c=cls(root=str(Path(root).resolve()),variant=variant,seed=seed)
        c.epochs=3; c.encoder_lr=3e-6; c.head_lr=c.reused_head_lr=1e-5
        c.parent_phase=('B' if variant in ('fgm_control','no_fgm','clean_two_pass') else
                        'A' if variant in ('weighted_control','unit_loss') else 'C')
        if variant in ('fgm_control','no_fgm','clean_two_pass'): c.epochs=3
        if variant in ('weighted_control','unit_loss'): c.encoder_lr=1e-6
        if variant=='unit_loss': c.gp_positive_weight=1.; c.rare_type_exponent=0.
        c.use_fgm=variant=='fgm_control'
        c.two_clean_passes=variant=='clean_two_pass'
        c.data_dir=str(Path(c.root)/'data');c.encoder_dir=str(Path(c.root)/'model')
        c.resource_dir=str(Path(c.root)/'output/leman_controlled_ablation_v1/resources')
        c.output_dir=str(Path(c.root)/'output/leman_attention_mechanism_v2'/f'{variant}_seed{seed}')
        c.warm_start=str(Path(c.root)/'output/leman_controlled_ablation_v1/F_seed42'/c.parent_phase/'best.pt')
        c.warm_start_sha256=PARENTS[c.parent_phase]
        c.max_words={'max_words3':3,'max_words5':5}.get(variant,4)
        if variant in ('encoder_gp','no_boundary'):c.boundary_weight=0.
        return c
