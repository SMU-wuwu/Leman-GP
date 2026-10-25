from pathlib import Path
ROOT=Path('/root/data/WorkSpace')
OUT=ROOT/'output/leman_diversity_20261021'
FROZEN=ROOT/'output/leman_final_v2_20261019'
OLD=FROZEN/'wide_context6_consensus8'
SOURCES={
 'F42_C': dict(path=str(ROOT/'output/leman_controlled_ablation_v1/F_seed42/C/best.pt'),
              sha256='145224e85c808ddc84cda9177e7cd4a6731b004edbc713b03008bb0695fe01cd'),
 'F42_B': dict(path=str(ROOT/'output/leman_controlled_ablation_v1/F_seed42/B/best.pt'),
              sha256='f6eb2c4b472fd5e90063a6968054681ff2a52cdd4c155c34599fe3578f485fe2'),
 'no_lexicon':dict(path=str(ROOT/'output/leman_short_mechanism_v1/no_lexicon_seed42/best.pt'),
                  sha256='4c49a858864a44854176dca43b462f8e939e8699b7081c1e46fca440ce42697d'),
}
OLD_NAMES=['old','fgm43','fgm44','ema43','fgm46','soup2','soup3','parent','v4ema','v2ema','rdrop','fgm47']
GROUPS=[['F42_C'],['F42_C','F42_B'],['F42_C','F42_B','no_lexicon']]

