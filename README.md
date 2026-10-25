# Leman-GP

Code, experiment settings, and derived result artifacts for:

> **Leman-GP: Lexicon-enhanced multi-head manhattan attention model with global pointer for Chinese agricultural named entity recognition**

## Overview

This repository accompanies the manuscript and contains the training, evaluation, ablation, mechanism-analysis, candidate-fusion, and plotting code used in the study. It is intended to support inspection and reproduction of the reported experiments.

The final test result is:

| Metric | Value |
|---|---:|
| Precision | 95.34% |
| Recall | 94.70% |
| Micro F1 | 95.02% |

## What is included

- `WorkSpace/`: training, lexicon construction, baselines, ablations, mechanism probes, candidate fusion, evaluation, and plotting code.
- `Paper_Final_V3/`: final-system manifests, calibration assets, test predictions, and result records.
- `Historical_Source_Snapshots/`: source snapshots kept with the original experiments.
- `Archive_Audit/`: environment and checkpoint-cleanup records.
- `README_先读.md`: Chinese guide to the archive and code entry points.

## What is not included

- **Raw AgCNER data.** The original train/dev/test splits and labels are not redistributed here. See `WorkSpace/data/README.md` for the official AgCNER download and citation.
- **Pretrained encoder weights.** The repository contains code and tokenizer resources, but not `model.safetensors` or other large neural checkpoints.

## Reproduction

1. Read `README_先读.md` and the README in the target experiment directory.
2. Download AgCNER from the original release and place `train2.txt`, `dev2.txt`, `test2.txt`, and `label.txt` under `WorkSpace/data/`.
3. Prepare the ERNIE 3.0 encoder weights/configuration described in the manuscript.
4. Install the dependencies required by the target experiment; version records are preserved in the corresponding `requirements.txt` and `run_config.json` files.
5. Run the experiment according to its local README/config. Many archived scripts use the original server path `/root/data/WorkSpace`; update the root path for your environment.

## Data and code availability

- AgCNER dataset: https://github.com/guojson/AgCNER
- Dataset paper: https://doi.org/10.1038/s41597-024-03578-5

The raw data are not included in this repository. The code and derived result artifacts are provided for academic review and reproducibility.

## Citation

If you use this repository, please cite the associated manuscript:

Wu Zhao, Lin Zhang. *Leman-GP: Lexicon-enhanced multi-head manhattan attention model with global pointer for Chinese agricultural named entity recognition.* Manuscript under review.

## Contact

Lin Zhang, Shanghai Maritime University — linzhang@shmtu.edu.cn

## License

No open-source license has been declared for this repository. Please contact the authors before reusing the code or result artifacts.
