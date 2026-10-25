# Structure comparison

| Component | Previous completed run | Upgrade | Reused values |
|---|---|---|---|
| ERNIE | subword hidden states averaged onto source characters | same encoder + train-only character embedding/residual | pretrained encoder and all ERNIE weights |
| Lexicon | train gold and jieba segments, max length 8; un-gated residual lexical vectors | max length 40; shared word dropout, sentence dropout, gated residual and norm | all matching word rows by string, BMES embedding, MLP, bilinear weights |
| Sequence | reset-before custom 2-layer BiGRU | same, followed by sequence norm/dropout | all GRU parameters |
| Context | negative L1 Q/K/V attention without residual or normalization | normalized Q/K, learnable temperature, dimension scaling, residual + LayerNorm | old Q/K/V/output projection weights |
| Span head | RoPE GlobalPointer raw dot scores | same projections/RoPE; training scores `/sqrt(64)`, candidate spans capped at 64 | all old span-head parameters |
| Auxiliary supervision | none | typed start/end multi-label categorical CE, weight .1 | none; new head |
| Training | 40 epochs, batch 192/GPU, train from pretrained ERNIE | 12-epoch low-LR warm-start fine-tuning, 4×RTX 4090 auto batch, EMA | model weights only; optimizer starts fresh |
| Evaluation | raw score threshold 0 | same raw metric plus dev full calibration and score-threshold crossfit | same original character spans and exact metric |

No LayerNorm is inserted into the ERNIE backbone and no scores/losses are clipped or replaced. A numerical nonfinite still stops before the optimizer step. The old PyTorch 2.7 cdist backward indexing fix stays in place.

## Four-card deployment revision

- Train/calibrate/test launch four processes, one per RTX 4090.
- Automatic batch candidates: 4, 8, 12, 16, 24, 32, 48, 64; memory budget 86% per card.
- Probe reserves Adam moments, one DDP bucket and EMA shadow in addition to actual weights/gradients/activations.
- Eval batch 16/GPU, workers 2/rank, OMP threads 2/rank.
- Separate code/output directories; original A100 warm-start checkpoint and lexicon reused.
- Model, losses, optimizer learning rates, 12 epochs, EMA and five-epoch snapshots are unchanged from F1 v2.
