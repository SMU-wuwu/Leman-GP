"""One FGM step: accumulate clean and perturbed gradients, then restore weights."""
from contextlib import nullcontext

import torch


class EmbeddingPerturbation:
    """Perturb only embedding rows read by this rank's nonpadding input tokens."""

    def __init__(self, embedding, input_ids, token_mask, epsilon):
        self.weight = embedding.weight
        if self.weight.grad is None:
            raise RuntimeError('FGM requires the clean embedding gradient first.')
        self.indices = torch.unique(input_ids[token_mask.bool()])
        if embedding.padding_idx is not None:
            self.indices = self.indices[self.indices != embedding.padding_idx]
        self.gradient = self.weight.grad.detach().index_select(0, self.indices)
        self.gradient_norm = torch.linalg.vector_norm(self.gradient, dtype=torch.float64)
        self.epsilon = epsilon
        self.original = None

    @torch.no_grad()
    def __enter__(self):
        if not torch.isfinite(self.gradient_norm).item():
            raise FloatingPointError('Non-finite FGM embedding gradient.')
        self.original = self.weight.index_select(0, self.indices).clone()
        scale = (self.epsilon / self.gradient_norm if self.gradient_norm.item() > 0
                 else self.gradient_norm.new_zeros(()))
        delta = self.gradient * scale.to(self.gradient.dtype)
        self.weight.index_copy_(0, self.indices, self.original + delta)
        self.actual_delta_norm = torch.linalg.vector_norm(
            self.weight.index_select(0, self.indices) - self.original, dtype=torch.float64)
        return self

    @torch.no_grad()
    def __exit__(self, exc_type, exc_value, traceback):
        self.weight.index_copy_(0, self.indices, self.original)
        return False


def fgm_backward(wrapped, batch, *, epsilon, clean_weight, autocast=nullcontext,
                 check_finite=None):
    """Return detached objective and diagnostics; caller clips and updates once.

    Clean and adversarial passes sample dropout independently. Each rank uses
    its local clean gradient to construct its own perturbation. The second DDP
    backward averages the sum of both weighted gradients across ranks.
    """
    if epsilon <= 0 or not 0 < clean_weight < 1:
        raise ValueError('FGM needs positive epsilon and a clean weight in (0, 1).')
    model = wrapped.module
    if check_finite is None:
        def check_finite(value, stage):
            if not torch.isfinite(value.detach()).all().item():
                raise FloatingPointError(stage)
    with wrapped.no_sync():
        with autocast():
            clean_loss, _ = wrapped(**batch)
        check_finite(clean_loss, 'fgm_clean_loss')
        (clean_weight * clean_loss).backward()
    attack = EmbeddingPerturbation(model.encoder.get_input_embeddings(),
                                  batch['input_ids'], batch['token_mask'], epsilon)
    check_finite(attack.gradient_norm, 'fgm_clean_embedding_gradient')
    with attack:
        with autocast():
            adversarial_loss, _ = wrapped(**batch)
        check_finite(adversarial_loss, 'fgm_adversarial_loss')
        ((1 - clean_weight) * adversarial_loss).backward()
    objective = clean_weight * clean_loss.detach() + (1 - clean_weight) * adversarial_loss.detach()
    return objective, {
        'clean_loss': float(clean_loss.detach()),
        'adversarial_loss': float(adversarial_loss.detach()),
        'active_embedding_rows': attack.indices.numel(),
        'clean_embedding_gradient_norm': float(attack.gradient_norm),
        'perturbation_l2': float(attack.actual_delta_norm),
        'zero_embedding_gradient': bool(attack.gradient_norm.item() == 0),
    }
