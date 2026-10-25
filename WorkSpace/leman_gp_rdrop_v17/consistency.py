"""R-Drop adapted to per-type span rankings; no change to inference or MLCCE."""
from contextlib import nullcontext
import torch


def span_symmetric_kl(first, second, char_mask, max_span):
    """Mean symmetric KL over valid-span categorical distributions plus null.

    The normalized distributions are a training regularizer, not a claim that
    entities are mutually exclusive. The original multilabel supervised loss
    remains responsible for all gold spans. The null category handles empty types.
    """
    length = first.shape[-1]
    pos = torch.arange(length,device=first.device)
    pair = (pos[None,:] >= pos[:,None]) & (pos[None,:]-pos[:,None] < max_span)
    valid = pair[None,None] & char_mask[:,None,:,None] & char_mask[:,None,None,:]
    def log_distribution(scores):
        scores = scores.float().masked_fill(~valid,-1e30).flatten(2)
        scores = torch.cat((scores,scores.new_zeros((*scores.shape[:2],1))),-1)
        return scores.log_softmax(-1)
    a,b = log_distribution(first),log_distribution(second)
    return (0.5*(a.exp()-b.exp())*(a-b)).sum(-1).mean()


def rdrop_backward(wrapped, batch, *, weight, loss_scale=1., autocast=nullcontext,
                   check_finite=None):
    """One DDP forward on a duplicated batch, one backward, independent masks.

    A single DDP forward avoids multiple-reducer-forward corner cases. The mean
    supervised loss on the duplicated batch equals the mean of two stochastic
    supervised losses. External no_sync controls gradient accumulation. Ordinary
    layer and per-example dropout masks differ across copies; the unchanged
    adapter's vocabulary-wide word-dropout mask is shared within this forward.
    """
    count = batch['input_ids'].shape[0]
    doubled = {name:torch.cat((value,value),0) for name,value in batch.items()}
    with autocast():
        supervised, scores = wrapped(**doubled)
    kl = span_symmetric_kl(scores[:count],scores[count:],batch['char_mask'],wrapped.module.cfg.max_span)
    loss = supervised + weight*kl
    if check_finite is not None:
        check_finite(loss,'rdrop_loss')
    (loss*loss_scale).backward()
    return loss.detach(),{'supervised_loss':float(supervised.detach()),'symmetric_kl':float(kl.detach())}
