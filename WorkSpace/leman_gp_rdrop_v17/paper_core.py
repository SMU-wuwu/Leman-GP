"""Leman-GP: character-aligned ERNIE -> paper Lexicon Adapter -> BiGRU
-> negative-L1 multihead attention -> RoPE Global Pointer.

Equation references refer to the supplied manuscript, before renumbering.
"""
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import List
from transformers import AutoConfig, AutoModel


# PyTorch 2.7 CUDA cdist backward computes `l * l_size` as signed int32.
# Cap EACH call's [batch*heads, key, query, head_dim] temporary well below
# INT32_MAX. The default cap is 512 MiB in FP32, not a model hyperparameter.
CDIST_BACKWARD_MAX_ELEMENTS = 1 << 27


def safe_l1_cdist(q, k, max_workspace_elements=CDIST_BACKWARD_MAX_ELEMENTS):
    """Exact L1 with bounded CUDA backward indexing and temporary memory.

    Partition independent batch/head matrices only; every query still attends
    to every key. Ordinary autograd accumulates both q and k gradients.
    """
    if q.shape[:-2] != k.shape[:-2] or q.shape[-1] != k.shape[-1]:
        raise ValueError('safe_l1_cdist requires matching batch/head and feature dimensions.')
    if not 0 < max_workspace_elements < 2 ** 31:
        raise ValueError('cdist workspace element limit must be below signed int32 maximum.')
    nq, nk, dim = q.shape[-2], k.shape[-2], q.shape[-1]
    per_matrix = nq * nk * dim
    if q.numel() == 0 or k.numel() == 0:
        return torch.cdist(q, k, p=1)
    if per_matrix > max_workspace_elements:
        raise ValueError('One distance matrix exceeds the safe workspace. Reduce character window length.')
    group_size = max_workspace_elements // per_matrix
    flat_q = q.reshape(-1, nq, dim)
    flat_k = k.reshape(-1, nk, dim)
    pieces = [torch.cdist(flat_q[start:start + group_size], flat_k[start:start + group_size], p=1)
              for start in range(0, flat_q.shape[0], group_size)]
    distances = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
    return distances.reshape(*q.shape[:-2], nq, nk)


@torch.jit.script
def _gru_direction(projected: torch.Tensor, mask: torch.Tensor,
                   gate_weight: torch.Tensor, candidate_weight: torch.Tensor,
                   reverse: bool) -> torch.Tensor:
    """Equations (11)-(14): reset BEFORE the candidate hidden projection."""
    hidden_size = candidate_weight.shape[0]
    state = projected.new_zeros((projected.shape[0], hidden_size))
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    for step in range(projected.shape[1]):
        index = projected.shape[1] - 1 - step if reverse else step
        gates = torch.nn.functional.linear(state, gate_weight) + projected[:, index, :2 * hidden_size]
        update = torch.sigmoid(gates[:, :hidden_size])
        reset = torch.sigmoid(gates[:, hidden_size:])
        candidate = torch.tanh(projected[:, index, 2 * hidden_size:]
                               + torch.nn.functional.linear(reset * state, candidate_weight))
        next_state = (1.0 - update) * state + update * candidate
        valid = mask[:, index:index + 1]
        state = torch.where(valid, next_state, state)
        outputs.append(state * valid)
    result = torch.stack(outputs, dim=1)
    return torch.flip(result, [1]) if reverse else result


class PaperGRUCell(nn.Module):
    def __init__(self, input_size, hidden):
        super().__init__()
        self.input_projection = nn.Linear(input_size, 3 * hidden, bias=False)
        self.hidden_gates = nn.Parameter(torch.empty(2 * hidden, hidden))
        self.hidden_candidate = nn.Parameter(torch.empty(hidden, hidden))
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -(hidden ** -0.5), hidden ** -0.5)

    def forward(self, x, mask, reverse=False):
        return _gru_direction(self.input_projection(x), mask, self.hidden_gates,
                              self.hidden_candidate, reverse)


class PaperBiGRU(nn.Module):
    def __init__(self, input_size, hidden, layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([PaperGRUCell(input_size if i == 0 else 2 * hidden, hidden),
                           PaperGRUCell(input_size if i == 0 else 2 * hidden, hidden)])
            for i in range(layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        for i, directions in enumerate(self.layers):
            x = torch.cat([directions[0](x, mask), directions[1](x, mask, True)], dim=-1)
            if i + 1 < len(self.layers):
                x = self.dropout(x)
        return x


class LexiconAdapter(nn.Module):
    """Equations (4)-(10): per-candidate BMES, MLP, bilinear attention, residual."""
    def __init__(self, hidden, vocab_size, cfg):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, cfg.word_dim, padding_idx=0)
        # B=0, M=1, E=2, S=3; PAD=4, so the B embedding is trainable.
        self.boundary_embedding = nn.Embedding(5, cfg.boundary_dim, padding_idx=4)
        self.projection = nn.Sequential(
            nn.Linear(cfg.word_dim + cfg.boundary_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),
        )
        self.bilinear = nn.Linear(hidden, hidden, bias=False)

    def forward(self, characters, word_ids, boundary_ids):
        valid = word_ids.ne(0)
        candidates = self.projection(torch.cat([
            self.word_embedding(word_ids), self.boundary_embedding(boundary_ids)
        ], dim=-1))
        scores = (self.bilinear(characters).unsqueeze(2) * candidates).sum(-1).float()
        # All-PAD positions contribute exactly zero rather than a softmax NaN.
        weights = scores.masked_fill(~valid, -1e30).softmax(-1) * valid
        lexical = (weights.to(candidates.dtype).unsqueeze(-1) * candidates).sum(2)
        return characters + lexical


class MultiHeadManhattanAttention(nn.Module):
    """Equations (17)-(20), with no temperature or dot-product branch."""
    def __init__(self, hidden, heads):
        super().__init__()
        if hidden % heads:
            raise ValueError('BiGRU output dimension must be divisible by attention heads.')
        self.heads = heads
        self.head_dim = hidden // heads
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, mask):
        b, length, hidden = x.shape
        def split(layer):
            return layer(x).reshape(b, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q), split(self.k), split(self.v)
        # Exact L1 in FP32. Bound each cdist backward call to avoid the PyTorch
        # CUDA int32 batch-offset overflow and a multi-GiB temporary tensor.
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = -safe_l1_cdist(q.float(), k.float())
            weights = scores.masked_fill(~mask[:, None, None, :], -1e30).softmax(-1)
        attended = torch.matmul(weights.to(v.dtype), v)
        result = self.out(attended.transpose(1, 2).reshape(b, length, hidden))
        return result * mask.unsqueeze(-1)


def rotate_pairs(x):
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)


class GlobalPointer(nn.Module):
    """Type-specific projections and the unscaled RoPE dot product of Eq. (23)."""
    def __init__(self, hidden, types, head_size):
        super().__init__()
        if head_size % 2:
            raise ValueError('RoPE head size must be even.')
        self.types, self.head_size = types, head_size
        self.projection = nn.Linear(hidden, types * head_size * 2, bias=False)
        nn.init.normal_(self.projection.weight, std=0.02)
        self.register_buffer('inv_freq', 10000.0 ** (-torch.arange(0, head_size, 2).float() / head_size))

    def forward(self, x, mask):
        b, length, _ = x.shape
        z = self.projection(x).reshape(b, length, self.types, 2, self.head_size)
        q, k = z[..., 0, :].float(), z[..., 1, :].float()
        # Loss and GP scores are FP32 even when the encoder uses BF16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            phase = torch.arange(length, device=x.device).float()[:, None] * self.inv_freq[None, :]
            cos = phase.cos().repeat_interleave(2, -1)[None, :, None, :]
            sin = phase.sin().repeat_interleave(2, -1)[None, :, None, :]
            q = q * cos + rotate_pairs(q) * sin
            k = k * cos + rotate_pairs(k) * sin
            scores = torch.einsum('bith,bjth->btij', q, k)
        valid = valid_spans(mask)
        return scores.masked_fill(~valid, -1e30)


def valid_spans(mask):
    length = mask.shape[-1]
    upper = torch.ones(length, length, device=mask.device, dtype=torch.bool).triu()
    return mask[:, None, :, None] & mask[:, None, None, :] & upper


class GlobalPointerLoss(nn.Module):
    """Standard multilabel categorical cross-entropy, independently per type.

    mean_(batch,type)[ log(1+sum_negative exp(s))
                       + log(1+sum_positive exp(-s)) ].
    The appended zero makes empty positive/negative sets well-defined.
    """
    def forward(self, logits, labels, attention_mask):
        valid = valid_spans(attention_mask)
        positive = labels.bool() & valid
        negative = ~labels.bool() & valid
        scores = logits.float()
        zeros = scores.new_zeros((*scores.shape[:2], 1))
        pos = (-scores).masked_fill(~positive, -torch.inf).flatten(2)
        neg = scores.masked_fill(~negative, -torch.inf).flatten(2)
        return (torch.logsumexp(torch.cat([pos, zeros], -1), -1)
                + torch.logsumexp(torch.cat([neg, zeros], -1), -1)).mean()


class LemanGP(nn.Module):
    def __init__(self, cfg, num_types, vocab_size, *, encoder_config=None, pretrained=True):
        super().__init__()
        enc_cfg = (AutoConfig.from_pretrained(cfg.encoder_dir, local_files_only=True)
                   if encoder_config is None else AutoConfig.for_model(**encoder_config))
        enc_cfg.use_cache = False
        if pretrained:
            self.encoder = AutoModel.from_pretrained(
                cfg.encoder_dir, config=enc_cfg, local_files_only=True,
                torch_dtype=torch.float32, attn_implementation='sdpa',
            )
        else:
            self.encoder = AutoModel.from_config(enc_cfg, attn_implementation='sdpa')
        pooler = getattr(self.encoder, 'pooler', None)
        if pooler is not None:
            pooler.requires_grad_(False)
        self.adapter = LexiconAdapter(enc_cfg.hidden_size, vocab_size, cfg)
        self.gru = PaperBiGRU(enc_cfg.hidden_size, cfg.gru_hidden, cfg.gru_layers, cfg.gru_dropout)
        self.attention = MultiHeadManhattanAttention(cfg.gru_hidden * 2, cfg.attention_heads)
        self.global_pointer = GlobalPointer(cfg.gru_hidden * 2, num_types, cfg.gp_head_size)
        self.loss_fn = GlobalPointerLoss()

    def forward(self, input_ids, token_mask, char_pool, char_mask, word_ids,
                boundary_ids, lengths, labels=None):
        # Keep BF16 encoder projections, with the math SDPA backend accumulating
        # attention intermediates in FP32. Backend selection changes no formula.
        with sdpa_kernel(SDPBackend.MATH):
            hidden = self.encoder(input_ids=input_ids, attention_mask=token_mask).last_hidden_state
        # Disabling autocast BEFORE custom projections matters: casting their
        # outputs to float afterwards cannot recover an overflow in the product.
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            # A character receives the mean of overlapping subword vectors.
            characters = torch.bmm(char_pool.float(), hidden.float())
            features = self.adapter(characters, word_ids, boundary_ids)
            sequence = self.gru(features, char_mask)
            scores = self.global_pointer(self.attention(sequence, char_mask), char_mask)
            if labels is None:
                return scores
            return self.loss_fn(scores, labels, char_mask), scores
