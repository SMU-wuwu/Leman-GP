"""Exact F1 v2 forward graph with a recall-oriented GP training objective."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel

from paper_core import (GlobalPointer, LexiconAdapter,
                        MultiHeadManhattanAttention, PaperBiGRU, safe_l1_cdist, valid_spans)


class RecallGlobalPointerLoss(nn.Module):
    """Type-weighted MLCCE with a modest multiplier on the positive term.

    At unit weights this reduces exactly to the parent loss. Weights are
    configuration, not learned state, so the parent state_dict stays compatible.
    """
    def __init__(self, weights, positive_weight):
        super().__init__()
        self.register_buffer('type_weights', torch.tensor(weights, dtype=torch.float32), persistent=False)
        self.positive_weight = positive_weight

    def forward(self, logits, labels, attention_mask):
        valid = valid_spans(attention_mask)
        positive = labels.bool() & valid
        negative = ~labels.bool() & valid
        scores = logits.float()
        zeros = scores.new_zeros((*scores.shape[:2], 1))
        pos = (-scores).masked_fill(~positive, -torch.inf).flatten(2)
        neg = scores.masked_fill(~negative, -torch.inf).flatten(2)
        terms = (self.positive_weight * torch.logsumexp(torch.cat([pos, zeros], -1), -1)
                 + torch.logsumexp(torch.cat([neg, zeros], -1), -1))
        weights = self.type_weights
        return (terms * weights).sum() / (scores.shape[0] * weights.sum())


class GatedLexiconAdapter(LexiconAdapter):
    def __init__(self, hidden, vocab_size, cfg):
        super().__init__(hidden, vocab_size, cfg)
        self.cfg = cfg
        self.gate = nn.Linear(2 * hidden, hidden)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, characters, word_ids, boundary_ids):
        valid = word_ids.ne(0)
        if self.training:
            # A word disappears at every covered character, including its BMES roles.
            keep = torch.rand(self.word_embedding.num_embeddings, device=word_ids.device) >= self.cfg.lexicon_word_dropout
            valid = valid & keep[word_ids]
            keep_sentence = torch.rand(word_ids.shape[0], 1, 1, device=word_ids.device) >= self.cfg.lexicon_dropout
            valid = valid & keep_sentence
        candidates = self.projection(torch.cat([
            self.word_embedding(word_ids), self.boundary_embedding(boundary_ids)
        ], -1))
        scores = (self.bilinear(characters).unsqueeze(2) * candidates).sum(-1) / math.sqrt(characters.shape[-1])
        weights = scores.masked_fill(~valid, -1e30).softmax(-1) * valid
        context = (weights.unsqueeze(-1) * candidates).sum(2)
        gate = torch.sigmoid(self.gate(torch.cat([characters, context], -1)))
        enriched = self.norm(characters + self.dropout(gate * context))
        return torch.where(valid.any(-1, keepdim=True), enriched, characters)


class ResidualManhattanAttention(MultiHeadManhattanAttention):
    def __init__(self, hidden, heads, dropout):
        super().__init__(hidden, heads)
        self.qnorm = nn.LayerNorm(self.head_dim)
        self.knorm = nn.LayerNorm(self.head_dim)
        self.raw_temperature = nn.Parameter(torch.full((heads,), math.log(math.expm1(1.0))))
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        b, length, hidden = x.shape
        def split(layer):
            return layer(x).reshape(b, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = self.qnorm(split(self.q)), self.knorm(split(self.k)), split(self.v)
        tau = F.softplus(self.raw_temperature).view(1, self.heads, 1, 1) + 1e-4
        scores = -safe_l1_cdist(q, k) * tau / math.sqrt(self.head_dim)
        weights = scores.masked_fill(~mask[:, None, None, :], -1e30).softmax(-1)
        attended = self.dropout(weights) @ v
        context = self.out(attended.transpose(1, 2).reshape(b, length, hidden))
        return self.norm(x + self.dropout(context)) * mask.unsqueeze(-1)


class ScaledGlobalPointer(GlobalPointer):
    def __init__(self, hidden, types, head_size, max_span):
        super().__init__(hidden, types, head_size)
        self.max_span = max_span

    def forward(self, x, mask):
        scores = super().forward(x, mask) / math.sqrt(self.head_size)
        length = scores.shape[-1]
        positions = torch.arange(length, device=scores.device)
        allowed = (positions[None, :] >= positions[:, None]) & (positions[None, :] - positions[:, None] < self.max_span)
        return scores.masked_fill(~allowed[None, None], -1e30)


def boundary_loss(logits, labels, mask):
    """The same logsumexp ranking loss, over start/end positions per type.

    labels: [batch,type,start,end]; logits: [batch,length,type,2].
    Positive-only and negative-only types remain trainable without pos_weight.
    """
    targets = torch.stack((labels.any(-1), labels.any(-2)), dim=2)
    scores = logits.permute(0, 2, 3, 1).float()
    valid = mask[:, None, None, :]
    zeros = scores.new_zeros((*scores.shape[:-1], 1))
    pos = (-scores).masked_fill(~targets | ~valid, -torch.inf)
    neg = scores.masked_fill(targets | ~valid, -torch.inf)
    return (torch.logsumexp(torch.cat((pos, zeros), -1), -1)
            + torch.logsumexp(torch.cat((neg, zeros), -1), -1)).mean()


class LemanGP(nn.Module):
    def __init__(self, cfg, num_types, vocab_size, *, char_vocab_size,
                 encoder_config=None, pretrained=True):
        super().__init__()
        self.cfg = cfg
        enc_cfg = (AutoConfig.from_pretrained(cfg.encoder_dir, local_files_only=True)
                   if encoder_config is None else AutoConfig.for_model(**encoder_config))
        enc_cfg.use_cache = False
        if pretrained:
            self.encoder = AutoModel.from_pretrained(
                cfg.encoder_dir, config=enc_cfg, local_files_only=True,
                dtype=torch.float32, attn_implementation='sdpa')
        else:
            self.encoder = AutoModel.from_config(enc_cfg, dtype=torch.float32, attn_implementation='sdpa')
        pooler = getattr(self.encoder, 'pooler', None)
        if pooler is not None:
            pooler.requires_grad_(False)
        hidden = enc_cfg.hidden_size
        self.char_embedding = nn.Embedding(char_vocab_size, cfg.char_dim, padding_idx=0)
        self.char_projection = nn.Linear(cfg.char_dim, hidden, bias=False)
        nn.init.normal_(self.char_projection.weight, std=0.02)
        self.char_scale = nn.Parameter(torch.tensor(0.1))
        self.adapter = GatedLexiconAdapter(hidden, vocab_size, cfg)
        self.gru = PaperBiGRU(hidden, cfg.gru_hidden, cfg.gru_layers, cfg.gru_dropout)
        self.sequence_norm = nn.LayerNorm(2 * cfg.gru_hidden)
        self.dropout = nn.Dropout(cfg.dropout)
        self.attention = ResidualManhattanAttention(2 * cfg.gru_hidden, cfg.attention_heads, cfg.dropout)
        self.global_pointer = ScaledGlobalPointer(2 * cfg.gru_hidden, num_types, cfg.gp_head_size, cfg.max_span)
        self.boundary = nn.Linear(2 * cfg.gru_hidden, num_types * 2)
        self.num_types = num_types
        weights = cfg.gp_type_weights or [1.0] * num_types
        if len(weights) != num_types or min(weights) <= 0 or cfg.gp_positive_weight <= 0:
            raise ValueError('GP loss needs one positive weight per entity type and a positive multiplier.')
        self.loss_fn = RecallGlobalPointerLoss(weights, cfg.gp_positive_weight)

    def forward(self, input_ids, token_mask, char_pool, char_mask, word_ids,
                boundary_ids, lengths, char_ids, labels=None):
        # Let PyTorch use its fused SDPA backend when eligible.
        hidden = self.encoder(input_ids=input_ids, attention_mask=token_mask).last_hidden_state
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            characters = torch.bmm(char_pool.float(), hidden.float())
            characters = characters + self.char_scale * self.char_projection(self.char_embedding(char_ids))
            features = self.adapter(characters, word_ids, boundary_ids) * char_mask.unsqueeze(-1)
            sequence = self.dropout(self.sequence_norm(self.gru(features, char_mask))) * char_mask.unsqueeze(-1)
            scores = self.global_pointer(self.attention(sequence, char_mask), char_mask)
            if labels is None:
                return scores
            auxiliary = self.boundary(sequence).reshape(*sequence.shape[:2], self.num_types, 2)
            loss = self.loss_fn(scores, labels, char_mask)
            return loss + self.cfg.boundary_weight * boundary_loss(auxiliary, labels, char_mask), scores
