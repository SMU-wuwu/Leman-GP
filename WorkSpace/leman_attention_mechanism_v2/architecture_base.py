"""Architecture variants with shared initialization inherited from the full model."""
import math
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from parent_model import LemanGP as FullModel, ResidualManhattanAttention, ScaledGlobalPointer, boundary_loss
from crf_reference import LinearCRF


class ResidualDotAttention(ResidualManhattanAttention):
    def forward(self, x, mask):
        b, length, hidden = x.shape
        def split(layer):
            return layer(x).reshape(b, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = self.qnorm(split(self.q)), self.knorm(split(self.k)), split(self.v)
        tau = F.softplus(self.raw_temperature).view(1, self.heads, 1, 1) + 1e-4
        scores = (q @ k.transpose(-1, -2)) * tau / math.sqrt(self.head_dim)
        weights = scores.masked_fill(~mask[:, None, None, :], -1e30).softmax(-1)
        context = self.out((self.dropout(weights) @ v).transpose(1, 2).reshape(b, length, hidden))
        return self.norm(x + self.dropout(context)) * mask.unsqueeze(-1)


class LemanGP(FullModel):
    def __init__(self, cfg, num_types, vocab_size, *, char_vocab_size,
                 encoder_config=None, pretrained=True):
        # Construct all original modules first. Removing a module cannot shift
        # random initialization of later shared modules.
        super().__init__(cfg, num_types, vocab_size, char_vocab_size=char_vocab_size,
                         encoder_config=encoder_config, pretrained=pretrained)
        if cfg.variant == 'no_lexicon':
            del self.adapter
        elif cfg.variant == 'no_attention':
            del self.attention
        elif cfg.variant == 'dot_attention':
            with torch.random.fork_rng(devices=[]):
                replacement = ResidualDotAttention(2*cfg.gru_hidden, cfg.attention_heads, cfg.dropout)
                replacement.load_state_dict(self.attention.state_dict(), strict=True)
            self.attention = replacement
        elif cfg.variant == 'crf':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(cfg.seed + 17001)
                self.classifier = nn.Linear(2*cfg.gru_hidden, 1+4*num_types)
                self.crf = LinearCRF(1+4*num_types)
            del self.global_pointer
        elif cfg.variant == 'encoder_gp':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(cfg.seed + 17002)
                self.global_pointer = ScaledGlobalPointer(self.encoder.config.hidden_size, num_types,
                                                          cfg.gp_head_size, cfg.max_span)
            for name in ('char_embedding', 'char_projection', 'char_scale', 'adapter', 'gru',
                         'sequence_norm', 'dropout', 'attention', 'boundary'):
                delattr(self, name)

    def forward(self, input_ids, token_mask, char_pool, char_mask, word_ids,
                boundary_ids, lengths, char_ids, labels=None, crf_tags=None):
        if self.cfg.variant in ('F', 'max_words3', 'max_words5', 'dot_attention'):
            return super().forward(input_ids, token_mask, char_pool, char_mask, word_ids,
                                   boundary_ids, lengths, char_ids, labels)
        hidden = self.encoder(input_ids=input_ids, attention_mask=token_mask).last_hidden_state
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            characters = torch.bmm(char_pool.float(), hidden.float())
            if self.cfg.variant == 'encoder_gp':
                scores = self.global_pointer(characters, char_mask)
                return scores if labels is None else (self.loss_fn(scores, labels, char_mask), scores)
            characters = characters + self.char_scale * self.char_projection(self.char_embedding(char_ids))
            features = (characters if self.cfg.variant == 'no_lexicon' else
                        self.adapter(characters, word_ids, boundary_ids)) * char_mask.unsqueeze(-1)
            sequence = self.dropout(self.sequence_norm(self.gru(features, char_mask))) * char_mask.unsqueeze(-1)
            attended = sequence if self.cfg.variant == 'no_attention' else self.attention(sequence, char_mask)
            scores = self.classifier(attended) if self.cfg.variant == 'crf' else self.global_pointer(attended, char_mask)
            if labels is None:
                return scores
            auxiliary = self.boundary(sequence).reshape(*sequence.shape[:2], self.num_types, 2)
            if self.cfg.variant == 'crf':
                if crf_tags is None:
                    raise ValueError('CRF training requires the BIOES collation path')
                loss = self.crf(scores, crf_tags, char_mask)
            else:
                loss = self.loss_fn(scores, labels, char_mask)
            return loss + self.cfg.boundary_weight * boundary_loss(auxiliary, labels, char_mask), scores


def initialization_record(model):
    """Small identity record for same-seed shared-parameter comparisons."""
    return {name: {'shape': list(value.shape),
                   'sha256': hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            for name, value in model.state_dict().items()}
