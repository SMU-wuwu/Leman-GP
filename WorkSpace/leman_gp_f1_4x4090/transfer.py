"""Explicit weight transfer and optimizer-independent EMA for short fine-tuning."""
from contextlib import contextmanager
from pathlib import Path
import json
import torch

from data import sha256


def load_parent(path, identity, types):
    path = Path(path)
    parent = torch.load(path, map_location='cpu', weights_only=False)
    if parent['schema'] != 'leman-paper-char-gp-v1':
        raise ValueError('Warm start expects the completed paper/fix2 best.pt, not a legacy BCE model.')
    if parent['types'] != types or parent['input_identity'] != identity:
        raise ValueError('Warm-start training data, labels and encoder/tokenizer must match the previous run.')
    if parent['source_hashes']['model.py'] != sha256(Path(__file__).with_name('paper_core.py')):
        raise ValueError('Warm-start model structure differs from the supplied paper_core.py.')
    lex_path = path.parent / 'lexicon.json'
    if sha256(lex_path) != parent['lexicon_sha256']:
        raise ValueError('Keep the original lexicon.json alongside the parent best.pt.')
    lexicon = json.loads(lex_path.read_text(encoding='utf-8'))
    return parent, lexicon


@torch.no_grad()
def transfer_weights(model, parent, old_lexicon, new_lexicon):
    state, old = model.state_dict(), parent['model']
    copied, new = [], []
    embedding_key = 'adapter.word_embedding.weight'
    for name, tensor in state.items():
        if name == embedding_key:
            continue
        if name in old:
            if tensor.shape != old[name].shape:
                raise ValueError(f'Warm-start shape mismatch: {name}')
            tensor.copy_(old[name])
            copied.append(name)
        else:
            new.append(name)
    old_ids = {word: i for i, word in enumerate(old_lexicon['words'])}
    pairs = [(i, old_ids[word]) for i, word in enumerate(new_lexicon['words']) if word in old_ids]
    ids_new, ids_old = zip(*pairs)
    state[embedding_key][list(ids_new)] = old[embedding_key][list(ids_old)].to(state[embedding_key].device)
    copied.append(embedding_key)
    missing = set(old) - set(state)
    if missing:
        raise ValueError(f'Parent parameters were not accounted for: {sorted(missing)}')
    report = {
        'parent_epoch': parent['best_epoch'], 'parent_dev': parent['dev'],
        'copied_keys': copied, 'new_keys': new,
        'copied_word_rows': len(pairs), 'new_word_rows': len(new_lexicon['words']) - len(pairs),
        'reused_parameter_elements': sum(old[k].numel() for k in copied),
        'optimizer_and_scheduler': 'new; this is weight transfer, not resume',
    }
    return set(copied), report


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        current = dict(model.named_parameters())
        torch._foreach_lerp_(list(self.shadow.values()), [current[n].detach() for n in self.shadow], 1 - decay)

    def state_dict(self):
        return {'updates': self.updates, 'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state):
        self.updates, self.decay = state['updates'], state['decay']
        for name, value in state['shadow'].items():
            self.shadow[name].copy_(value)

    def model_state(self, model):
        return {name: self.shadow.get(name, value) for name, value in model.state_dict().items()}

    @contextmanager
    @torch.no_grad()
    def average_parameters(self, model):
        parameters = dict(model.named_parameters())
        backup = {name: parameters[name].detach().clone() for name in self.shadow}
        try:
            for name, value in self.shadow.items():
                parameters[name].copy_(value)
            yield
        finally:
            for name, value in backup.items():
                parameters[name].copy_(value)


def optimizer_groups(model, cfg, reused):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('encoder.'):
            family, lr = 'encoder', cfg.encoder_lr
        elif name in reused and name != 'adapter.word_embedding.weight':
            family, lr = 'reused_head', cfg.reused_head_lr
        else:
            family, lr = 'new_head', cfg.head_lr
        decay = cfg.weight_decay if parameter.ndim > 1 and not name.endswith('.bias') else 0.0
        group = groups.setdefault((family, decay), {'params': [], 'name': family,
                                                  'lr': lr, 'weight_decay': decay})
        group['params'].append(parameter)
    return list(groups.values())
