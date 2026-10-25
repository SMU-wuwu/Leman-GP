"""Add BIOES targets to the copied character/span collation for the CRF variant."""
import torch
from data import Collator as SpanCollator
from crf_data_reference import label_names, extract_spans


class Collator(SpanCollator):
    def __init__(self, types, pad_id, variant):
        super().__init__(types, pad_id)
        self.variant = variant

    def __call__(self, items):
        batch = super().__call__(items)
        if self.variant == 'crf':
            tags = torch.zeros_like(batch['char_ids'])
            for i, item in enumerate(items):
                occupied = set()
                for t, a, b in sorted(item['spans']):
                    positions = set(range(a, b))
                    if occupied & positions:
                        raise ValueError('Flat BIOES CRF cannot represent overlapping training entities')
                    occupied.update(positions)
                    base = 1+4*t
                    if b-a == 1:
                        tags[i, a] = base+3
                    else:
                        tags[i, a] = base
                        tags[i, a+1:b-1] = base+1
                        tags[i, b-1] = base+2
            batch['crf_tags'] = tags
        return batch
