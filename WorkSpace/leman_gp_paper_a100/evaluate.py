"""One GPU forward per batch; exact character spans, including overlaps."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import time

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import Collator, write_json
from runtime import amp_context, gather_objects, rank, sum_tensor, to_device, world_size


def metric(counts):
    tp, pred, gold = counts
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * tp / (pred + gold) if pred + gold else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'true_positive': int(tp), 'predicted': int(pred), 'support': int(gold)}


def score_predictions(samples, predictions, types):
    counts = [[0, 0, 0] for _ in types]
    for sid, (_, truth) in enumerate(samples):
        gold, pred = set(map(tuple, truth)), set(predictions.get(sid, []))
        for typ, _, _ in pred & gold:
            counts[typ][0] += 1
        for typ, _, _ in pred:
            counts[typ][1] += 1
        for typ, _, _ in gold:
            counts[typ][2] += 1
    per_type = {name: metric(counts[i]) for i, name in enumerate(types)}
    micro = metric([sum(c[i] for c in counts) for i in range(3)])
    return {'micro': micro, 'macro_f1': sum(m['f1'] for m in per_type.values()) / len(types),
            'per_type': per_type, 'total_samples': len(samples),
            'metric': 'exact original character-span + entity type; inclusive start, exclusive end',
            'overlap_policy': 'retain all; deduplicate identical spans from overlapping windows'}


@torch.inference_mode()
def evaluate(model, dataset, cfg, device, split='dev', output_dir=None):
    model.eval()
    # No DistributedSampler padding: every window is evaluated exactly once.
    subset = Subset(dataset, list(range(rank(), len(dataset), world_size())))
    generator = torch.Generator().manual_seed(cfg.seed + 7000 + rank())
    kwargs = {'num_workers': cfg.workers, 'pin_memory': device.type == 'cuda',
              'generator': generator, 'collate_fn': Collator(dataset.types, dataset.pad_id)}
    if cfg.workers:
        kwargs.update(persistent_workers=False, prefetch_factor=2, multiprocessing_context='spawn')
    loader = DataLoader(subset, batch_size=cfg.eval_batch_size or cfg.batch_size, **kwargs)
    predictions = defaultdict(set)
    loss_sum, windows, start = 0.0, 0, time.perf_counter()
    bar = tqdm(loader, desc=f'{split} GPU {rank()}', disable=rank() != 0, mininterval=1, dynamic_ncols=True)
    for batch in bar:
        metadata = batch['metadata']
        with amp_context(device):
            loss, scores = model(**to_device(batch, device))
        # One batched transfer. No Python scalar access to CUDA label matrices.
        candidates = torch.nonzero(scores > cfg.threshold, as_tuple=False).cpu().tolist()
        for row, typ, a, b in candidates:
            sid, offset, length = metadata[row]
            if 0 <= a <= b < length:
                predictions[sid].add((typ, offset + a, offset + b + 1))
        loss_sum += float(loss) * len(metadata)
        windows += len(metadata)
    parts = gather_objects(dict(predictions))
    total_loss, total_windows = sum_tensor([loss_sum, windows], device)
    merged = defaultdict(set)
    for part in parts:
        for sid, spans in part.items():
            merged[sid].update(spans)
    result = score_predictions(dataset.samples, merged, dataset.types)
    result.update({'loss': total_loss / max(1, total_windows), 'windows': int(total_windows),
                   'seconds': time.perf_counter() - start, 'threshold': cfg.threshold,
                   'coverage': dataset.stats, 'world_size': world_size(),
                   'device': str(device), 'amp': 'bf16' if device.type == 'cuda' else 'fp32'})
    if rank() == 0 and output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / f'{split}_results.json', result)
        with open(output / 'entity_metrics.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['entity_type', 'precision', 'recall', 'f1',
                                                  'support', 'true_positive', 'predicted'])
            writer.writeheader()
            for name, metrics in result['per_type'].items():
                writer.writerow({'entity_type': name, **metrics})
        with open(output / f'{split}_predictions.jsonl', 'w', encoding='utf-8') as f:
            for sid, (text, gold) in enumerate(dataset.samples):
                def named(spans):
                    return [{'type': dataset.types[t], 'start': a, 'end': b, 'text': text[a:b]}
                            for t, a, b in sorted(spans)]
                f.write(json.dumps({'sid': sid, 'text': text, 'gold': named(gold),
                                    'predicted': named(merged[sid])}, ensure_ascii=False) + '\n')
    return result
