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
from runtime import amp_context, barrier, gather_objects, rank, sum_tensor, to_device, world_size


_EVALUATION_LOADERS = {}


def evaluation_loader(dataset, cfg, device):
    if id(dataset) not in _EVALUATION_LOADERS:
        subset = Subset(dataset, list(range(rank(), len(dataset), world_size())))
        generator = torch.Generator().manual_seed(cfg.seed + 7000 + rank())
        kwargs = {'num_workers': cfg.workers, 'pin_memory': device.type == 'cuda',
                  'generator': generator, 'collate_fn': Collator(dataset.types, dataset.pad_id)}
        if cfg.workers:
            kwargs.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context='spawn')
        _EVALUATION_LOADERS[id(dataset)] = DataLoader(subset, batch_size=cfg.eval_batch_size or cfg.batch_size, **kwargs)
    return _EVALUATION_LOADERS[id(dataset)]


def release_loader(dataset):
    loader = _EVALUATION_LOADERS.pop(id(dataset), None)
    if loader is not None and loader._iterator is not None:
        loader._iterator._shutdown_workers()


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
    loader = evaluation_loader(dataset, cfg, device)
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


@torch.inference_mode()
def collect_scores(model, dataset, cfg, device, floor=-4.0, shard_dir=None, score_scale=1.0):
    """One forward; store scored original spans, merging duplicate windows by max."""
    model.eval()
    predictions = defaultdict(dict)
    loader = evaluation_loader(dataset, cfg, device)
    for batch in tqdm(loader, desc='Scored spans', disable=rank() != 0,
                      mininterval=1, dynamic_ncols=True):
        inputs = to_device(batch, device)
        inputs.pop('labels', None)
        with amp_context(device):
            scores = model(**inputs)
        # Bound nonzero-index memory by one entity type rather than materializing
        # [batch, types, chars, chars] candidate indices in one CUDA allocation.
        for typ in range(scores.shape[1]):
            plane = scores[:, typ] * score_scale
            indices = torch.nonzero(plane > floor, as_tuple=False)
            values = plane[tuple(indices.T)].float().cpu().tolist()
            for (row, a, b), score in zip(indices.cpu().tolist(), values):
                sid, offset, length = batch['metadata'][row]
                if 0 <= a <= b < length:
                    key = (typ, offset + a, offset + b + 1)
                    predictions[sid][key] = max(predictions[sid].get(key, -float('inf')), score)
    if shard_dir is not None:
        # Disk shards avoid serializing large scored-span lists onto NCCL GPU buffers.
        import pickle
        shard_dir = Path(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)
        with open(shard_dir / f'rank{rank()}.pkl', 'wb') as stream:
            pickle.dump(dict(predictions), stream, protocol=pickle.HIGHEST_PROTOCOL)
        predictions.clear()
        barrier()
        if rank() != 0:
            return None
        def parts():
            for process in range(world_size()):
                path = shard_dir / f'rank{process}.pkl'
                with open(path, 'rb') as stream:
                    yield pickle.load(stream)
                path.unlink()
        pieces = parts()
    else:
        pieces = gather_objects(dict(predictions))
    merged = defaultdict(dict)
    for part in pieces:
        for sid, spans in part.items():
            for key, score in spans.items():
                merged[sid][key] = max(merged[sid].get(key, -float('inf')), score)
    return {sid: [(*key, score) for key, score in sorted(merged[sid].items())]
            for sid in range(len(dataset.samples))}


def write_predictions(path, samples, predictions, types):
    with open(path, 'w', encoding='utf-8') as f:
        for sid, (text, gold) in enumerate(samples):
            def named(spans):
                return [{'type': types[t], 'start': a, 'end': b, 'text': text[a:b]}
                        for t, a, b in sorted(spans)]
            f.write(json.dumps({'sid': sid, 'text': text, 'gold': named(gold),
                                'predicted': named(predictions.get(sid, []))}, ensure_ascii=False) + '\n')
