"""GP original exact spans; CRF window emissions merged before sentence decoding."""
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from controlled_data import Collator, label_names, extract_spans
from data import write_json
from evaluate import evaluate as gp_evaluate, score_predictions, write_predictions
from runtime import rank, world_size, amp_context, to_device, gather_objects, sum_tensor


@torch.inference_mode()
def evaluate(model, dataset, cfg, device, split='dev', output_dir=None):
    if cfg.variant != 'crf':
        return gp_evaluate(model, dataset, cfg, device, split, output_dir)
    model.eval()
    started = time.perf_counter()
    # Partition by full sentence. All of a sentence's windows stay on one rank.
    ids = list(range(rank(), len(dataset.samples), world_size()))
    selected = set(ids)
    window_ids = [i for i, item in enumerate(dataset.items) if item['sid'] in selected]
    generator = torch.Generator().manual_seed(cfg.seed + 7000 + rank())
    loader = DataLoader(Subset(dataset, window_ids), batch_size=cfg.eval_batch_size,
                        collate_fn=Collator(dataset.types, dataset.pad_id, 'crf'),
                        num_workers=0, pin_memory=device.type == 'cuda', generator=generator)
    labels = label_names(dataset.types)
    totals = {sid: np.zeros((len(dataset.samples[sid][0]), len(labels)), dtype=np.float32) for sid in ids}
    counts = {sid: np.zeros(len(dataset.samples[sid][0]), dtype=np.int32) for sid in ids}
    loss_sum = windows = 0
    for batch in loader:
        with amp_context(device):
            loss, emissions = model(**to_device(batch, device))
        values = emissions.float().cpu().numpy()
        for i, (sid, offset, n) in enumerate(batch['metadata']):
            totals[sid][offset:offset+n] += values[i, :n]
            counts[sid][offset:offset+n] += 1
        loss_sum += float(loss) * len(batch['metadata'])
        windows += len(batch['metadata'])
    predictions = {}
    order = sorted(ids, key=lambda sid: len(totals[sid]))
    mapping = {t: i for i, t in enumerate(dataset.types)}
    for start in range(0, len(order), cfg.eval_batch_size):
        group = order[start:start+cfg.eval_batch_size]
        length = max(len(totals[sid]) for sid in group)
        emissions = torch.zeros(len(group), length, len(labels), device=device)
        mask = torch.zeros(len(group), length, dtype=torch.bool, device=device)
        for i, sid in enumerate(group):
            assert np.all(counts[sid] > 0), sid
            n = len(totals[sid])
            emissions[i, :n] = torch.from_numpy(totals[sid]/counts[sid][:, None]).to(device)
            mask[i, :n] = True
        for sid, tags in zip(group, model.crf.decode(emissions, mask)):
            predictions[sid] = {(mapping[t], a, b) for t, a, b in extract_spans([labels[k] for k in tags])}
    merged = {}
    for part in gather_objects(predictions):
        merged.update(part)
    loss_sum, windows = sum_tensor([loss_sum, windows], device)
    result = score_predictions(dataset.samples, merged, dataset.types)
    result.update(loss=loss_sum/max(1, windows), windows=int(windows), seconds=time.perf_counter()-started,
                  overlap_policy='Average window emissions; full-sentence BIOES CRF Viterbi',
                  coverage=dataset.stats, world_size=world_size(), device=str(device))
    if rank() == 0 and output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/f'{split}_results.json', result)
        write_predictions(output/f'{split}_predictions.jsonl', dataset.samples, merged, dataset.types)
    return result
