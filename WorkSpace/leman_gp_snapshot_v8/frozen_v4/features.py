"""Read the trained auxiliary boundary head without editing the frozen model."""
from collections import defaultdict
from pathlib import Path
import pickle

import torch
from torch.nn import functional as F
from tqdm import tqdm

from evaluate import evaluation_loader
from runtime import amp_context, barrier, rank, to_device, world_size


@torch.inference_mode()
def collect_features(model, dataset, cfg, device, shard_dir, floor=-4.0):
    model.eval()
    scale = cfg.gp_head_size ** 0.5
    state, records = {}, defaultdict(list)

    def read_boundary(module, args):
        # This is precisely the masked sequence given to attention and to the
        # auxiliary head in the training forward. eval() removes dropout RNG.
        with torch.autocast(device_type=device.type, enabled=False):
            boundary = model.boundary(args[0].float()).reshape(*args[0].shape[:2], model.num_types, 2)
            state['log_boundary'] = F.logsigmoid(boundary)

    handle = model.attention.register_forward_pre_hook(read_boundary)
    try:
        loader = evaluation_loader(dataset, cfg, device)
        for batch in tqdm(loader, desc='GP + boundary / dev', disable=rank() != 0,
                          mininterval=1, dynamic_ncols=True):
            inputs = to_device(batch, device)
            inputs.pop('labels', None)
            with amp_context(device):
                scores = model(**inputs)
            log_boundary = state.pop('log_boundary')
            for typ in range(scores.shape[1]):
                plane = scores[:, typ] * scale
                indices = torch.nonzero(plane > floor, as_tuple=False)
                row, a, b = indices.T
                penalties = 0.5 * (log_boundary[row, a, typ, 0] + log_boundary[row, b, typ, 1])
                values = torch.stack((plane[row, a, b], penalties), -1).float().cpu().tolist()
                for (row, a, b), (score, penalty) in zip(indices.cpu().tolist(), values):
                    sid, offset, length = batch['metadata'][row]
                    if 0 <= a <= b < length:
                        records[sid].append((typ, offset + a, offset + b + 1, score, penalty))
    finally:
        handle.remove()
    directory = Path(shard_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / f'rank{rank()}.pkl', 'wb') as stream:
        pickle.dump(dict(records), stream, protocol=pickle.HIGHEST_PROTOCOL)
    barrier()
    if rank() != 0:
        return None
    merged = {sid: [] for sid in range(len(dataset.samples))}
    for process in range(world_size()):
        path = directory / f'rank{process}.pkl'
        with open(path, 'rb') as stream:
            for sid, rows in pickle.load(stream).items():
                merged[sid].extend(rows)
        path.unlink()
    return merged
