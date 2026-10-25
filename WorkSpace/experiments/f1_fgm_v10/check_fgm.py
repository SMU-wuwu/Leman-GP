"""Focused checks of the production FGM backward and two-rank gradient average."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'leman_gp_fgm_v10'))
from adversarial import EmbeddingPerturbation, fgm_backward
from config import Config


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(19, 5, padding_idx=0)

    def get_input_embeddings(self):
        return self.embedding


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(5, 2)

    def forward(self, input_ids, token_mask, targets):
        hidden = self.encoder.embedding(input_ids) * token_mask.unsqueeze(-1)
        score = self.head(hidden.sum(1) / token_mask.sum(1, keepdim=True))
        return (score - targets).square().mean(), score


def batch_for(rank):
    return {'input_ids': torch.tensor([[rank + 1, 4, 0], [5, 6 + rank, 8]]),
            'token_mask': torch.tensor([[True, True, False], [True, True, True]]),
            'targets': torch.tensor([[0.2, 0.9], [-0.5, 0.3]], dtype=torch.float64)}


def reference(model, batch, epsilon, weight):
    """Independent autograd gradients, without the production perturbation helper."""
    parameters = tuple(model.parameters())
    loss, _ = model(**batch)
    clean = torch.autograd.grad(loss, parameters)
    indices = torch.unique(batch['input_ids'][batch['token_mask']])
    direction = clean[0][indices]
    original = model.encoder.embedding.weight.detach().clone()
    with torch.no_grad():
        model.encoder.embedding.weight[indices] += epsilon * direction / direction.norm()
    adv_loss, _ = model(**batch)
    adv = torch.autograd.grad(adv_loss, parameters)
    with torch.no_grad():
        model.encoder.embedding.weight.copy_(original)
    return [weight * a + (1 - weight) * b for a, b in zip(clean, adv)], float(
        weight * loss.detach() + (1 - weight) * adv_loss.detach())


def ddp_worker(rank, init_method, result_folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=init_method, rank=rank, world_size=2)
    try:
        torch.manual_seed(104)
        model = TinyModel().double()
        wrapped = DDP(model, gradient_as_bucket_view=True, broadcast_buffers=False)
        original = deepcopy(model.state_dict())
        maximum = 0.0
        for _ in range(2):
            wrapped.zero_grad(set_to_none=True)
            independent = TinyModel().double()
            independent.load_state_dict(original)
            expected, expected_loss = reference(independent, batch_for(rank), 0.5, 0.5)
            for grad in expected:
                dist.all_reduce(grad)
                grad /= 2
            objective, report = fgm_backward(wrapped, batch_for(rank), epsilon=0.5,
                                             clean_weight=0.5, autocast=nullcontext)
            assert abs(float(objective) - expected_loss) < 1e-12
            assert abs(report['perturbation_l2'] - 0.5) < 1e-12
            for parameter, grad in zip(model.parameters(), expected):
                maximum = max(maximum, float((parameter.grad - grad).abs().max()))
                torch.testing.assert_close(parameter.grad, grad, rtol=1e-11, atol=1e-12)
            assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
        Path(result_folder, f'rank{rank}.json').write_text(json.dumps(
            {'rank': rank, 'max_gradient_difference': maximum, 'repeated_steps': 2}))
    finally:
        dist.destroy_process_group()


def local_checks():
    torch.manual_seed(81)
    model = TinyModel().double()
    batch = batch_for(0)
    model(**batch)[0].backward()
    before = model.encoder.embedding.weight.detach().clone()
    state = torch.get_rng_state().clone()
    attack = EmbeddingPerturbation(model.encoder.embedding, batch['input_ids'],
                                   batch['token_mask'], 0.5)
    try:
        with attack:
            delta = model.encoder.embedding.weight - before
            assert abs(float(delta.detach().norm()) - 0.5) < 1e-12
            inactive = sorted(set(range(19)) - set(attack.indices.tolist()))
            assert torch.count_nonzero(delta[inactive]).item() == 0
            raise RuntimeError('intentional interruption')
    except RuntimeError as error:
        assert str(error) == 'intentional interruption'
    assert torch.equal(before, model.encoder.embedding.weight)
    assert torch.equal(state, torch.get_rng_state())
    model.encoder.embedding.weight.grad.zero_()
    zero = EmbeddingPerturbation(model.encoder.embedding, batch['input_ids'], batch['token_mask'], 0.5)
    with zero:
        assert torch.equal(before, model.encoder.embedding.weight)
        assert float(zero.actual_delta_norm) == 0
    cfg = Config.from_environment()
    assert (cfg.seed, cfg.epochs, cfg.batch_size, cfg.encoder_lr, cfg.save_every_epochs) == (43, 3, 32, 3e-6, 5)
    assert cfg.output_dir.endswith('leman_fgm_v10_seed43')
    assert cfg.warm_start.endswith('leman_finetune_v4_seed42/best.pt')
    for name in ('model.py', 'paper_core.py', 'data.py', 'evaluate.py', 'runtime.py', 'recipe.py', 'transfer.py'):
        assert (ROOT/'leman_gp_fgm_v10'/name).read_bytes() == (ROOT/'leman_gp_finetune_v4'/name).read_bytes()
    return {'norm_and_active_rows': 'PASS', 'exact_restore_on_exception': 'PASS',
            'zero_gradient': 'PASS', 'perturbation_rng_unchanged': 'PASS',
            'config_and_unchanged_graph': 'PASS'}


if __name__ == '__main__':
    results = local_checks()
    with tempfile.TemporaryDirectory(prefix='leman_fgm_check_') as folder:
        init_method = 'file://' + str(Path(folder, 'gloo_init'))
        mp.spawn(ddp_worker, args=(init_method, folder), nprocs=2, join=True)
        results['two_rank_gloo_gradient_reference'] = [json.loads(Path(folder, f'rank{rank}.json').read_text())
                                                        for rank in range(2)]
    results['status'] = 'PASS'
    results['neural_training_started'] = False
    results['torch'] = torch.__version__
    Path(__file__).with_name('validation.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
