"""Small synthetic-only structural/DDP tests. Does not read the task dataset."""
import argparse
import itertools
import json
from pathlib import Path
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertConfig, BertModel, BertTokenizerFast

from config import Config, VARIANTS
from controlled_data import Collator, extract_spans, label_names
from model import LemanGP
from parent_model import LemanGP as Parent
from paper_core import GlobalPointerLoss
from crf_reference import LinearCRF
from adversarial import fgm_backward
from runtime import seed_everything, setup, to_device, gradient_norm_fp64, amp_context


def tiny_config(variant='F'):
    c = Config.make('/tmp/synthetic_only', variant, 42, 'C')
    for k, v in dict(word_dim=8, boundary_dim=4, char_dim=8, gru_hidden=8, gru_layers=2,
                     attention_heads=2, gp_head_size=8, max_chars=16, overlap_chars=4,
                     max_span=16, gp_type_weights=[1., 1.]).items():
        setattr(c, k, v)
    return c


def encoder_config():
    return BertConfig(vocab_size=32, hidden_size=32, num_hidden_layers=1,
                      num_attention_heads=2, intermediate_size=48, max_position_embeddings=64).to_dict()


def items(words=4):
    result = []
    for sid, n in enumerate([5, 3]):
        result.append(dict(sid=sid, offset=0, length=n, input_ids=list(range(5, 5+n)),
            offsets=[(i, i+1) for i in range(n)], char_ids=list(range(2, 2+n)),
            word_ids=np.ones((n, words), dtype=np.int32)*2,
            boundary_ids=np.ones((n, words), dtype=np.int8)*3, spans=[(0, 0, 2), (1, n-1, n)]))
    return result


def structural():
    torch.set_num_threads(2)
    c = tiny_config()
    seed_everything(42)
    parent = Parent(c, 2, 10, char_vocab_size=12, encoder_config=encoder_config(), pretrained=False)
    rng = torch.get_rng_state().clone()
    seed_everything(42)
    model = LemanGP(c, 2, 10, char_vocab_size=12, encoder_config=encoder_config(), pretrained=False)
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in parent.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k
    batch = to_device(Collator(['A', 'B'], 0, 'F')(items()), torch.device('cpu'))
    max_gradient_difference = 0.0
    for training in [False, True]:
        parent.train(training); model.train(training)
        parent.zero_grad(); model.zero_grad()
        seed_everything(121)
        a, scores_a = parent(**batch)
        seed_everything(121)
        b, scores_b = model(**batch)
        assert torch.equal(scores_a, scores_b) and torch.equal(a, b)
        a.backward(); b.backward()
        for (k, p), (j, q) in zip(parent.named_parameters(), model.named_parameters()):
            assert k == j
            if p.grad is None:
                assert q.grad is None, k
            else:
                # The copied TorchScript GRU can switch its profiled execution
                # plan between the first two calls; compare gradients in FP32 tolerance.
                torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5, msg=k)
                max_gradient_difference = max(max_gradient_difference, float((p.grad-q.grad).abs().max()))
    records = {}
    full_state = parent.state_dict()
    for variant in VARIANTS:
        seed_everything(42)
        m = LemanGP(tiny_config(variant), 2, 10, char_vocab_size=12, encoder_config=encoder_config(), pretrained=False)
        assert torch.equal(rng, torch.get_rng_state()), variant
        shared = 0
        for key, value in m.state_dict().items():
            if key in full_state and value.shape == full_state[key].shape:
                assert torch.equal(value, full_state[key]), (variant, key)
                shared += 1
        x = to_device(Collator(['A', 'B'], 0, variant)(items(m.cfg.max_words)), torch.device('cpu'))
        loss, _ = m(**x); loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(gradient_norm_fp64(m.parameters())), variant
        assert all(p.grad is not None for p in m.parameters() if p.requires_grad), variant
        records[variant] = {'shared_tensors_equal': shared, 'loss_finite': True, 'all_parameters_used': True}
    logits = torch.randn(2, 2, 5, 5)
    from parent_model import RecallGlobalPointerLoss
    assert torch.allclose(GlobalPointerLoss()(logits, batch['labels'], batch['char_mask']),
                          RecallGlobalPointerLoss([1., 1.], 1.)(logits, batch['labels'], batch['char_mask']))
    crf = LinearCRF(3)
    emissions = torch.randn(1, 3, 3)
    tags = torch.tensor([[1, 2, 0]])
    mask = torch.tensor([[True, True, False]])
    with torch.no_grad():
        crf.transitions.copy_(torch.randn(3, 3)*.2)
        crf.start.copy_(torch.randn(3)*.2); crf.end.copy_(torch.randn(3)*.2)
    paths = list(itertools.product(range(3), repeat=2))
    values = torch.stack([crf.start[a]+emissions[0, 0, a]+crf.transitions[a,b]+emissions[0, 1, b]+crf.end[b] for a,b in paths])
    expected = torch.logsumexp(values, 0)-values[paths.index((1, 2))]
    assert torch.allclose(crf(emissions, tags, mask), expected, atol=1e-6)
    assert crf.decode(emissions, mask)[0] == list(paths[int(values.argmax())])
    packed = Collator(['A', 'B'], 0, 'crf')(items())
    labels = label_names(['A', 'B'])
    assert set(extract_spans([labels[k] for k in packed['crf_tags'][0].tolist()])) == {('A', 0, 2), ('B', 4, 5)}
    return {'full_vs_parent_eval_train_logits_loss_exact': True,
            'full_vs_parent_gradients_close': {'atol': 1e-7, 'rtol': 1e-5, 'passed': True},
            'max_gradient_absolute_difference': max_gradient_difference,
            'initialization_rng_equal_all_variants': True, 'variants': records,
            'A_loss_is_standard_MLCCE': True, 'CRF_bruteforce_and_BIOES_roundtrip': True}


def ddp():
    device = setup()
    records = {}
    try:
        for variant in VARIANTS:
            seed_everything(42)
            model = LemanGP(tiny_config(variant), 2, 10, char_vocab_size=12,
                            encoder_config=encoder_config(), pretrained=False).to(device)
            wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False)
            seed_everything(100 + dist.get_rank())
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
            batch = to_device(Collator(['A', 'B'], 0, variant)(items(model.cfg.max_words)), device)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss, stats = fgm_backward(wrapped, batch, epsilon=.5, clean_weight=.5,
                                           autocast=lambda: amp_context(device))
                norm = gradient_norm_fp64(model.parameters())
                assert torch.isfinite(loss) and torch.isfinite(norm), variant
                optimizer.step()
            # All-reduced updates leave both ranks' parameters identical.
            for parameter in model.parameters():
                reference = parameter.detach().clone()
                dist.broadcast(reference, src=0)
                assert torch.equal(parameter, reference), variant
            records[variant] = {'two_FGM_steps_passed': True, 'loss': float(loss),
                                'ranks_parameters_equal': True, 'bf16': True}
            del wrapped, optimizer, model
            torch.cuda.empty_cache()
        return records
    finally:
        dist.destroy_process_group()


def fixture(root):
    root.mkdir(parents=True, exist_ok=False)
    (root/'SYNTHETIC_QA_ONLY').touch()
    (root/'data').mkdir(); (root/'model').mkdir()
    chars = list('甲乙丙丁戊己庚辛壬癸农业植物病害')
    (root/'model/vocab.txt').write_text('\n'.join(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]']+chars)+'\n')
    tok = BertTokenizerFast(vocab_file=str(root/'model/vocab.txt'), do_lower_case=False)
    tok.save_pretrained(root/'model')
    seed_everything(20261020)
    enc = BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=48, max_position_embeddings=64)
    BertModel(enc).save_pretrained(root/'model')
    def samples(n):
        blocks = []
        for i in range(n):
            letters = (chars[i % 10:i % 10+6] + chars[:4])[:8]
            tags = ['B-A','I-A','O','B-B','O','O','O','O']
            blocks.append('\n'.join(f'{c} {t}' for c,t in zip(letters, tags)))
        return '\n\n'.join(blocks)+'\n\n'
    (root/'data/train2.txt').write_text(samples(13))
    (root/'data/dev2.txt').write_text(samples(5))
    (root/'data/label.txt').write_text('O\nB-A\nI-A\nB-B\nI-B\n')
    overrides = dict(epochs=2, batch_size=2, eval_batch_size=2, workers=0, preprocessing_workers=1,
                     word_dim=8, boundary_dim=4, char_dim=8, gru_hidden=8, gru_layers=2,
                     attention_heads=2, gp_head_size=8, max_chars=16, overlap_chars=4,
                     max_span=16, save_every_steps=1, save_every_epochs=1)
    (root/'qa_config.json').write_text(json.dumps(overrides, indent=2)+'\n')
    return {'root': str(root), 'real_task_data_read': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['structural', 'ddp', 'fixture'])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = fixture(args.output) if args.mode == 'fixture' else structural() if args.mode == 'structural' else ddp()
    if args.mode != 'fixture' and int(__import__('os').environ.get('RANK', '0')) == 0:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
