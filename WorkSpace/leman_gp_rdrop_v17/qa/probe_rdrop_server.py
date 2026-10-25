"""Two-rank R-Drop backward and memory check. Never updates or saves weights."""
import gc
import json
from pathlib import Path
import sys

ROOT = Path('/root/data/WorkSpace')
sys.path.insert(0,str(ROOT/'leman_gp_rdrop_v17'))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from config import Config
from consistency import rdrop_backward
from data import load_split, write_json
from model import LemanGP
from recipe import training_recipe
from runtime import (amp_context, broadcast_object, environment, gather_objects,
                     gradient_norm_fp64, rank, rng_state, restore_rng, seed_everything, setup)


def main():
    device = setup()
    try:
        cfg = Config.from_environment()
        parent = torch.load(cfg.warm_start,map_location='cpu',weights_only=False)
        cache = sorted((Path(cfg.warm_start).parent/'cache').glob('train_*.pkl'))
        assert len(cache) == 1, cache
        dataset = load_split(cache[0])
        cfg.gp_type_weights = training_recipe(dataset.samples,parent['types'],cfg,
                    parent['input_identity']['train2.txt'])['ordered_weights']
        seed_everything(cfg.seed)
        model = LemanGP(cfg,len(parent['types']),parent['vocab_size'],
                char_vocab_size=parent['char_vocab_size'],
                encoder_config=parent['encoder_config'],pretrained=False)
        model.load_state_dict(parent['model'],strict=True)
        model.to(device)
        del parent
        gc.collect()
        wrapped = DDP(model,device_ids=[device.index],output_device=device.index,
                    broadcast_buffers=False,gradient_as_bucket_view=True)
        # Three persistent copies reserve Adam moments and EMA, in addition to
        # the real DDP buckets and gradients created by these backward passes.
        reserve = [torch.empty_like(p) for p in model.parameters() if p.requires_grad for _ in range(3)]
        bs = cfg.batch_size
        tokens,chars = dataset.stats['max_encoder_tokens_observed'],dataset.stats['max_chars_observed']
        batch = {
            'input_ids':torch.full((bs,tokens),dataset.items[0]['input_ids'][0],device=device,dtype=torch.long),
            'token_mask':torch.ones(bs,tokens,device=device,dtype=torch.bool),
            'char_pool':torch.zeros(bs,chars,tokens,device=device),
            'char_mask':torch.ones(bs,chars,device=device,dtype=torch.bool),
            'char_ids':torch.ones(bs,chars,device=device,dtype=torch.long),
            'word_ids':torch.ones(bs,chars,cfg.max_words,device=device,dtype=torch.long),
            'boundary_ids':torch.zeros(bs,chars,cfg.max_words,device=device,dtype=torch.long),
            'lengths':torch.full((bs,),chars,dtype=torch.long),
            'labels':torch.zeros(bs,len(dataset.types),chars,chars,device=device,dtype=torch.bool),
        }
        char_indices = torch.arange(chars,device=device)
        batch['char_pool'][:,char_indices,char_indices.clamp_max(tokens-1)] = 1
        batch['labels'][:,0,0,0] = True
        seed_everything(cfg.seed+rank())
        state = rng_state()
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        with wrapped.no_sync():
            rdrop_backward(wrapped,batch,weight=cfg.consistency_weight,
                    loss_scale=0.5,autocast=lambda:amp_context(device))
        loss,stats = rdrop_backward(wrapped,batch,weight=cfg.consistency_weight,
                    loss_scale=0.5,autocast=lambda:amp_context(device))
        norm = gradient_norm_fp64(model.parameters())
        assert torch.isfinite(loss) and torch.isfinite(norm)
        torch.cuda.synchronize()
        report = {'rank':rank(),'loss':float(loss),'gradient_norm':float(norm),
            'stats':stats,'peak_gib':torch.cuda.max_memory_allocated(device)/2**30,
            'reserved_state_gib':sum(p.numel()*p.element_size() for p in reserve)/2**30,
            'original_examples_per_rank':bs,'forward_batch_size':2*bs,
            'probe_tokens':tokens,'probe_chars':chars,'optimizer_steps':0}
        rows = gather_objects(report)
        assert max(r['gradient_norm'] for r in rows)-min(r['gradient_norm'] for r in rows) < 1e-7
        restore_rng(state)
        if rank() == 0:
            target = ROOT/'output/leman_soup_v16/rdrop_probe.json'
            write_json(target,{'status':'PASS','environment':environment(device),'ranks':rows,
                'model_weights_updated':False,'formal_training_started':False,'test_accessed':False})
            print(json.dumps(rows),flush=True)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()
