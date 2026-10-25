"""Average aligned fine-tuned weights, then extract dev candidates on two GPUs."""
import gc
import gzip
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'leman_gp_fgm_v10'))
import torch
import torch.distributed as dist
from config import Config
from data import load_split, prepare_split, sha256, write_json
from evaluate import release_loader
from features import collect_features
from model import LemanGP
from runtime import barrier, broadcast_object, environment, rank
from dual_runtime import setup


def make_soups(root, output, policy):
    """Float64 summation and one cast; do not average optimizers or RNG state."""
    accumulator, reference, manifest = {}, None, {}
    for index, name in enumerate(('fgm43', 'fgm44', 'old')):
        source = policy['parents'][name]
        path = root/source['path']
        digest = sha256(path)
        assert digest == source['sha256'], name
        saved = torch.load(path, map_location='cpu', weights_only=False)
        metadata = {k:saved[k] for k in ('config','encoder_config','types','vocab_size',
                    'char_vocab_size','lexicon_sha256','input_identity')}
        if reference is None:
            reference = metadata
        else:
            for key in metadata:
                if key != 'config':
                    assert metadata[key] == reference[key], (name,key)
            # Inference architecture, windowing and vocabulary must be identical.
            for key in policy['architecture_fields']:
                assert metadata['config'][key] == reference['config'][key], (name,key)
        weights = saved['model']
        if index == 0:
            accumulator = {k:v.double() if v.is_floating_point() else v.clone()
                           for k,v in weights.items()}
            dtypes = {k:v.dtype for k,v in weights.items()}
        else:
            assert weights.keys() == accumulator.keys()
            for key, value in weights.items():
                assert value.shape == accumulator[key].shape and value.dtype == dtypes[key]
                if value.is_floating_point():
                    accumulator[key].add_(value.double())
                else:
                    assert torch.equal(value, accumulator[key]), key
        manifest[name] = {'path':str(path),'sha256':digest,
                          'best_epoch':saved['best_epoch'],'weight_source':saved['weight_source']}
        del saved,weights
        gc.collect()
        if index in (1,2):
            recipe = 'soup2' if index == 1 else 'soup3'
            averaged = {k:(v/(index+1)).to(dtypes[k]) if v.is_floating_point() else v.clone()
                        for k,v in accumulator.items()}
            assert all(torch.isfinite(v).all() for v in averaged.values() if v.is_floating_point())
            target = output/(recipe+'.pt')
            metadata = {**reference,'model':averaged,'recipe':recipe,
                        'parents':dict(manifest),'policy_sha256':sha256(HERE/'POLICY.json'),
                        'averaging':'equal arithmetic mean; float64 accumulation; cast to parent dtype',
                        'test_accessed':False}
            torch.save(metadata,target)
            write_json(output/(recipe+'_weights.json'),{
                k:v for k,v in metadata.items() if k not in ('model','encoder_config')})
            del averaged,metadata
            gc.collect()
    del accumulator
    gc.collect()


def main():
    policy = json.loads((HERE/'POLICY.json').read_text())
    requested = Config.from_environment()
    root = Path(requested.root)
    output = root/'output/leman_soup_v16'
    output.mkdir(parents=True,exist_ok=True)
    device = setup()
    try:
        if rank() == 0:
            make_soups(root,output,policy)
        barrier()
        for name in ('soup2','soup3'):
            path = output/(name+'.pt')
            digest = broadcast_object(sha256(path) if rank() == 0 else None)
            ckpt = torch.load(path,map_location='cpu',weights_only=False)
            cfg = Config(**ckpt['config'])
            cfg.root,cfg.data_dir,cfg.encoder_dir = requested.root,requested.data_dir,requested.encoder_dir
            cfg.eval_batch_size,cfg.workers = 16,2
            run = root/'output/leman_fgm_v10_seed43'
            assert sha256(run/'lexicon.json') == ckpt['lexicon_sha256']
            for filename in ('train2.txt','dev2.txt','label.txt'):
                assert sha256(Path(cfg.data_dir)/filename) == ckpt['input_identity'][filename]
            lexicon = json.loads((run/'lexicon.json').read_text())
            cached = broadcast_object(str(prepare_split(cfg,'dev',lexicon,ckpt['types'])) if rank() == 0 else None)
            dataset = load_split(cached)
            model = LemanGP(cfg,len(ckpt['types']),ckpt['vocab_size'],
                            char_vocab_size=ckpt['char_vocab_size'],
                            encoder_config=ckpt['encoder_config'],pretrained=False)
            model.load_state_dict(ckpt['model'],strict=True)
            header = {'types':ckpt['types'],'checkpoint_sha256':digest,'recipe':name,
                      'parents':ckpt['parents'],'dev_sha256':ckpt['input_identity']['dev2.txt'],
                      'policy_sha256':sha256(HERE/'POLICY.json'),'floor':-4.0,
                      'environment':environment(device),'test_accessed':False}
            del ckpt
            gc.collect()
            model.to(device)
            started = time.perf_counter()
            rows = collect_features(model,dataset,cfg,device,output/'shards'/name,floor=-4.0)
            if rank() == 0:
                target = output/(name+'_features.jsonl.gz')
                with gzip.open(target,'wt',encoding='utf-8') as stream:
                    stream.write(json.dumps(header,ensure_ascii=False)+'\n')
                    for sid in range(len(dataset.samples)):
                        stream.write(json.dumps({'sid':sid,'rows':sorted(rows[sid])})+'\n')
                write_json(output/(name+'_extraction.json'),{**header,
                    'seconds':time.perf_counter()-started,'cache_sha256':sha256(target)})
                print(f'{name}: saved {target}',flush=True)
            release_loader(dataset)
            del rows,model,dataset
            gc.collect()
            torch.cuda.empty_cache()
            barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
