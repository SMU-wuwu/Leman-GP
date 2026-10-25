"""GPU inference with the audited v14 seed44 fusion calibrator."""
import argparse
import faulthandler
import gc
import gzip
import json
from pathlib import Path
import sys
import time

ENTRY = Path(__file__).resolve().parent
HERE = ENTRY.parent/'leman_gp_fgm_v14_apply'
sys.path.insert(0, str(HERE.parent/'leman_gp_fgm_v10_analysis'))
from analyze import build_features
from common import PACKAGE, named_predictions, read_json, sha256, verify_package
from calibration_core import apply_logistic, candidates_from_scores
sys.path.append(str(HERE.parent))
from leman_gp_fgm_v10.config import Config
from data import load_split, prepare_split, write_json
from evaluate import release_loader, score_predictions, write_predictions
from features import collect_features
from model import LemanGP
from runtime import barrier, broadcast_object, environment, rank, setup, source_hashes
from search import predict
import torch
import torch.distributed as dist
sys.path.insert(0,str(ENTRY.parent/'leman_gp_soup_v16'))
from dual_runtime import setup


def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev','test'], default='dev')
    args = parser.parse_args()
    verify_package()
    manifest = read_json(HERE/'PACKAGE_MANIFEST.json')
    for name,digest in manifest['files'].items():
        assert sha256(HERE/name) == digest
    for package, digest in manifest['shared_dependencies'].items():
        dependency = HERE.parent/package/'PACKAGE_MANIFEST.json'
        assert sha256(dependency) == digest
        for name, expected in read_json(dependency)['files'].items():
            assert sha256(dependency.parent/name) == expected
    requested = Config.from_environment()
    root = Path(requested.root)
    output = root/'output/leman_fgm_v14_analysis_seed44'
    selection_path = output/'selection.json'
    assert sha256(selection_path) == manifest['selection_sha256']
    selection = read_json(selection_path)
    fitted = selection['fit']
    floor = fitted['candidate_floor']
    names = fitted['component_models']
    expected = selection['checkpoint_sha256']
    runs = {'old':'output/leman_finetune_v4_seed42',
            'fgm43':'output/leman_fgm_v10_seed43',
            'fgm44':'output/leman_fgm_v14_seed44'}
    device = setup()
    try:
        started = time.perf_counter()
        all_records, identities, inference_seconds = [], {}, {}
        samples, types = None, None
        for name in names:
            run = root/runs[name]
            path = run/'best.pt'
            digest = broadcast_object(sha256(path) if rank() == 0 else None)
            assert digest == expected[name]
            checkpoint = torch.load(path,map_location='cpu',weights_only=False)
            expected_source = source_hashes(PACKAGE/'frozen_v4' if name == 'old' else HERE.parent/'leman_gp_fgm_v10')
            assert checkpoint['source_hashes'] == expected_source
            # v10 copied the same model graph and data code from v4.
            for module in ('model.py','paper_core.py','data.py','features.py'):
                assert sha256(PACKAGE/'frozen_v4'/module) == sha256(HERE.parent/'leman_gp_fgm_v10'/module)
            cfg = Config(**checkpoint['config'])
            cfg.root,cfg.data_dir,cfg.encoder_dir = requested.root,requested.data_dir,requested.encoder_dir
            cfg.eval_batch_size,cfg.workers = 16,2
            cfg.output_dir = str(run)
            assert sha256(run/'lexicon.json') == checkpoint['lexicon_sha256']
            for group in ('encoder_files','encoder_weights'):
                for filename,h in checkpoint['input_identity'][group].items():
                    assert sha256(Path(cfg.encoder_dir)/filename) == h
            for filename in ('train2.txt','dev2.txt','label.txt'):
                assert sha256(Path(cfg.data_dir)/filename) == checkpoint['input_identity'][filename]
            lexicon = read_json(run/'lexicon.json')
            cached = broadcast_object(str(prepare_split(cfg,args.split,lexicon,checkpoint['types'])) if rank() == 0 else None)
            dataset = load_split(cached)
            if samples is None:
                samples,types = dataset.samples,dataset.types
            else:
                assert samples == dataset.samples and types == dataset.types
            model = LemanGP(cfg,len(types),checkpoint['vocab_size'],
                            char_vocab_size=checkpoint['char_vocab_size'],
                            encoder_config=checkpoint['encoder_config'],pretrained=False)
            model.load_state_dict(checkpoint['model'],strict=True)
            identities[name] = {'checkpoint_sha256':digest,'best_epoch':checkpoint['best_epoch'],
                                'weight_source':checkpoint['weight_source']}
            del checkpoint
            gc.collect()
            model.to(device)
            began = time.perf_counter()
            all_records.append(collect_features(model,dataset,cfg,device,output/'apply_shards'/name,floor=floor))
            inference_seconds[name] = time.perf_counter()-began
            barrier()
            if rank() == 0:
                cache_dir = output/'dual_replay_cache_20261018'
                cache_dir.mkdir(exist_ok=True)
                with gzip.open(cache_dir/f'{name}.jsonl.gz','wt',encoding='utf-8') as stream:
                    for sid,rows in sorted(all_records[-1].items()):
                        stream.write(json.dumps({'sid':sid,'rows':rows})+'\n')
                print(f'{name}: feature collection complete; releasing GPU model', flush=True)
            release_loader(dataset)
            del model,dataset
            gc.collect()
            torch.cuda.empty_cache()
            barrier()
        if rank() == 0:
            print('All three models complete; applying frozen v14 calibration', flush=True)
            assert types == selection['types']
            gazetteer_path = PACKAGE/'inputs/v6_train_gazetteer.json'
            assert sha256(gazetteer_path) == selection['gazetteer_sha256']
            print('Building calibration features',flush=True)
            keys,values = build_features(all_records,samples,types,read_json(gazetteer_path))
            print(f'Built {len(keys)} calibration rows',flush=True)
            if floor != -4.0:
                present = [{(sid,t,a,b) for sid,rows in source.items() for t,a,b,_,_ in rows}
                           for source in all_records]
                for i,key in enumerate(keys):
                    for model,available in enumerate(present):
                        if key not in available:
                            values[i,6*model] = floor/8.0
            scores = apply_logistic(keys,values,fitted['logistic'])
            print('Calibrated scores; decoding',flush=True)
            predictions = predict(candidates_from_scores(keys,scores,len(samples)),
                                  fitted['decoder'],types,range(len(samples)))
            print('Decoded predictions; comparing historical dev replay',flush=True)
            if args.split == 'dev':
                assert predictions == named_predictions(output/'selected_full_dev_predictions.jsonl',types), 'GPU inference differs from audited cache replay'
            result = score_predictions(samples,predictions,types)
            print('Historical replay identical; recording environment',flush=True)
            result.update(version=14,split=args.split,checkpoint=identities,
                          selection_sha256=sha256(selection_path),family=fitted['family'],
                          candidate_floor=floor,environment=environment(device),
                          test_accessed=args.split == 'test',selection_data='dev',
                          inference_seconds=inference_seconds,
                          total_seconds=time.perf_counter()-started,
                          apply_manifest_sha256=sha256(HERE/'PACKAGE_MANIFEST.json'))
            folder = output/f'{args.split}_evaluation_2gpu_20261018'
            folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/f'{args.split}_results.json',result)
            write_predictions(folder/f'{args.split}_predictions.jsonl',samples,predictions,types)
            print(f'{args.split} v14: Micro={100*result["micro"]["f1"]:.6f}%, '
                  f'Macro13={100*result["macro_f1"]:.6f}%',flush=True)
        barrier()
    finally:
        faulthandler.cancel_dump_traceback_later()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
