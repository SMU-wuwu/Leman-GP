"""GPU inference with the audited FGM single/fusion calibrator."""
import argparse
import gc
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, choices=[10,11], default=10)
    parser.add_argument('--split', choices=['dev','test'], default='dev')
    args = parser.parse_args()
    verify_package()
    manifest = read_json(HERE/'PACKAGE_MANIFEST.json')
    for name,digest in manifest['files'].items():
        assert sha256(HERE/name) == digest
    requested = Config.from_environment()
    root = Path(requested.root)
    output = root/('output/leman_fgm_v10_analysis_seed43' if args.version == 10
                   else 'output/leman_candidates_v11_seed43')
    selection_path = output/'selection.json'
    assert sha256(selection_path) == manifest['selection_sha256'][str(args.version)]
    selection = read_json(selection_path)
    fitted = selection['fit']
    floor = selection.get('candidate_floor', -4.0)
    names = ['fgm'] if fitted['family'] in ('fgm','expanded_fgm') else ['v4','fgm']
    expected = {'v4':selection.get('parent_checkpoint_sha256'),
                'fgm':selection.get('checkpoint',{}).get('checkpoint_sha256')}
    expected.update(selection.get('checkpoint_sha256',{}))
    device = setup()
    try:
        all_records, identities = [], {}
        samples, types = None, None
        for name in names:
            run = root/('output/leman_finetune_v4_seed42' if name == 'v4' else 'output/leman_fgm_v10_seed43')
            path = run/'best.pt'
            digest = broadcast_object(sha256(path) if rank() == 0 else None)
            assert digest == expected[name]
            checkpoint = torch.load(path,map_location='cpu',weights_only=False)
            expected_source = source_hashes(PACKAGE/'frozen_v4' if name == 'v4' else HERE.parent/'leman_gp_fgm_v10')
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
            all_records.append(collect_features(model,dataset,cfg,device,output/'apply_shards'/name,floor=floor))
            release_loader(dataset)
            del model,dataset
            gc.collect()
            torch.cuda.empty_cache()
        if rank() == 0:
            assert types == selection['types']
            gazetteer_path = PACKAGE/'inputs/v6_train_gazetteer.json'
            assert sha256(gazetteer_path) == selection['gazetteer_sha256']
            keys,values = build_features(all_records,samples,types,read_json(gazetteer_path))
            if floor != -4.0:
                present = [{(sid,t,a,b) for sid,rows in source.items() for t,a,b,_,_ in rows}
                           for source in all_records]
                for i,key in enumerate(keys):
                    for model,available in enumerate(present):
                        if key not in available:
                            values[i,6*model] = floor/8.0
            scores = apply_logistic(keys,values,fitted['logistic'])
            predictions = predict(candidates_from_scores(keys,scores,len(samples)),
                                  fitted['decoder'],types,range(len(samples)))
            if args.split == 'dev':
                assert predictions == named_predictions(output/'selected_full_dev_predictions.jsonl',types), 'GPU inference differs from audited cache replay'
            result = score_predictions(samples,predictions,types)
            result.update(version=args.version,split=args.split,checkpoint=identities,
                          selection_sha256=sha256(selection_path),family=fitted['family'],
                          candidate_floor=floor,environment=environment(device),
                          test_accessed=args.split == 'test',selection_data='dev')
            folder = output/f'{args.split}_evaluation'
            folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/f'{args.split}_results.json',result)
            write_predictions(folder/f'{args.split}_predictions.jsonl',samples,predictions,types)
            print(f'{args.split} v{args.version}: Micro={100*result["micro"]["f1"]:.6f}%, '
                  f'Macro13={100*result["macro_f1"]:.6f}%',flush=True)
        barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
