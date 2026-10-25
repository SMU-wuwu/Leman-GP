"""Apply the frozen dev calibrator using the unchanged v4 GPU forward graph."""
import argparse
import csv
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'leman_gp_typecal_v9'))

import torch
import torch.distributed as dist

from common import PACKAGE, paths, read_json, sha256, verify_package
from calibration_core import apply_logistic, candidates_from_scores
from gazetteer import make_features
from config import Config
from data import write_json
from evaluate import release_loader, score_predictions, write_predictions
from features import collect_features
from inference import load_candidate
from runtime import barrier, broadcast_object, environment, rank, setup, world_size
from search import predict, scored_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    args = parser.parse_args()
    root, output = paths()
    manifest = verify_package()
    assert sha256(PACKAGE/'PACKAGE_MANIFEST.json') == '9feb7005d4de355bc4419536be8416dc4c2c88721fe0527e5975cc7769aefd97'
    assert sha256(output/'selection.json') == 'cc251754565f385a937fc6f80b7591c642a0ad7909ddaefeb307003b92f1305a'
    selection = read_json(output / 'selection.json')
    assert selection['identity']['manifest_sha256'] == sha256(PACKAGE / 'PACKAGE_MANIFEST.json')
    requested = Config.from_environment()
    requested.output_dir = str(root / 'output/leman_finetune_v4_seed42')
    requested.warm_start = str(root / 'output/leman_f1_4x4090_seed42/best.pt')
    if sha256(Path(requested.data_dir) / 'dev2.txt') != selection['dev_sha256']:
        raise ValueError('Keep the dev2.txt used to freeze this calibrator.')
    device = setup()
    try:
        names = ['upgraded']
        all_records, identities = {}, {}
        reference_samples, reference_types = None, None
        for candidate in names:
            path = Path(requested.warm_start) if candidate == 'parent' else Path(requested.output_dir) / 'best.pt'
            actual_sha = broadcast_object(sha256(path) if rank() == 0 else None)
            if actual_sha != manifest['checkpoint_sha256'][candidate]:
                raise ValueError(f'Use the completed {candidate} best.pt; the weight SHA256 changed.')
            model, dataset, cfg, info = load_candidate(requested, candidate, args.split, device)
            if reference_samples is None:
                reference_samples, reference_types = dataset.samples, dataset.types
            else:
                assert reference_samples == dataset.samples and reference_types == dataset.types
            if rank() == 0:
                print(f'{candidate}: {world_size()} CUDA GPUs, BF16, {cfg.eval_batch_size}/GPU; '
                      f'{torch.cuda.get_device_name(device)}', flush=True)
            all_records[candidate] = collect_features(
                model, dataset, cfg, device, output / 'cache' / f'{args.split}_{candidate}', floor=-4.0)
            identities[candidate] = info
            release_loader(dataset)
            del model, dataset
            gc.collect()
            torch.cuda.empty_cache()
        if rank() == 0:
            assert reference_types == selection['types']
            gazetteer_path = PACKAGE/'inputs/v6_train_gazetteer.json'
            assert sha256(gazetteer_path) == selection['gazetteer_sha256']
            gazetteer = read_json(gazetteer_path)
            fitted = selection['fit']
            keys, values = make_features(all_records['upgraded'],
                [text for text, _ in reference_samples], reference_types, 'train_gazetteer', gazetteer)
            scores = apply_logistic(keys, values, fitted['logistic'])
            candidates = candidates_from_scores(keys, scores, len(reference_samples))
            decoder = fitted['decoder']
            predictions = predict(candidates, decoder, reference_types, range(len(reference_samples)))
            if args.split == 'dev':
                from common import named_predictions
                expected = named_predictions(output/'selected_full_dev_predictions.jsonl', reference_types)
                assert predictions == expected, 'GPU inference differs from audited cache replay'
            result = score_predictions(reference_samples, predictions, reference_types)
            result.update(method=selection['method'], split=args.split, checkpoint=identities,
                          frozen_selection_sha256=sha256(output / 'selection.json'),
                          decoder=decoder, feature_family=fitted['family'],
                          gazetteer_sha256=selection['gazetteer_sha256'], environment=environment(device),
                          data_sha256=sha256(Path(requested.data_dir) / f'{args.split}2.txt'),
                          test_accessed=args.split == 'test', metric_selection_data='dev only',
                          overlap_policy='retain all' if decoder['strategy'] == 'all'
                          else 'optimal non-overlapping intervals')
            folder = output / f'{args.split}_evaluation'
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / f'{args.split}_results.json', result)
            write_predictions(folder / f'{args.split}_predictions.jsonl', reference_samples,
                              predictions, reference_types)
            with (folder / 'entity_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['entity_type', 'precision', 'recall', 'f1',
                                                            'support', 'true_positive', 'predicted'])
                writer.writeheader()
                for name, row in result['per_type'].items():
                    writer.writerow({'entity_type': name, **row})
            print(f'{args.split}: Micro={100*result["micro"]["f1"]:.6f}%, '
                  f'Macro13={100*result["macro_f1"]:.6f}%; artifacts: {folder}', flush=True)
        barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
