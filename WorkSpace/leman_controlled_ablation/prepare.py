"""Build only train-derived resources and train/dev caches; no GPU or optimizer."""
import argparse
from pathlib import Path
from collections import Counter
from config import Config
from data import entity_types, load_split, sha256, write_json
from runtime import input_identity, source_hashes
from train import prepare_resources


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/root/data/WorkSpace')
    args = p.parse_args()
    result = {}
    for variant in ('F', 'max_words3', 'max_words5'):
        cfg = Config.make(args.root, variant, 42, 'A')
        types = entity_types(Path(cfg.data_dir)/'label.txt')
        identity = input_identity(cfg)
        paths = prepare_resources(cfg, identity, types)
        datasets = list(map(load_split, paths))
        result[variant] = dict(cache_paths=paths, stats={s: d.stats for s, d in zip(('train', 'dev'), datasets)})
        for split, dataset in zip(('train', 'dev'), datasets):
            lengths = Counter(b-a for _, spans in dataset.samples for _, a, b in spans)
            result[variant]['stats'][split]['gold_longer_than_max_span'] = sum(n for length, n in lengths.items() if length > cfg.max_span)
            if dataset.stats['gold_not_fully_contained_in_any_window'] or result[variant]['stats'][split]['gold_longer_than_max_span']:
                raise ValueError(f'{split} has gold outside the declared window/span coverage')
    report = {'input_identity': identity, 'source_hashes': source_hashes(Path(__file__).parent),
              'lexicon_sha256': sha256(Path(cfg.resource_dir)/'lexicon.json'), 'variants': result,
              'shared_resources': cfg.resource_dir, 'test_read': False, 'training_started': False}
    write_json(Path(cfg.resource_dir)/'PREPARED.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
