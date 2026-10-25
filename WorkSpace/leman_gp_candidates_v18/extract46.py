import gc
import gzip
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'output/leman_fgm_seed46_dual4090/source_snapshot'))

import torch
import torch.distributed as dist
from config import Config
from data import load_split, prepare_split, sha256, write_json
from evaluate import release_loader
from features import collect_features
from model import LemanGP
from runtime import barrier, broadcast_object, environment, rank, setup


def main():
    device = setup()
    requested = Config.from_environment()
    root = Path(requested.root)
    run = root / "output/leman_fgm_seed46_dual4090"
    output = root / "output/leman_candidates_v18_seed46"
    output.mkdir(parents=True, exist_ok=True)
    try:
        path = run / "best.pt"
        digest = broadcast_object(sha256(path) if rank() == 0 else None)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert digest == "3d790b4796c31932d7ad51a01f57dcfb9ebb573ebae021ca757995961a4eb605"
        assert ckpt["config"]["seed"] == 46
        assert ckpt["best_epoch"] == 3
        cfg = Config(**ckpt["config"])
        cfg.root, cfg.data_dir, cfg.encoder_dir = requested.root, requested.data_dir, requested.encoder_dir
        lexicon = json.loads((run / "lexicon.json").read_text())
        cached = broadcast_object(str(prepare_split(cfg, "dev", lexicon, ckpt["types"])) if rank() == 0 else None)
        dataset = load_split(cached)
        model = LemanGP(
            cfg,
            len(ckpt["types"]),
            ckpt["vocab_size"],
            char_vocab_size=ckpt["char_vocab_size"],
            encoder_config=ckpt["encoder_config"],
            pretrained=False,
        )
        model.load_state_dict(ckpt["model"], strict=True)
        header = {
            "types": ckpt["types"],
            "checkpoint": {
                "checkpoint": str(path),
                "checkpoint_sha256": digest,
                "checkpoint_epoch": ckpt["best_epoch"],
                "weight_source": ckpt.get("weight_source", "ema"),
                "dev": ckpt["dev"],
                "source_hashes": ckpt["source_hashes"],
                "lexicon_sha256": ckpt["lexicon_sha256"],
            },
            "dev_sha256": ckpt["input_identity"]["dev2.txt"],
            "floor": -32.0,
            "environment": environment(device),
            "test_accessed": False,
        }
        del ckpt
        gc.collect()
        model.to(device)
        started = time.perf_counter()
        records = collect_features(model, dataset, cfg, device, output / "shards/fgm", floor=-32.0)
        if rank() == 0:
            target = output / "fgm_features.jsonl.gz"
            temporary = target.with_suffix(".gz.tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(header, ensure_ascii=False) + "\n")
                for sid in range(len(dataset.samples)):
                    stream.write(json.dumps({"sid": sid, "rows": sorted(records[sid])}) + "\n")
            temporary.replace(target)
            write_json(
                output / "extraction.json",
                {**header, "seconds": time.perf_counter() - started, "cache_sha256": sha256(target)},
            )
            print(f"FGM dev candidates saved: {target}", flush=True)
        release_loader(dataset)
        barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
