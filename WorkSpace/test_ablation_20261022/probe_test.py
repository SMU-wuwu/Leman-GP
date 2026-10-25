"""Deterministic, read-only mechanism panel from the held-out test split."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/root/data/WorkSpace")
sys.path.insert(0, str(ROOT / "leman_short_mechanism_v1"))

from config import Config
from data import entity_types, load_split, prepare_split, sha256, write_json
from model import LemanGP
from probes import probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='F')
    args = parser.parse_args()
    out = ROOT / "output/test_ablation_20261022"
    panel_out = out / "test_mechanism_panel" if args.variant == 'F' else out / 'test_mechanism_panels' / args.variant
    c = Config.make(ROOT, args.variant, 42)
    c.workers = 0
    c.eval_batch_size = 8
    c.probe_sentences = 128
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    run = Path(c.output_dir)
    checkpoint = run / "final_ema.pt"
    checkpoint_meta = json.loads((run / "PRIMARY_ENDPOINT.json").read_text())
    assert sha256(checkpoint) == checkpoint_meta["sha256"]
    lex = json.loads((Path(c.resource_dir) / "lexicon.json").read_text())
    types = entity_types(Path(c.data_dir) / "label.txt")
    ck = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    for key, value in ck["config"].items():
        if hasattr(c, key):
            setattr(c, key, value)
    c.workers = 0
    c.eval_batch_size = 8
    c.probe_sentences = 128
    model = LemanGP(c, len(types), len(lex["words"]),
                    char_vocab_size=len(lex["characters"]),
                    encoder_config=ck["encoder_config"], pretrained=False)
    model.load_state_dict(ck["model"], strict=True, assign=True)
    del ck
    model.eval().requires_grad_(False)
    assert not torch.cuda.is_initialized()

    cache_cfg = Config.make(ROOT, "F", 42)
    cache_cfg.output_dir = str(out)
    test = load_split(prepare_split(cache_cfg, "test", lex, types))
    assert len(test.samples) == 5974 and sum(len(g) for _, g in test.samples) == 20524
    write_json(panel_out / "PROTOCOL.json", {
        "split": "test", "sentences_in_source_split": len(test.samples),
        "panel_sentences": c.probe_sentences,
        "panel_selection": "lowest SHA-256 ranks from a fixed text-hash rule; independent of model outcomes",
        "checkpoint_sha256": sha256(checkpoint), "test_sha256": sha256(Path(c.data_dir) / "test2.txt"),
        "training": False, "optimizer_steps": 0, "device": "cpu", "precision": "fp32", "variant": args.variant,
        "purpose": "descriptive internal measurements; not used for fitting, selection, or thresholding",
    })
    probe(model, test, c, torch.device("cpu"), panel_out, "test_fixed_panel")
    panel_path = panel_out / "mechanisms" / "test_fixed_panel" / "PANEL.json"
    panel_meta = json.loads(panel_path.read_text())
    panel_meta["scope"] = "fixed test panel; descriptive; overlapping windows may repeat positions; not corpus counts"
    panel_path.write_text(json.dumps(panel_meta, ensure_ascii=False, indent=2) + "\n")
    print("TEST_PANEL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
