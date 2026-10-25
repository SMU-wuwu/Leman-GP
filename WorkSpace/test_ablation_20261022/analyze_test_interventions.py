"""Audit paired fixed-weight intervention predictions on the AgCNER test set."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def counts(gold, pred):
    return np.asarray([[len(g & p), len(p), len(g)] for g, p in zip(gold, pred)], dtype=np.int64)


def f1(c):
    return 200.0 * c[0] / max(1, c[1] + c[2])


def interval(texts, a, b, seed=20261022, n=4000):
    groups = {}
    for i, text in enumerate(texts):
        groups.setdefault(text, []).append(i)
    aa = np.asarray([a[ids].sum(0) for ids in groups.values()])
    bb = np.asarray([b[ids].sum(0) for ids in groups.values()])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(aa), size=(n, len(aa)))
    vals = np.asarray([f1(bb[ix].sum(0)) - f1(aa[ix].sum(0)) for ix in draws])
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975)), len(groups)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Frozen test intervention output")
    p.add_argument("--full", type=Path, required=True, help="Full-model test prediction JSONL")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    complete = json.loads((a.input / "COMPLETE.json").read_text())
    protocol = json.loads((a.input / "PROTOCOL.json").read_text())
    assert complete["split"] == protocol["split"] == "test"
    assert complete["weights_unchanged"] is True and complete["optimizer_steps"] == 0
    saved_rows = read(a.full)
    baseline_rows = read(a.input / "baseline" / "test_predictions.jsonl")
    assert len(saved_rows) == len(baseline_rows)
    assert [r["text"] for r in saved_rows] == [r["text"] for r in baseline_rows]
    assert [[(e["type"], e["start"], e["end"]) for e in r["gold"]] for r in saved_rows] == [
        [(e["type"], e["start"], e["end"]) for e in r["gold"]] for r in baseline_rows]
    saved_predictions = [set((e["type"], e["start"], e["end"]) for e in r["predicted"])
                         for r in saved_rows]
    baseline_predictions = [set((e["type"], e["start"], e["end"]) for e in r["predicted"])
                            for r in baseline_rows]
    baseline_differing_entities = sum(len(x ^ y) for x, y in zip(saved_predictions, baseline_predictions))
    assert baseline_differing_entities <= 20
    ref_rows = baseline_rows
    texts = [r["text"] for r in ref_rows]
    gold = [set((e["type"], e["start"], e["end"]) for e in r["gold"]) for r in ref_rows]
    reference = baseline_predictions
    ref_counts = counts(gold, reference)
    rows = []
    expected = ["baseline", "attention_identity", "attention_norm_only", "attention_half_context",
                "attention_uniform", "lexicon_norm_only", "lexicon_drop_top", "lexicon_drop_bottom"]
    for i, mode in enumerate(expected):
        pth = a.input / mode / "test_predictions.jsonl"
        rr = read(pth)
        assert len(rr) == len(ref_rows) and [r["text"] for r in rr] == texts
        assert [set((e["type"], e["start"], e["end"]) for e in r["gold"]) for r in rr] == gold
        pred = [set((e["type"], e["start"], e["end"]) for e in r["predicted"]) for r in rr]
        cc = counts(gold, pred)
        tp, predicted, support = map(int, cc.sum(axis=0))
        score = 200 * tp / (predicted + support)
        if mode == "baseline":
            assert pred == reference, "Frozen baseline must reproduce the intervention-run baseline predictions."
            delta = low = high = 0.0
            clusters = len(set(texts))
        else:
            delta = score - f1(ref_counts.sum(axis=0))
            low, high, clusters = interval(texts, ref_counts, cc, seed=20261022 + i)
        rows.append({"split": "test", "variant": mode, "reference": "full frozen model",
                     "precision": 100 * tp / predicted if predicted else 0,
                     "recall": 100 * tp / support if support else 0,
                     "micro_f1": score, "delta_micro_f1_pp": delta,
                     "ci_low_pp": low, "ci_high_pp": high,
                     "tp": tp, "fp": predicted - tp, "fn": support - tp,
                     "text_clusters": clusters})
    save(a.out / "frozen_interventions_test.csv", rows)
    audit = {"split": "AgCNER test", "sentences": len(texts), "weights_unchanged": True,
             "optimizer_steps": 0, "training": False,
             "baseline_matches_full_test_predictions": baseline_differing_entities == 0,
             "baseline_differing_entities_vs_saved_full": baseline_differing_entities,
             "paired_interventions_reference": "shared frozen-weight intervention baseline",
             "checkpoint_sha256": protocol["checkpoint_sha256"],
             "test_sha256": protocol.get("test_sha256", ""),
             "conditions": [r["variant"] for r in rows]}
    (a.out / "FROZEN_INTERVENTIONS_TEST_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
