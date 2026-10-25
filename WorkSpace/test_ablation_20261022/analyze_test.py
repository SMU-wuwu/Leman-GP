"""Recount frozen checkpoint predictions on AgCNER test and export figure data."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ORDER = [
    "F", "no_lexicon", "no_bmes", "no_recurrence", "no_gru_linear",
    "no_rope", "no_attention", "max_words3", "max_words5", "gp_reset",
    "crf", "uniform_lexicon", "fixed_gate", "no_char", "no_boundary",
    "fgm_control", "no_fgm", "clean_two_pass", "weighted_control",
    "unit_loss", "encoder_gp",
]
REFERENCES = {v: "F" for v in ORDER}
REFERENCES.update({"F": None, "fgm_control": None, "weighted_control": None,
                   "crf": "gp_reset", "no_fgm": "fgm_control",
                   "clean_two_pass": "fgm_control", "unit_loss": "weighted_control"})


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def counts(gold, pred):
    return np.asarray([[len(g & p), len(p), len(g)] for g, p in zip(gold, pred)], dtype=np.int64)


def f1_from_counts(c):
    return 200.0 * c[0] / max(1, c[1] + c[2])


def bootstrap_difference(texts, ref_counts, variant_counts, n=4000, seed=20261022):
    clusters = {}
    for i, text in enumerate(texts):
        clusters.setdefault(text, []).append(i)
    ref = np.asarray([ref_counts[ids].sum(axis=0) for ids in clusters.values()], dtype=np.int64)
    var = np.asarray([variant_counts[ids].sum(axis=0) for ids in clusters.values()], dtype=np.int64)
    rng = np.random.default_rng(seed)
    values = np.empty(n, dtype=np.float64)
    for i in range(n):
        draw = rng.integers(0, len(ref), size=len(ref))
        values[i] = f1_from_counts(var[draw].sum(axis=0)) - f1_from_counts(ref[draw].sum(axis=0))
    return float(np.quantile(values, .025)), float(np.quantile(values, .975)), len(ref)


def metric_row(v, c):
    tp, pred, gold = map(int, c.sum(axis=0))
    return {"variant": v, "precision": 100 * tp / pred if pred else 0,
            "recall": 100 * tp / gold if gold else 0,
            "micro_f1": 200 * tp / (pred + gold) if pred + gold else 0,
            "tp": tp, "fp": pred - tp, "fn": gold - tp,
            "predicted": pred, "gold": gold}


def boundary_errors(gold, pred):
    err = Counter()
    for gset, pset in zip(gold, pred):
        for typ, start, end in gset - pset:
            label = ("type_confusion" if any(a == start and b == end for _, a, b in pset)
                     else "boundary_shift" if any(t == typ and max(start, a) < min(end, b)
                                                  for t, a, b in pset)
                     else "missed")
            err[f"FN_{label}"] += 1
        for typ, start, end in pset - gset:
            label = ("type_confusion" if any(a == start and b == end for _, a, b in gset)
                     else "boundary_shift" if any(t == typ and max(start, a) < min(end, b)
                                                  for t, a, b in gset)
                     else "spurious")
            err[f"FP_{label}"] += 1
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Downloaded CPU inference output root")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=4000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    texts = None
    gold = None
    predictions = {}
    input_hashes = {}
    for v in ORDER:
        d = a.input / v
        complete = json.loads((d / "COMPLETE.json").read_text())
        protocol = json.loads((d / "PROTOCOL.json").read_text())
        assert complete["split"] == protocol["split"] == "test"
        assert complete["all_windows_evaluated"] is True
        assert complete["weights_unchanged"] is True and complete["optimizer_steps"] == 0
        rows = [json.loads(line) for line in (d / "test_predictions.jsonl").read_text().splitlines()]
        assert len(rows) == 5974 and [r["sid"] for r in rows] == list(range(len(rows)))
        cur_texts = [r["text"] for r in rows]
        cur_gold = [set((e["type"], int(e["start"]), int(e["end"])) for e in r["gold"]) for r in rows]
        cur_pred = [set((e["type"], int(e["start"]), int(e["end"])) for e in r["predicted"]) for r in rows]
        for row in rows:
            assert all(e["text"] == row["text"][e["start"]:e["end"]]
                       for k in ("gold", "predicted") for e in row[k])
        if texts is None:
            texts, gold = cur_texts, cur_gold
        else:
            assert texts == cur_texts and gold == cur_gold, f"test annotations differ for {v}"
        assert sum(map(len, cur_gold)) == 20524
        predictions[v] = cur_pred
        input_hashes[v] = {"predictions": hashlib.sha256((d / "test_predictions.jsonl").read_bytes()).hexdigest(),
                           "checkpoint": protocol["checkpoint_sha256"],
                           "test": protocol["test_sha256"]}

    count_arrays = {v: counts(gold, predictions[v]) for v in ORDER}
    metric_rows = [metric_row(v, count_arrays[v]) for v in ORDER]
    metrics_by_variant = {r["variant"]: r for r in metric_rows}
    for row in metric_rows:
        v = row["variant"]
        ref = REFERENCES[v]
        row["reference"] = ref or ""
        if ref:
            lo, hi, n_clusters = bootstrap_difference(texts, count_arrays[ref], count_arrays[v], a.bootstrap)
            row.update(delta_micro_f1_pp=row["micro_f1"] - metrics_by_variant[ref]["micro_f1"],
                       ci_low_pp=lo, ci_high_pp=hi, bootstrap_replicates=a.bootstrap,
                       text_clusters=n_clusters)
        else:
            row.update(delta_micro_f1_pp="", ci_low_pp="", ci_high_pp="",
                       bootstrap_replicates="", text_clusters="")
    write_csv(a.out / "test_ablation_metrics.csv", metric_rows)

    errors = []
    for v in ("F", "no_lexicon", "no_bmes", "no_rope"):
        for category, n in sorted(boundary_errors(gold, predictions[v]).items()):
            errors.append({"split": "test", "variant": v, "category": category, "count": n})
    write_csv(a.out / "error_categories_test.csv", errors)

    subgroups = []
    for v in ("F", "gp_reset", "crf"):
        grouped = defaultdict(lambda: [0, 0])
        for gset, pset in zip(gold, predictions[v]):
            for entity in gset:
                length = entity[2] - entity[1]
                group = "length_1-2" if length <= 2 else "length_3-5" if length <= 5 else "length_6-10" if length <= 10 else "length_11+"
                grouped[group][0] += entity in pset
                grouped[group][1] += 1
        for group, (tp, support) in grouped.items():
            subgroups.append({"split": "test", "variant": v, "group": group, "tp": tp,
                              "support": support, "recall": tp / support})
    write_csv(a.out / "subgroup_recall_test.csv", subgroups)

    per_type_rows = []
    types = sorted({t for g in gold for t, _, _ in g})
    for v in ORDER:
        for typ in types:
            tp = sum(len({e for e in g if e[0] == typ} & {e for e in p if e[0] == typ})
                     for g, p in zip(gold, predictions[v]))
            pred_n = sum(sum(e[0] == typ for e in p) for p in predictions[v])
            gold_n = sum(sum(e[0] == typ for e in g) for g in gold)
            per_type_rows.append({"variant": v, "type": typ, "tp": tp, "predicted": pred_n,
                                  "gold": gold_n, "precision": tp / pred_n if pred_n else 0,
                                  "recall": tp / gold_n if gold_n else 0,
                                  "f1": 2 * tp / (pred_n + gold_n) if pred_n + gold_n else 0})
    write_csv(a.out / "per_type_test.csv", per_type_rows)

    report = {"split": "AgCNER test", "sentences": len(texts), "text_clusters": len(set(texts)),
              "gold_entities": sum(map(len, gold)), "variants": len(ORDER), "bootstrap_replicates": a.bootstrap,
              "bootstrap_unit": "sentence text cluster; paired by cluster", "optimizer_steps": 0,
              "weights_changed": False, "results": metrics_by_variant,
              "inputs": input_hashes}
    (a.out / "TEST_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"variants": len(ORDER), "sentences": len(texts),
                      "F": metrics_by_variant["F"], "test": metrics_by_variant["F"].get("micro_f1")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
