"""Summarize held-out-test mechanism probes at sentence/entity level."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csvout(path, data):
    if not data:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def interval(per_sentence, texts_by_sid, n=4000, seed=20261022):
    clusters = defaultdict(list)
    for sid, value in per_sentence.items():
        clusters[texts_by_sid[sid]].append(float(value))
    vals = np.asarray([np.mean(v) for v in clusters.values()], dtype=float)
    rng = np.random.default_rng(seed)
    draws = vals[rng.integers(0, len(vals), size=(n, len(vals)))].mean(axis=1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((a.panel / "PROTOCOL.json").read_text())
    panel_meta = json.loads((a.panel / "mechanisms/test_fixed_panel/PANEL.json").read_text())
    assert protocol["split"] == "test" and panel_meta["scope"].lower().find("test") >= 0
    assert protocol["panel_sentences"] == panel_meta["sentence_count"] == 128
    assert protocol["training"] is False and protocol["optimizer_steps"] == 0

    test_rows = [json.loads(line) for line in a.predictions.read_text().splitlines()]
    samples = {r["sid"]: r for r in test_rows}
    selected = set(panel_meta["sentence_ids"])
    assert selected <= samples.keys()
    texts = {sid: samples[sid]["text"] for sid in selected}
    gold = {sid: [(e["type"], int(e["start"]), int(e["end"])) for e in samples[sid]["gold"]]
            for sid in selected}

    # Collapse duplicated lexical candidates caused by overlapping model windows.
    by_pos = defaultdict(lambda: defaultdict(list))
    for r in rows(a.panel / "mechanisms/test_fixed_panel/lexical_slots.csv"):
        sid, pos = int(r["sid"]), int(r["position"])
        if sid not in selected:
            continue
        key = (r["word"], int(r["bmes"]))
        by_pos[(sid, pos)][key].append(float(r["weight"]))
    per_sentence_mass = defaultdict(list)
    ambiguous_gold_positions = 0
    for (sid, pos), candidates in by_pos.items():
        if len(candidates) < 2:
            continue
        correct = False
        for t, start, end in gold[sid]:
            word = texts[sid][start:end]
            role = 3 if end - start == 1 else 0 if pos == start else 2 if pos == end - 1 else 1
            if start <= pos < end and any(w == word and bmes == role for w, bmes in candidates):
                correct = True
                break
        if not correct:
            continue
        ambiguous_gold_positions += 1
        mass = sum(float(np.mean(ws)) for (word, bmes), ws in candidates.items()
                   if any(start <= pos < end and texts[sid][start:end] == word and
                          bmes == (3 if end-start == 1 else 0 if pos == start else 2 if pos == end-1 else 1)
                          for _, start, end in gold[sid]))
        per_sentence_mass[sid].append(mass)
    per_sentence_mass = {sid: float(np.mean(v)) for sid, v in per_sentence_mass.items() if v}
    lo, hi = interval(per_sentence_mass, texts)
    lexical_summary = [{"split": "test", "panel_sentences": 128,
                        "sentences_with_ambiguous_gold_positions": len(per_sentence_mass),
                        "ambiguous_gold_positions": ambiguous_gold_positions,
                        "mean_gold_candidate_weight": float(np.mean(list(per_sentence_mass.values()))),
                        "ci_low": lo, "ci_high": hi,
                        "summary_unit": "text-cluster mean; 95% paired text-cluster bootstrap interval"}]
    csvout(a.out / "lexical_mechanism_test.csv", lexical_summary)

    margin_rows = rows(a.panel / "mechanisms/test_fixed_panel/gold_span_margins.csv")
    entity_margin = defaultdict(list)
    for r in margin_rows:
        if r["boundary_margin"] == "" or r["boundary_margin"] == "None":
            continue
        sid = int(r["sid"])
        if sid not in selected:
            continue
        key = (sid, r["type"], int(r["start"]), int(r["end"]))
        entity_margin[key].append(float(r["boundary_margin"]))
    grouped_sentences = defaultdict(lambda: defaultdict(list))
    for (sid, typ, start, end), values in entity_margin.items():
        length = end - start
        group = "length_1-2" if length <= 2 else "length_3-5" if length <= 5 else "length_6-10" if length <= 10 else "length_11+"
        grouped_sentences[group][sid].append(float(np.mean(values)))
    margin_summary = []
    for group, by_sid in sorted(grouped_sentences.items()):
        ps = {sid: float(np.mean(v)) for sid, v in by_sid.items()}
        low, high = interval(ps, texts, seed=20261023)
        margin_summary.append({"split": "test", "group": group, "sentences": len(ps),
                               "entities": sum(len(v) for v in by_sid.values()),
                               "mean_gold_boundary_margin": float(np.mean(list(ps.values()))),
                               "ci_low": low, "ci_high": high,
                               "summary_unit": "text-cluster mean; 95% paired text-cluster bootstrap interval",
                               "positive_margin_definition": "gold-span score minus strongest same-type nearby incorrect-boundary score"})
    csvout(a.out / "span_margin_test.csv", margin_summary)

    attention_rows = rows(a.panel / "mechanisms/test_fixed_panel/attention_heads.csv")
    per_head = defaultdict(lambda: defaultdict(list))
    for r in attention_rows:
        sid, head = int(r["sid"]), int(r["head"])
        if sid not in selected or not r["within_entity_mass"] or not r["uniform_within_entity_mass"]:
            continue
        denom = float(r["uniform_within_entity_mass"])
        if denom > 0:
            per_head[head][sid].append(float(r["within_entity_mass"]) / denom)
    attention_summary = []
    for head, by_sid in sorted(per_head.items()):
        ps = {sid: float(np.mean(v)) for sid, v in by_sid.items()}
        low, high = interval(ps, texts, seed=20261024 + head)
        attention_summary.append({"split": "test", "head": head + 1, "sentences": len(ps),
                                  "mean_within_entity_attention_enrichment": float(np.mean(list(ps.values()))),
                                  "ci_low": low, "ci_high": high,
                                  "summary_unit": "text-cluster mean; 95% paired text-cluster bootstrap interval"})
    csvout(a.out / "attention_enrichment_test.csv", attention_summary)

    boundary = rows(a.panel / "mechanisms/test_fixed_panel/boundary_metrics.csv")
    for r in boundary:
        r["split"] = "test"
        if "type" in r:
            r["entity_type"] = r.pop("type")
    csvout(a.out / "boundary_head_metrics_test.csv", boundary)
    summary = {"split": "AgCNER test", "panel_sentences": 128,
               "sentence_selection": "fixed text-hash selection, independent of predictions",
               "checkpoint_sha256": protocol["checkpoint_sha256"],
               "test_annotation_sha256": protocol["test_sha256"],
               "optimizer_steps": 0, "training": False,
               "files": ["lexical_mechanism_test.csv", "span_margin_test.csv",
                         "attention_enrichment_test.csv", "boundary_head_metrics_test.csv"]}
    (a.out / "MECHANISM_TEST_AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
