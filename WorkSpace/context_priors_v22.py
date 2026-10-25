"""Train-only empirical Bayes priors for entity boundary contexts."""
from collections import Counter

import numpy as np

NAMES = ("start_left_char", "start_left_bigram", "start_crossing_bigram",
         "end_right_char", "end_right_bigram", "end_crossing_bigram")
STRENGTH = 20.0
CAP = 8.0


def contexts(padded, position):
    return (padded[position - 1], padded[position - 2:position],
            padded[position - 1:position + 1], padded[position],
            padded[position:position + 2], padded[position - 1:position + 1])


def build_priors(train_samples, num_types):
    totals = [Counter() for _ in NAMES]
    positive = [[Counter() for _ in range(num_types)] for _ in NAMES]
    support = np.zeros(num_types, dtype=np.int64)
    total_boundary_positions = 0
    for text, spans in train_samples:
        padded = "\x02\x02" + text + "\x03\x03"
        total_boundary_positions += len(text) + 1
        for position in range(2, len(text) + 3):
            for i, key in enumerate(contexts(padded, position)):
                totals[i][key] += 1
        for typ, start, end in spans:
            support[typ] += 1
            for i, key in enumerate(contexts(padded, start + 2)[:3]):
                positive[i][typ][key] += 1
            for i, key in enumerate(contexts(padded, end + 2)[3:], 3):
                positive[i][typ][key] += 1
    assert all(sum(x.values()) == total_boundary_positions for x in totals)
    prior = support / total_boundary_positions
    tables = []
    for i, total in enumerate(totals):
        per_type = []
        for typ in range(num_types):
            assert sum(positive[i][typ].values()) == int(support[typ])
            p0 = prior[typ]
            table = {}
            for key, count in total.items():
                successes = positive[i][typ][key]
                probability = (successes + STRENGTH * p0) / (count + STRENGTH)
                value = np.log(probability / (1 - probability)) - np.log(p0 / (1 - p0))
                table[key] = float(np.clip(value, -CAP, CAP))
            per_type.append(table)
        tables.append(per_type)
    return tables, {"features": list(NAMES), "strength": STRENGTH, "log_odds_cap": CAP,
                    "train_boundary_positions": total_boundary_positions, "type_mentions": support.tolist(),
                    "type_boundary_prevalence": prior.tolist(),
                    "context_vocabularies": [len(x) for x in totals],
                    "unseen_context_value": 0.0,
                    "source": "train2 character occurrences and original gold span boundaries only"}


def candidate_features(keys, samples, tables):
    padded = ["\x02\x02" + text + "\x03\x03" for text, _ in samples]
    result = np.zeros((len(keys), len(NAMES)), dtype=np.float64)
    for row, (sid, typ, start, end) in enumerate(keys):
        context = contexts(padded[sid], start + 2)[:3] + contexts(padded[sid], end + 2)[3:]
        for i, key in enumerate(context):
            result[row, i] = tables[i][typ].get(key, 0.0)
    return result
