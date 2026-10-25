"""Batch exact weighted interval DP for a grid of 13 type thresholds."""
from bisect import bisect_right
import numpy as np


class GridIntervals:
    def __init__(self, keys, utilities, samples, minimum=0.01):
        grouped = [[] for _ in samples]
        for key, value in zip(keys, utilities):
            sid, typ, start, end = key
            if value >= minimum:
                grouped[sid].append((typ, start, end, float(value)))
        self.minimum = minimum
        typ, utility, compatible, hit, stages, last = [], [], [], [], [], []
        for sid, rows in enumerate(grouped):
            rows.sort(key=lambda r: (r[2], r[1], r[0]))
            offset = len(typ)
            typ.append(0); utility.append(0.0); compatible.append(offset); hit.append(0)
            ends = [r[2] for r in rows]
            gold = set(samples[sid][1])
            for j, (t, a, b, value) in enumerate(rows):
                if j == len(stages):
                    stages.append([])
                stages[j].append(len(typ))
                typ.append(t); utility.append(value)
                compatible.append(offset + bisect_right(ends, a, hi=j))
                hit.append(int((t, a, b) in gold))
            last.append(len(typ) - 1)
        self.typ = np.asarray(typ, dtype=np.int16)
        self.utility = np.asarray(utility)
        self.compatible = np.asarray(compatible, dtype=np.int32)
        self.hit = np.asarray(hit, dtype=np.int16)
        self.stages = [np.asarray(x, dtype=np.int32) for x in stages]
        self.last = np.asarray(last, dtype=np.int32)
        self.gold = sum(len(set(gold)) for _, gold in samples)

    def counts(self, thresholds):
        thresholds = np.atleast_2d(thresholds)
        if thresholds.min() < self.minimum:
            raise ValueError("Threshold falls below the retained utility floor")
        shape = (len(self.typ), len(thresholds))
        optimum = np.zeros(shape)
        true_positive = np.zeros(shape, dtype=np.int16)
        predicted = np.zeros(shape, dtype=np.int16)
        for rows in self.stages:
            previous = rows - 1
            compatible = self.compatible[rows]
            # Match the scalar decoder's floating-point order: compute each
            # candidate's utility-minus-threshold before adding its prefix DP.
            take_value = (optimum[compatible]
                          + (self.utility[rows, None]
                             - thresholds[:, self.typ[rows]].T))
            choose = take_value > optimum[previous]
            optimum[rows] = np.where(choose, take_value, optimum[previous])
            true_positive[rows] = np.where(
                choose, true_positive[compatible] + self.hit[rows, None], true_positive[previous])
            predicted[rows] = np.where(choose, predicted[compatible] + 1, predicted[previous])
        tp = true_positive[self.last].sum(0, dtype=np.int64)
        pred = predicted[self.last].sum(0, dtype=np.int64)
        return tp, pred

    def scores(self, thresholds):
        tp, pred = self.counts(thresholds)
        return 2 * tp / (pred + self.gold)


def optimize(prepared, initial=None, refine=False):
    """Maximize dev Micro using all 13 type cutoffs; no gold-specific rules."""
    history = []
    if initial is None:
        global_grid = np.arange(0.1, 6.50001, 0.1)
        trials = np.repeat(global_grid[:, None], 13, axis=1)
        values = prepared.scores(trials)
        k = int(np.argmax(values))
        current, value = trials[k].copy(), float(values[k])
    else:
        current = np.array(initial, dtype=float)
        value = float(prepared.scores(current)[0])
    initial_score = value
    schedules = [(0.1, 0.8, 2)]
    if refine:
        schedules += [(0.025, 0.3, 3), (0.005, 0.08, 2)]
    for step, radius, passes in schedules:
        for pass_id in range(passes):
            previous = value
            for typ in range(13):
                offsets = np.arange(-round(radius / step), round(radius / step) + 1) * step
                grid = np.unique(np.r_[current[typ], np.clip(current[typ] + offsets, 0.01, 7.99)])
                # The incumbent is first so exact ties retain it.
                grid = np.r_[current[typ], grid[grid != current[typ]]]
                trials = np.repeat(current[None, :], len(grid), axis=0)
                trials[:, typ] = grid
                values = prepared.scores(trials)
                k = int(np.argmax(values))
                if values[k] > value + 1e-14:
                    current, value = trials[k].copy(), float(values[k])
            history.append({"step": step, "pass": pass_id, "micro_f1": value,
                            "thresholds": current.tolist()})
            if value <= previous + 1e-14:
                break
    return {"thresholds_array": current.tolist(), "micro_f1": value,
            "initial_micro_f1": initial_score, "history": history}
