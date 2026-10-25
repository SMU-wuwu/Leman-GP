"""Small ridge logistic span calibrator; no model weights or architecture change."""
import numpy as np


def feature_rows(parent, upgraded, family, scale=8.0):
    def best(rows):
        result = {}
        for t, a, b, score, penalty in rows:
            key = (t, a, b)
            if key not in result or (score, penalty) > result[key]:
                result[key] = (score, penalty)
        return result

    keys, values = [], []
    for sid in sorted(upgraded):
        u = best(upgraded[sid])
        p = {} if family == 'single_v4' else best(parent[sid])
        spans = sorted(u if family == 'single_v4' else u.keys() | p.keys())
        for key in spans:
            row = []
            for model in ([u] if family == 'single_v4' else [p, u]):
                score, penalty = model.get(key, (-4.0, 0.0))
                row.extend([score / scale, penalty, np.exp(penalty)])
            row.append(np.log1p(key[2] - key[1]))
            if family != 'single_v4':
                row.extend([float(key in p), float(key in u)])
            keys.append((sid, *key))
            values.append(row)
    return keys, np.asarray(values, dtype=np.float64)


def fit_logistic(keys, values, samples, types, ids, spec):
    ids = set(ids)
    train = np.array([k[0] in ids for k in keys])
    support = np.zeros(len(types), dtype=int)
    for sid in ids:
        for t, _, _ in samples[sid][1]:
            support[t] += 1
    eligible = np.flatnonzero(support >= 200).tolist()
    mean, std = values[train].mean(0), values[train].std(0)
    std = np.where(std < 1e-8, 1.0, std)
    typ = np.array([key[1] for key in keys])
    design = np.column_stack([np.ones(len(keys)), (values - mean) / std,
                              *[(typ == t).astype(float) for t in eligible]])
    gold = {sid: set(map(tuple, samples[sid][1])) for sid in ids}
    labels = np.array([float(tuple(key[1:]) in gold[key[0]]) for key in np.array(keys)[train]])
    x = design[train]
    coefficients = np.zeros(x.shape[1])
    coefficients[0] = np.log((labels.sum() + 0.5) / (len(labels) - labels.sum() + 0.5))
    regularizer = np.full(x.shape[1], spec['ridge'])
    regularizer[0] = 0.0

    def objective(beta):
        scores = np.einsum('ij,j->i', x, beta, optimize=False)
        return float((np.logaddexp(0, scores) - labels * scores).sum() +
                     0.5 * (regularizer * beta**2).sum())

    for iteration in range(100):
        logits = np.einsum('ij,j->i', x, coefficients, optimize=False)
        probability = np.exp(-np.logaddexp(0, -logits))
        gradient = np.einsum('ij,i->j', x, probability - labels, optimize=False) + regularizer * coefficients
        if np.max(np.abs(gradient)) < 1e-7:
            break
        curvature = probability * (1 - probability)
        hessian = np.einsum('ni,n,nj->ij', x, curvature, x, optimize=False) + np.diag(regularizer)
        step = np.linalg.solve(hessian, gradient)
        rate, current = 1.0, objective(coefficients)
        while objective(coefficients - rate * step) > current - 1e-4 * rate * np.dot(gradient, step):
            rate *= 0.5
            if rate < 1e-12:
                break
        coefficients -= rate * step
        if np.max(np.abs(rate * step)) < 1e-10:
            break
    return {'mean': mean.tolist(), 'std': std.tolist(), 'coefficients': coefficients.tolist(),
            'eligible_types': eligible, 'support': support.tolist(), 'iterations': iteration + 1,
            'fit_candidate_rows': int(train.sum()), 'fit_positive_rows': int(labels.sum()),
            'gradient_max': float(np.max(np.abs(gradient))),
            'objective': objective(coefficients)}, np.einsum('ij,j->i', design, coefficients, optimize=False)


def candidates_from_scores(keys, scores, num_samples):
    candidates = {sid: [] for sid in range(num_samples)}
    for (sid, t, a, b), value in zip(keys, scores):
        if value > -4.0:
            candidates[sid].append((t, a, b, float(value)))
    return candidates


def apply_logistic(keys, values, fitted):
    mean = np.asarray(fitted['mean'], dtype=np.float64)
    std = np.asarray(fitted['std'], dtype=np.float64)
    typ = np.array([key[1] for key in keys])
    design = np.column_stack([np.ones(len(keys)), (values - mean) / std,
                              *[(typ == t).astype(float) for t in fitted['eligible_types']]])
    return np.einsum('ij,j->i', design, np.asarray(fitted['coefficients']), optimize=False)
