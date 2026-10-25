"""Copied unchanged from v34 cpu_interval.py."""
import numpy as np

def interval_marginals(rows, length):
    starts, ends = [[] for _ in range(length+1)], [[] for _ in range(length+1)]
    for i, (_, start, end, logit) in enumerate(rows):
        starts[start].append(i)
        ends[end].append(i)
    left = np.zeros(length+1, dtype=np.float64)
    right = np.zeros(length+1, dtype=np.float64)
    for pos in range(1, length+1):
        value = left[pos-1]
        for i in ends[pos]:
            t, a, b, z = rows[i]
            value = np.logaddexp(value, left[a]+z)
        left[pos] = value
    for pos in range(length-1, -1, -1):
        value = right[pos+1]
        for i in starts[pos]:
            t, a, b, z = rows[i]
            value = np.logaddexp(value, right[b]+z)
        right[pos] = value
    assert abs(left[length]-right[0]) < 1e-9
    marginals = np.asarray([np.exp(min(0.0, left[a]+z+right[b]-left[length]))
                            for t,a,b,z in rows], dtype=np.float64)
    return marginals, float(left[length])
