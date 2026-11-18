"""Verify all reported seed-level metric summaries, using only Python's stdlib."""
import json, math, statistics
from pathlib import Path
from collections import defaultdict
root = Path(__file__).resolve().parent
runs = json.loads((root / 'runs.json').read_text())
summary = json.loads((root / 'summary.json').read_text())
groups = defaultdict(list)
seen = set()
for run in runs:
    key = (run['group'], run['seed'])
    if key in seen: raise ValueError('Duplicate group/seed: ' + str(key))
    seen.add(key)
    for mode, metrics in run['evaluations'].items():
        for metric, value in metrics.items():
            if not math.isfinite(value): raise ValueError('Nonfinite metric')
            groups[(run['group'], mode, metric)].append(value)
checked = 0
for group, entry in summary.items():
    for mode, metrics in entry['metrics'].items():
        for metric, expected in metrics.items():
            values = groups[(group, mode, metric)]
            if len(values) != 5: raise ValueError('Expected five training seeds')
            for field, actual in [('mean', statistics.fmean(values)), ('sd', statistics.stdev(values))]:
                if not math.isclose(actual, expected[field], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(str((group, mode, metric, field, actual, expected[field])))
            checked += 1
print(f'PASS: {len(runs)} final evaluations, {len(summary)} five-seed groups, {checked} metric summaries.')
