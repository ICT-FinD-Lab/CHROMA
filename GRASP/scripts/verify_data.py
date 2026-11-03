"""Check experiment-input hashes and per-seed reference measurements."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'assets/manifest.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        path = ROOT / record['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Checksum mismatch: {record["path"]}')

    counts = {}
    for name, expected in [('main_table_per_seed.csv', 1170), ('controls_per_seed.csv', 640)]:
        with (ROOT / 'results' / name).open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected:
            raise ValueError(f'Unexpected number of rows in {name}: {len(rows)}')
        cells = defaultdict(set)
        for row in rows:
            key = (row['dataset'], row['backbone'], row['method'], float(row['sigma']))
            seed = int(row['seed'])
            if seed in cells[key]:
                raise ValueError(f'Duplicate seed: {key}, {seed}')
            cells[key].add(seed)
            raw, sep, acc = (float(row[k]) for k in ('raw_auc', 'auc_sep', 'accuracy'))
            if not (0 <= raw <= 1 and 0 <= acc <= 1 and abs(sep - max(raw, 1 - raw)) < 1e-7):
                raise ValueError(f'Invalid metrics: {key}, {seed}')
        if any(seeds != set(range(5)) for seeds in cells.values()):
            raise ValueError(f'Incomplete reference cell in {name}')
        counts[name] = {'runs': len(rows), 'cells': len(cells)}
    print(json.dumps({'verified_files': len(manifest['files']), 'reference_results': counts}, indent=2))


if __name__ == '__main__':
    main()
