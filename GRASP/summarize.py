"""Aggregate per-seed measurements; compute AUC separability before averaging."""
import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
import statistics
from run_protocol import check_digest, protocol_digest


def input_records(path):
    if path.is_dir():
        for result in sorted(path.rglob('result.json')):
            yield json.loads(result.read_text(encoding='utf-8')), result.with_name('launch.json')
        return

    # The published CSVs predate protocol IDs. Recognize their contents, even
    # when copied or renamed; a matching filename alone is insufficient.
    manifest = Path(__file__).resolve().parent / 'assets/manifest.json'
    entries = json.loads(manifest.read_text(encoding='utf-8'))['files']
    reference_hashes = {entry['sha256'] for entry in entries if entry['path'].startswith('results/')}
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
        source.seek(0)
        with io.TextIOWrapper(source, encoding='utf-8', newline='') as text:
            for row in csv.DictReader(text):
                if digest in reference_hashes:
                    row['protocol_sha256'] = digest
                    row['smoke'] = False
                yield row, None


def run_settings(row, launch_path):
    digest = row.get('protocol_sha256')
    if launch_path is not None and launch_path.is_file():
        launch = json.loads(launch_path.read_text(encoding='utf-8'))
        computed = protocol_digest(launch)
        for key in ('dataset', 'backbone', 'method', 'sigma', 'seed', 'smoke',
                    'epochs', 'attack_backend', 'group_sha256'):
            if key in row and row[key] != launch.get(key):
                raise ValueError(f'{key} differs between result and {launch_path}')
        if any(stored != computed for stored in (digest, launch.get('protocol_sha256'))
               if stored is not None):
            raise ValueError(f'Protocol ID disagrees with {launch_path}')
        digest = computed
        for key in ('epochs', 'attack_backend', 'group_sha256'):
            if key in launch:
                row[key] = launch[key]
    if digest:
        check_digest(digest)
    # Also catch contradictions in flattened/exported records with a stale ID.
    fields = ('epochs', 'attack_backend', 'group_sha256', 'campaign', 'torch_version', 'numpy_version')
    return (digest or None, *(str(row.get(key, '')) for key in fields))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='Per-seed CSV or directory containing result.json files')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    groups=defaultdict(list); seen=set()
    protocols = {}
    for row, launch in input_records(a.input):
        if row.get('protocol_sha256') and 'smoke' not in row:
            p.error('Records exported with protocol_sha256 must retain the smoke flag.')
        smoke = str(row.get('smoke', False)).strip().lower()
        if smoke in ('true', '1'):
            continue
        if smoke not in ('false', '0'):
            p.error(f'Invalid smoke flag {smoke!r}. Expected true, false, 1, or 0.')
        key=(row['dataset'],row['backbone'],row['method'],float(row['sigma']))
        seed=int(row['seed'])
        if (*key,seed) in seen: p.error(f'Duplicate seed: {key}, seed {seed}')
        try:
            settings = run_settings(row, launch)
        except ValueError as error:
            p.error(f'{key}, seed {seed}: {error}')
        if key in protocols:
            if settings[0] is None or protocols[key][0] is None:
                p.error(f'Cannot combine seeds for {key} without a verified protocol. '
                        'Use the original run directories or export protocol_sha256 in the CSV.')
            if settings != protocols[key]:
                p.error(f'Mixed experiment settings for {key} at seed {seed}. Summarize each configuration separately.')
        protocols[key] = settings
        seen.add((*key,seed))
        auc=float(row['raw_auc']);accuracy=float(row['accuracy'])
        if not (0<=auc<=1 and 0<=accuracy<=1): p.error('Metrics must be fractions in [0,1]')
        groups[key].append((seed,max(auc,1-auc),accuracy))
    if not groups:
        p.error('No formal results found. Smoke runs are excluded.')
    out=[]
    for (dataset,backbone,method,sigma),values in sorted(groups.items()):
        out.append(dict(dataset=dataset,backbone=backbone,method=method,sigma=sigma,n=len(values),
                        seeds=','.join(str(v[0]) for v in sorted(values)),
                        complete_five_seed=set(v[0] for v in values)==set(range(5)),
                        auc_sep_mean_pct=100*statistics.mean(v[1] for v in values),
                        auc_sep_std_pct=100*statistics.pstdev(v[1] for v in values),
                        accuracy_mean_pct=100*statistics.mean(v[2] for v in values),
                        accuracy_std_pct=100*statistics.pstdev(v[2] for v in values)))
    if a.output.exists(): p.error(f'Output exists; refusing to overwrite: {a.output}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
    incomplete = sum(not row['complete_five_seed'] for row in out)
    print(f'Formal runs: {len(seen)}. Cells: {len(out)}. Incomplete cells: {incomplete}.')


if __name__=='__main__': main()
