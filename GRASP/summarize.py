"""Aggregate per-seed measurements; compute AUC separability before averaging."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='Per-seed CSV or directory containing result.json files')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.input.is_dir():
        rows=[json.loads(f.read_text()) for f in sorted(a.input.rglob('result.json'))]
    else:
        with a.input.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    groups=defaultdict(list); seen=set()
    for row in rows:
        smoke = str(row.get('smoke', False)).strip().lower()
        if smoke in ('true', '1'):
            continue
        if smoke not in ('false', '0'):
            p.error(f'Invalid smoke flag {smoke!r}. Expected true, false, 1, or 0.')
        key=(row['dataset'],row['backbone'],row['method'],float(row['sigma']))
        seed=int(row['seed'])
        if (*key,seed) in seen: p.error(f'Duplicate seed: {key}, seed {seed}')
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
