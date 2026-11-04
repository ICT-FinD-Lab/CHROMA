"""Write the 1,170-run main-table manifest without launching any experiments."""
import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
profiles=json.loads((ROOT/'configs/main_table.json').read_text())
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--datasets',nargs='+',choices=sorted(profiles),default=sorted(profiles))
a=p.parse_args()
if a.output.exists(): p.error(f'Output already exists: {a.output}')
rows=[]
for dataset in a.datasets:
    for backbone in ('gcn','gat'):
        for seed in range(5):
            settings=[('CLEAN',0)]+[(m,s) for s in (.5,1,1.5,2) for m in ('USP','GRASP','CHROMA')]
            for method,sigma in settings:
                rows.append(dict(dataset=dataset,backbone=backbone,method=method,sigma=sigma,seed=seed,
                                 epochs=profiles[dataset][backbone]['epochs'],
                                 command=f'python main.py --dataset {dataset} --backbone {backbone} --method {method} --sigma {sigma} --seed {seed} --device 0'))
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(f'Wrote {len(rows)} commands to {a.output}; no experiments launched.')
