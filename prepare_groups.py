"""Construct CHROMA groups from graph structure, using the frozen configuration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=sorted(json.loads((ROOT/'configs/main_table.json').read_text())))
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--output',type=Path,default=ROOT/'assets/groups')
    p.add_argument('--rebuild-coloring',action='store_true',help='Ignore the arxiv initial scaffold; expensive')
    a=p.parse_args()
    target=a.output/f'{a.dataset}.npy'
    if target.exists():
        p.error(f'Partition already exists: {target}. Choose another --output to rebuild.')
    import torch
    import numpy as np
    from dataset import load_dataset
    from coloring.initialization import independent_set_coloring
    from coloring.initialization import build_initial_partition
    from coloring.merging import generate_soft_path,select_structural_knee,structural_knee_metadata
    from coloring.metrics import assert_proper_coloring
    torch.set_num_threads(8)
    cfg=json.loads((ROOT/'configs/chroma.json').read_text())
    started=time.time()
    data=load_dataset(a.data_dir.resolve().as_posix()+'/',a.dataset)
    edge,n=data.graph['edge_index'],data.graph['num_nodes']
    cached=ROOT/'assets/groups/ogbn-arxiv_initial.npy'
    if a.dataset=='ogbn-arxiv' and cached.is_file() and not a.rebuild_coloring:
        initial=torch.from_numpy(np.load(cached,allow_pickle=False)).long()
        assert initial.numel()==n
        assert_proper_coloring(edge,n,initial)
        scaffold_source='packaged arxiv no-merge scaffold'
    else:
        # The optional fast strategy preserves the original NetworkX tie order.
        if a.dataset=='ogbn-arxiv':
            import networkx as nx
            from fast_independent import cached_independent_strategy,verify_equivalence
            verify_equivalence()
            original=nx.coloring.greedy_color
            def cached_greedy(G,strategy='largest_first',interchange=False):
                if strategy=='independent_set': strategy=cached_independent_strategy
                return original(G,strategy=strategy,interchange=interchange)
            nx.coloring.greedy_color=cached_greedy
            try: coloring=independent_set_coloring(edge,n)
            finally: nx.coloring.greedy_color=original
        else: coloring=independent_set_coloring(edge,n)
        initial=build_initial_partition(edge,n,coloring,hard_max_group_ratio=cfg['scaffold_cap']).group
        scaffold_source='canonical independent-set coloring and deterministic cap split'
    states=generate_soft_path(edge,n,initial,lambda_local=cfg['lambda_local'],
                              global_conflict_cap=cfg['tau_E'],local_conflict_p95_cap=cfg['tau_Q'],
                              max_group_ratio_cap=cfg['tau_M'],audit_top_b=cfg['audit_B'],minimum_colors=2)
    selected=select_structural_knee(states,risk_weight=cfg['knee_risk_weight'])
    a.output.mkdir(parents=True,exist_ok=True)
    np.save(target,selected.group.numpy(),allow_pickle=False)
    report=dict(dataset=a.dataset,config=cfg,scaffold_source=scaffold_source,seconds=time.time()-started,
                num_nodes=n,selected_metrics=selected.metrics,
                value_sha256=hashlib.sha256(selected.group.numpy().tobytes()).hexdigest(),
                knee=structural_knee_metadata(states,risk_weight=cfg['knee_risk_weight']),
                path=[dict(step=s.step,metrics=s.metrics,merged_colors=s.merged_colors,score=s.merge_score) for s in states])
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Saved {target}; selected step {selected.step}, {selected.metrics["num_colors"]} groups')


if __name__=='__main__': main()
