"""Run training and record the selected-checkpoint metrics."""
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    from torch_geometric.utils import to_dense_adj, remove_self_loops
    import logger as logger_module
    import data_utils
    import dataset as dataset_module
    from GraphAttacker import GraphAttacker

    plan=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    torch.set_num_threads(8)
    if plan['device']!='cpu' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device is unavailable; use --device cpu explicitly')
    if plan['dataset']=='ogbn-arxiv':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    if plan['group_file']:
        contents = Path(plan['group_file']).read_bytes()
        if hashlib.sha256(contents).hexdigest() != plan['group_sha256']:
            raise ValueError('Group file changed after launch')
        values = np.load(io.BytesIO(contents), allow_pickle=False)
        if not isinstance(values, np.ndarray) or values.dtype.kind not in ('i', 'u'):
            raise ValueError('group IDs must be integers')
        group = torch.from_numpy(values.astype(values.dtype.newbyteorder('='), copy=False))
        folder=Path('groups')/plan['dataset']; folder.mkdir(parents=True)
        torch.save(group,folder/f'{plan["dataset"]}_tgrasp.pt')
    epochs=[]; attack_values=[]; started=time.time()
    original_add=logger_module.Logger.add_result
    def log_epoch(self,run,result):
        original_add(self,run,result)
        values=[float(v) for v in result]
        record=dict(epoch=len(epochs),train_acc=values[0],valid_acc=values[1],test_acc=values[2],valid_loss=values[3])
        epochs.append(record)
        with open('epochs.jsonl','a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
    logger_module.Logger.add_result=log_epoch
    original_checkpoint=logger_module.load_model
    def keep_checkpoint(args,model,optimizer,run,id=None,delete=False):
        if delete: return None,None
        return original_checkpoint(args,model,optimizer,run,id=id,delete=False)
    logger_module.load_model=keep_checkpoint

    def save_split(split):
        values={k:v.cpu().numpy() for k,v in split.items()}
        joined=np.concatenate(list(values.values()))
        assert len(np.unique(joined))==len(joined), 'Split overlap detected'
        np.savez('actual_split.npz',**values)
    original_random=dataset_module.NCDataset.get_idx_split
    def tracked_random(self,*args,**kwargs):
        result=original_random(self,*args,**kwargs);save_split(result);return result
    dataset_module.NCDataset.get_idx_split=tracked_random
    original_fixed=data_utils.load_fixed_splits
    def tracked_fixed(*args,**kwargs):
        result=original_fixed(*args,**kwargs);save_split(result[0]);return result
    data_utils.load_fixed_splits=tracked_fixed

    original_init=GraphAttacker.__init__
    def init_attacker(self,*args,**kwargs):
        assert not args and kwargs['args'].attack_pair_samples==0
        print(f'Evaluation backend: {plan["attack_backend"]}',flush=True)
        if plan['attack_backend']=='dense_exact':
            graph=kwargs['graph']
            edge,_=remove_self_loops(graph['edge_index'])
            kwargs['A_vec']=to_dense_adj(edge,max_num_nodes=graph['num_nodes']).reshape(-1).to(kwargs['device_auc'])
        else: kwargs['A_vec']=torch.empty(0)
        original_init(self,**kwargs)
    GraphAttacker.__init__=init_attacker
    dense_embedding_auc = GraphAttacker.embedding_auc
    def tracked_embedding_auc(self):
        if plan['attack_backend']=='blockwise_exact':
            from exact_auc import exact_auc
            result=exact_auc(self.H_A,self.A,block=256,
                             backend='lut' if self.H_A.is_cuda else 'search',output='evaluation')
            value = result['raw_auc']
        else:
            value = dense_embedding_auc(self)
        attack_values.append(float(value))
        return value
    GraphAttacker.embedding_auc = tracked_embedding_auc
    sys.argv=[str(ROOT/'train.py')]+plan['training_args']
    runpy.run_path(sys.argv[0],run_name='__main__')
    assert len(attack_values)==1 and len(epochs)==plan['epochs']
    selected=max(epochs,key=lambda row:row['valid_acc'])
    raw=attack_values[0]
    result={key:plan[key] for key in ('dataset','backbone','method','sigma','seed','smoke','epochs','attack_backend',
                                      'protocol_sha256')}
    if plan['group_file']:
        result['group_sha256'] = plan['group_sha256']
    result.update(raw_auc=raw,auc_sep=max(raw,1-raw),accuracy=selected['test_acc'],
                  validation_accuracy=selected['valid_acc'],selected_epoch=selected['epoch'],
                  elapsed_seconds=time.time()-started,
                  split_sha256=hashlib.sha256(Path('actual_split.npz').read_bytes()).hexdigest(),
                  torch_version=torch.__version__,numpy_version=np.__version__,
                  device=str(torch.cuda.get_device_name(0)) if plan['device']!='cpu' else 'cpu')
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()
