"""Report installed versions and CUDA availability without installing anything."""
import importlib
import importlib.metadata
import json
import sys

names={'torch':'torch','torch-geometric':'torch_geometric','torch-scatter':'torch_scatter',
       'torch-sparse':'torch_sparse','torchmetrics':'torchmetrics','networkx':'networkx',
       'numpy':'numpy','pandas':'pandas','scikit-learn':'sklearn','scipy':'scipy','ogb':'ogb',
       'gdown':'gdown','matplotlib':'matplotlib','seaborn':'seaborn','tqdm':'tqdm'}
result={'python':sys.version,'packages':{},'missing':[]}
for distribution,module in names.items():
    try:
        importlib.import_module(module)
        result['packages'][distribution]=importlib.metadata.version(distribution)
    except Exception as error:
        result['missing'].append({'package':distribution,'error':str(error)})
try:
    import torch
    result['cuda_available']=torch.cuda.is_available()
    result['cuda_device_count']=torch.cuda.device_count()
except ImportError:
    pass
print(json.dumps(result,indent=2))
raise SystemExit(bool(result['missing']))
