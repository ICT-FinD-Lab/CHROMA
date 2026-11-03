"""Train and evaluate USP, GRASP, or TGRASP with the main experiment settings."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(args):
    profiles = json.loads((ROOT/'configs/main_table.json').read_text(encoding='utf-8'))
    profile = profiles[args.dataset][args.backbone]
    if not math.isfinite(args.sigma) or args.sigma < 0:
        raise ValueError('sigma must be finite and nonnegative')
    if args.seed < 0:
        raise ValueError('seed must be nonnegative')
    if args.sigma == 0:
        method = 'CLEAN'
    else:
        if args.method == 'CLEAN':
            raise ValueError('CLEAN requires --sigma 0')
        method = args.method
    run_id = f'{args.dataset}_{args.backbone}_{method}_sigma{args.sigma:g}_seed{args.seed}'
    if args.smoke:
        run_id += '_smoke'
    work = args.output.resolve()/run_id
    command = ['--model','MPNN','--dataset',args.dataset,'--gnn',args.backbone,
               '--data_dir',args.data_dir.resolve().as_posix()+'/',
               '--device', '0','--runs','1','--seed',str(args.seed),'--save_model',
               '--defence_sigma',str(args.sigma),'--defence_p','0.7' if method=='GRASP' else '0.0',
               '--defence_k','1.5','--defence_limit','--defence_stage_train','--defence_stage_test',
               '--attack_pair_samples','0','--attack_node_scope','all']
    if args.device == 'cpu': command.append('--cpu')
    elif not args.device.isdecimal(): raise ValueError('device must be a GPU index or cpu')
    for key,value in profile.items():
        if key=='epochs' and args.smoke: value=2
        if isinstance(value,bool):
            if value: command.append('--'+key)
        else: command += ['--'+key,str(value)]
    group = None
    if method=='TGRASP':
        group = args.groups.resolve()/f'{args.dataset}.npy'
        command += ['--defense_group_noise','--defense_group_algorithm','tgrasp',
                    '--group_dir',(work/'groups').as_posix()]
    return dict(dataset=args.dataset,backbone=args.backbone,method=method,sigma=args.sigma,seed=args.seed,
                smoke=args.smoke,device=args.device,work_dir=str(work),group_file=str(group) if group else None,
                data_dir=str(args.data_dir.resolve()),epochs=2 if args.smoke else profile['epochs'],
                attack_backend='blockwise_exact' if args.dataset=='ogbn-arxiv' else 'dense_exact',
                training_args=command,profile=profile,
                profile_file_sha256=sha(ROOT/'configs/main_table.json'),
                tgrasp_config_sha256=sha(ROOT/'configs/tgrasp.json'))


def main():
    profiles=json.loads((ROOT/'configs/main_table.json').read_text(encoding='utf-8'))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=sorted(profiles), default='cora')
    p.add_argument('--backbone',choices=['gcn','gat'],default='gcn')
    p.add_argument('--method',choices=['USP','GRASP','TGRASP','CLEAN'],default='TGRASP')
    p.add_argument('--sigma',type=float,default=1.)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='0',help='CUDA device index, or cpu')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--groups',type=Path,default=ROOT/'assets/groups')
    p.add_argument('--output',type=Path,default=ROOT/'runs')
    p.add_argument('--smoke',action='store_true',help='Two-epoch environment check, excluded from result summaries')
    p.add_argument('--dry-run',action='store_true',help='Print resolved parameters; no training/downloads')
    a=p.parse_args()
    try: plan=make_plan(a)
    except ValueError as e: p.error(str(e))
    if a.dry_run:
        print(json.dumps(plan,indent=2)); return
    if plan['group_file'] and not Path(plan['group_file']).is_file():
        p.error(f"Missing partition. Run: python prepare_groups.py --dataset {a.dataset} --data-dir {a.data_dir}")
    work=Path(plan['work_dir'])
    if work.exists():
        p.error(f'Run directory already exists; choose a different --output. Not overwriting: {work}')
    # Fixed paper splits are copied only when absent; a different split is an error.
    split=ROOT/f'assets/splits/{a.dataset}_split.npz'
    if split.is_file():
        dest=a.data_dir/split.name
        if dest.exists() and sha(dest)!=sha(split):
            p.error(f'Existing split differs from packaged split: {dest}')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): shutil.copyfile(split,dest)
    work.mkdir(parents=True,exist_ok=False)
    plan['started_at_unix']=time.time()
    if plan['group_file']: plan['group_sha256']=sha(plan['group_file'])
    code_files = [
        *ROOT.glob('*.py'),
        *(ROOT / 'coloring').rglob('*.py'),
        *(ROOT / 'diagnostics').rglob('*.py'),
        ROOT / 'scripts' / 'run_entry.py',
    ]
    plan['code_sha256']={f.relative_to(ROOT).as_posix():sha(f) for f in sorted(code_files)}
    launch=work/'launch.json'
    launch.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    env=os.environ.copy()
    # Use the settings recorded for this run.
    for key in list(env):
        if key.startswith(('GRASP_','CAMPAIGN_')): env.pop(key)
    env.update(PYTHONHASHSEED='0',PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',
               GRASP_NORMALIZATION_PROFILE='kdd_table1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    env['CUDA_VISIBLE_DEVICES']='' if a.device=='cpu' else a.device
    print(f'Running {plan["dataset"]} / {plan["backbone"]} / {plan["method"]}; log: {work/"train.log"}',flush=True)
    with (work/'train.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_entry.py'),str(launch)],
                              cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
    (work/'exit.json').write_text(json.dumps(dict(returncode=result.returncode,finished_at_unix=time.time()))+'\n')
    if result.returncode:
        print((work/'train.log').read_text(encoding='utf-8')[-5000:],file=sys.stderr)
        raise SystemExit(result.returncode)
    if not (work/'result.json').is_file(): raise RuntimeError('Training returned without result.json')
    print((work/'result.json').read_text(encoding='utf-8'))


if __name__=='__main__': main()
