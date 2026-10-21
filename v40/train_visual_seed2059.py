#!/usr/bin/env python3
"""City A visual fine-tuning with GT only in loss and UAV rotation augmentation."""
import json,os,subprocess,sys
from pathlib import Path
v40=Path(__file__).resolve().parent;root=v40.parent
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
source=v40/'checkpoint1/best_no_gt_20260926/parent_epoch1_visual.pt'
settings=json.loads((v40/'checkpoint1/best_no_gt_20260926/seed2059_environment.json').read_text())
env=os.environ.copy();env.update({k:str(v) for k,v in settings['environment'].items()});env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
cmd=[sys.executable,'-u','v40/train_visual_backbone_gtloss.py','--city','citya','--source-checkpoint',str(source),'--samples','18000','--epochs','1','--batch-size','48','--backbone-lr','0.000005','--head-lr','0.00003','--unfreeze-last-blocks','8','--target-sigma-m','4','--distance-loss-weight','10','--live-sat-topk','16','--live-sat-random','16','--uav-quarter-turn-augment','--seed','2059','--output',str(out)]
(out/'command.json').write_text(json.dumps({'gpu':0,'command':cmd},indent=2)+'\n')
with (out/'console.log').open('w') as f:rc=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
print('exit',rc,'output',out,flush=True);raise SystemExit(rc)
