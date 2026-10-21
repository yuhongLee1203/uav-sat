#!/usr/bin/env python3
"""City A visual fine-tuning with GT only in loss and UAV rotation augmentation."""
import json,os,subprocess,sys
from pathlib import Path
v40=Path(__file__).resolve().parent;root=v40.parent
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
source=v40/'output/causal_4m_search_20260926_a/visual_continue/checkpoints/citya/visual_retrieval_A_only.pt'
settings=json.loads((v40/'output/official96_fixed_citya_val_joint8_blend0_20260925/run_settings.json').read_text())
env=os.environ.copy();env.update({k:str(v) for k,v in settings['environment'].items()});env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
cmd=[sys.executable,'-u','v40/train_visual_backbone_gtloss.py','--city','citya','--source-checkpoint',str(source),'--samples','18000','--epochs','4','--batch-size','48','--backbone-lr','0.000003','--head-lr','0.00002','--unfreeze-last-blocks','8','--target-sigma-m','2','--distance-loss-weight','20','--live-sat-topk','16','--live-sat-random','16','--uav-quarter-turn-augment','--output',str(out)]
(out/'command.json').write_text(json.dumps({'gpu':0,'command':cmd},indent=2)+'\n')
with (out/'console.log').open('w') as f:rc=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
print('exit',rc,'output',out,flush=True);raise SystemExit(rc)
