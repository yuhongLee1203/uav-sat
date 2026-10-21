#!/usr/bin/env python3
"""Evaluate a completed temporal run with frozen augmented visual weights."""
import json,os,subprocess,sys,time
from pathlib import Path
v40=Path(__file__).resolve().parent;root=v40.parent
train=v40/'output/causal_4m_temporal_augmented_visual_20260926_a'
out=v40/'output/causal_4m_temporal_augmented_visual_tta_20260926_a'
out.mkdir(parents=True,exist_ok=False)
while not (train/'shared_temporal.pt').exists():time.sleep(10)
base=json.loads((v40/'output/causal_4m_visual_augmented_mild_epoch1_prior_20260926_a/run_settings.json').read_text())
env=os.environ.copy();env.update({k:str(v) for k,v in base['environment'].items()})
env.update({'CUDA_VISIBLE_DEVICES':'5','V40_FEATURE_CACHE_ROOT':str(out/'cache'),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
cmd=[sys.executable,'-u','v40/evaluate_official96_validation.py','--city','citya','--checkpoint',str(train/'best_temporal_interim.pt'),'--output',str(out),'--tta-rotations','0,1,2,3']
(out/'command.json').write_text(json.dumps({'command':cmd,'environment':{k:v for k,v in env.items() if k.startswith(('V40_','MS_','UAVSAT_'))}},indent=2)+'\n')
with (out/'console.log').open('w') as f:rc=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
if rc:raise RuntimeError(str(out/'console.log'))
report=json.loads((out/'report.json').read_text());print(json.dumps({'MLE_m':report['MLE_m'],'report':str(out/'report.json')}),flush=True)
