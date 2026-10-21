#!/usr/bin/env python3
"""Wait for a GT-loss visual run, then check its best checkpoint on City A val."""
import json,os,subprocess,sys,time
from pathlib import Path
v40=Path(__file__).resolve().parent;root=v40.parent
train=Path(sys.argv[1]);gpu=sys.argv[2];out=Path(sys.argv[3])
out.mkdir(parents=True,exist_ok=False)
report_file=train/'report.json'
while not report_file.exists():
 time.sleep(10)
train_report=json.loads(report_file.read_text())
(out/'train_selection.json').write_text(json.dumps({'train_report':str(report_file),'best_checkpoint_improved_in_this_run':train_report['best_checkpoint_improved_in_this_run'],'checkpoint':train_report['checkpoint']},indent=2)+'\n')
base=json.loads((v40/'output/causal_4m_large_ms_20260926_a/grid18_bw25/run_settings.json').read_text())
env=os.environ.copy();env.update({k:str(v) for k,v in base['environment'].items()})
env.update({'CUDA_VISIBLE_DEVICES':gpu,'V40_VISUAL_CHECKPOINT_ROOT':str(train/'checkpoints'),'V40_FEATURE_CACHE_ROOT':str(out/'cache'),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
cmd=[sys.executable,'-u','v40/evaluate_official96_validation.py','--city','citya','--checkpoint',str(v40/'output/causal_4m_search_20260926_a/temporal_grid/best_temporal_interim.pt'),'--output',str(out),'--tta-rotations','0,1,2,3']
(out/'command.json').write_text(json.dumps({'command':cmd,'environment':{k:v for k,v in env.items() if k.startswith(('V40_','MS_','UAVSAT_'))}},indent=2)+'\n')
with (out/'console.log').open('w') as f:rc=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
if rc:raise RuntimeError(str(out/'console.log'))
report=json.loads((out/'report.json').read_text())
print(json.dumps({'train':str(train),'MLE_m':report['MLE_m'],'P90_m':report['P90_m'],'validation':str(out/'report.json')}),flush=True)
