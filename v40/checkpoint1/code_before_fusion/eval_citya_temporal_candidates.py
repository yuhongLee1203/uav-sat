#!/usr/bin/env python3
"""Validation-only comparison of independently trained GT-loss temporal weights."""
import json,os,subprocess,sys
from pathlib import Path
v40=Path(__file__).resolve().parent;root=v40.parent
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
base=json.loads((v40/'output/causal_4m_tta_ms_20260926_a/grid10_bw12/run_settings.json').read_text())
env=os.environ.copy();env.update({k:str(v) for k,v in base['environment'].items()});env.update({'CUDA_VISIBLE_DEVICES':'6','V40_ACQ_RAW_VISUAL_WEIGHT':'20','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
variants=[('shifted',v40/'output/causal_4m_search_20260926_a/temporal_shifted/best_temporal_interim.pt'),('corrected_grid',v40/'output/causal_4m_corrected_grid_20260926_a/best_temporal_interim.pt'),('corrected_measurement',v40/'output/causal_4m_corrected_measurement_20260926_a/best_temporal_interim.pt')]
rows=[]
for name,ckpt in variants:
 trial=out/name;trial.mkdir();e=dict(env);e['V40_FEATURE_CACHE_ROOT']=str(out/'cache')
 cmd=[sys.executable,'-u','v40/evaluate_official96_validation.py','--city','citya','--checkpoint',str(ckpt),'--output',str(trial),'--tta-rotations','0,1,2,3']
 with (trial/'console.log').open('w') as f:rc=subprocess.run(cmd,cwd=root,env=e,stdout=f,stderr=subprocess.STDOUT).returncode
 if rc:print('FAIL',name,flush=True);continue
 report=json.loads((trial/'report.json').read_text());row={'name':name,'MLE_m':report['MLE_m'],'P90_m':report['P90_m']};rows.append(row);print(row,flush=True)
 (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
