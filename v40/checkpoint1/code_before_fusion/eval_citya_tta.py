#!/usr/bin/env python3
"""Compare rotation averaging on fixed City A validation without GT inputs."""
import json, os, subprocess, sys
from pathlib import Path
v40=Path(__file__).resolve().parent
root=v40.parent
out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=False)
base=json.loads((v40/'output/official96_fixed_citya_val_joint8_blend0_20260925/run_settings.json').read_text())
env=os.environ.copy(); env.update({k:str(v) for k,v in base['environment'].items()})
env.update({'CUDA_VISIBLE_DEVICES':'6','V40_VISUAL_CHECKPOINT_ROOT':str(v40/'output/causal_4m_search_20260926_a/visual_epoch2_snapshot/checkpoints'),'V40_GT_FRONT18_ORACLE':'0','V40_GT_FRONT18_PATCH_ORACLE':'0','V40_GT_FINAL_MS_ORACLE_CENTER':'0','V40_GT_FINAL_MS_PATCH_ORACLE':'0','V40_FINAL_MS_LATTICE_CENTER':'visual_anchor','V40_MEANSHIFT_SCORE_TAU':'0.5','MS_GRID_SIZE':'10','MS_BANDWIDTH_M':'7','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
rows=[]
for name,rot in [('rot02','0,2'),('rot0123','0,1,2,3')]:
 trial=out/name;trial.mkdir();e=dict(env);e['V40_FEATURE_CACHE_ROOT']=str(trial/'cache')
 cmd=[sys.executable,'-u','v40/evaluate_official96_validation.py','--city','citya','--checkpoint',str(v40/'output/causal_4m_search_20260926_a/temporal_grid/best_temporal_interim.pt'),'--output',str(trial),'--tta-rotations',rot]
 with (trial/'console.log').open('w') as f:rc=subprocess.run(cmd,cwd=root,env=e,stdout=f,stderr=subprocess.STDOUT).returncode
 if rc:raise RuntimeError(str(trial/'console.log'))
 report=json.loads((trial/'report.json').read_text());row={'name':name,'rotations':rot,'MLE_m':report['MLE_m'],'P90_m':report['P90_m']};rows.append(row);print(row,flush=True)
(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
