"""Three parallel CityA no-GT validation replays with final MS fixed to 6x6."""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
BEST=HERE/'checkpoint1/best_no_gt_20260926'
OUT=HERE/'output/no_gt_6x6_20260927_a'
SETTINGS=json.loads((BEST/'run_settings_original.json').read_text())['environment']
FUSED=HERE/'output/fused_temporal_citya_grid_parallel_20260926/grid_06x06/train/best_temporal_interim.pt'
CASES=[
 ('legacy_best_visual', '0', BEST/'best_temporal_interim.pt', BEST/'visual_retrieval_A_only.pt','0'),
 ('fused_best_visual', '1', FUSED, BEST/'visual_retrieval_A_only.pt','5'),
 ('legacy_parent_visual', '0', BEST/'best_temporal_interim.pt', BEST/'parent_epoch1_visual.pt','6'),
]

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 jobs=[]
 for name,fused,temporal,visual,gpu in CASES:
  if not temporal.is_file() or not visual.is_file():raise FileNotFoundError(name)
  env=os.environ.copy();env.update(SETTINGS)
  env.update({'CUDA_VISIBLE_DEVICES':gpu,'MS_GRID_SIZE':'6',
              'V40_TEMPORAL_FUSION_MLP':fused,
              'V40_VISUAL_CHECKPOINT_FILE':str(visual),
              'V40_FEATURE_CACHE_ROOT':str(OUT/name/'cache'),
              'V40_GT_FRONT18_ORACLE':'0','V40_GT_FRONT18_PATCH_ORACLE':'0',
              'V40_GT_FINAL_MS_ORACLE_CENTER':'0','V40_GT_FINAL_MS_PATCH_ORACLE':'0',
              'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
  dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,'-u',str(HERE/'evaluate_official96_validation.py'),
       '--checkpoint',str(temporal),'--city','citya','--tta-rotations','0,1,2,3',
       '--output',str(dest)]
  log=(dest/'console.log').open('w')
  p=subprocess.Popen(cmd,cwd=HERE.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
  jobs.append((name,gpu,p,log,dest));print('[START]',name,'gpu',gpu,'pid',p.pid,flush=True)
 failed=[]
 for name,gpu,p,log,dest in jobs:
  code=p.wait();log.close()
  if code:failed.append((name,code,str(dest/'console.log')))
  else:
   r=json.loads((dest/'report.json').read_text())
   if not r['valid_no_GT_inference_result'] or r['current_frame_GT_model_input']:
    raise RuntimeError('GT inference detected in '+name)
   settings=json.loads((dest/'run_settings.json').read_text())
   if settings['environment']['MS_GRID_SIZE']!='6':raise RuntimeError('MS not 6x6')
   print('[RESULT]',name,'gpu',gpu,'MLE',r['MLE_m'],'P90',r['P90_m'],flush=True)
 if failed:raise RuntimeError(str(failed))
 results={name:json.loads((OUT/name/'report.json').read_text())['MLE_m'] for name,*_ in CASES}
 (OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
 print('[DONE]',OUT/'summary.json',flush=True)

if __name__=='__main__':main()
