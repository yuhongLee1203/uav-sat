"""Screen upstream causal-search settings; final Mean Shift is fixed at 6x6."""
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
BEST=HERE/'checkpoint1/best_no_gt_20260926'
BASE=json.loads((BEST/'run_settings_original.json').read_text())['environment']
OUT=HERE/'output/no_gt_6x6_20260927_a/combination_sweep'
TEMP=HERE/'output/fused_temporal_citya_grid_parallel_20260926/grid_06x06/train/best_temporal_interim.pt'
CASES=[
 ('extra_weight10',{'V40_IMAGE_GUIDED_EXTRA_HYPOTHESIS':'1','V40_ACQ_RAW_VISUAL_WEIGHT':'10'},False),
 ('extra_weight5',{'V40_IMAGE_GUIDED_EXTRA_HYPOTHESIS':'1','V40_ACQ_RAW_VISUAL_WEIGHT':'5'},False),
 ('extra_weight15',{'V40_IMAGE_GUIDED_EXTRA_HYPOTHESIS':'1','V40_ACQ_RAW_VISUAL_WEIGHT':'15'},False),
 ('guided_weight10',{'V40_IMAGE_GUIDED_FRONT18':'1','V40_ACQ_RAW_VISUAL_WEIGHT':'10'},False),
 ('extra_bank40',{'V40_IMAGE_GUIDED_EXTRA_HYPOTHESIS':'1','V40_ACQ_BANK_RADIUS_M':'40'},False),
 ('extra_bank80',{'V40_IMAGE_GUIDED_EXTRA_HYPOTHESIS':'1','V40_ACQ_BANK_RADIUS_M':'80'},False),
]
def run_case(gpu,name,extra,flips):
 dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(BASE)
 env.update({'CUDA_VISIBLE_DEVICES':str(gpu),'MS_GRID_SIZE':'6',
             'V40_TEMPORAL_FUSION_MLP':'1',
             'V40_VISUAL_CHECKPOINT_FILE':str(BEST/'visual_retrieval_A_only.pt'),
             'V40_FEATURE_CACHE_ROOT':str(dest/'cache'),
             'V40_GT_FRONT18_ORACLE':'0','V40_GT_FRONT18_PATCH_ORACLE':'0',
             'V40_GT_FINAL_MS_ORACLE_CENTER':'0','V40_GT_FINAL_MS_PATCH_ORACLE':'0',
             'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
 env.update(extra)
 cmd=[sys.executable,'-u',str(HERE/'evaluate_official96_validation.py'),
      '--checkpoint',str(TEMP),'--city','citya','--tta-rotations','0,1,2,3',
      '--output',str(dest)]
 if flips:cmd.append('--tta-flips')
 with (dest/'console.log').open('w') as f: code=subprocess.run(cmd,cwd=HERE.parent,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 if code:raise RuntimeError(name+' failed: '+str(dest/'console.log'))
 report=json.loads((dest/'report.json').read_text())
 if not report['valid_no_GT_inference_result'] or report['current_frame_GT_model_input']:
  raise RuntimeError(name+' used GT')
 settings=json.loads((dest/'run_settings.json').read_text())['environment']
 if settings.get('MS_GRID_SIZE')!='6':raise RuntimeError(name+' changed final grid')
 print('[RESULT]',gpu,name,report['MLE_m'],report['P90_m'],flush=True)
 return {'case':name,'gpu':gpu,'MLE_m':report['MLE_m'],'P90_m':report['P90_m'],'LSR@15_pct':report['LSR@15_pct']}

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  fs=[pool.submit(run_case,[0,5,6][i%3],name,extra,flips)
      for i,(name,extra,flips) in enumerate(CASES)]
  rows=[f.result() for f in fs]
 (OUT/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x['MLE_m']),indent=2)+'\n')
 print('[DONE]',OUT/'summary.json',flush=True)

if __name__=='__main__':main()
