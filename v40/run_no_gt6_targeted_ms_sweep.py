"""Controlled validation-only final-MS bandwidth sweep; fixed 18->GRU->Kalman->6x6 architecture."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'output/no_gt_6x6_20260927_a/cross_prior_sweep/cross_prior_0.00'
OUT=HERE/'output/no_gt_6x6_20261008_ms_sweep'
CASES=[('bandwidth12','12','0'),('bandwidth18','18','5'),('bandwidth35','35','6')]

def main():
    settings=json.loads((SOURCE/'run_settings.json').read_text())['environment']
    original=json.loads((SOURCE/'report.json').read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for name,bw,gpu in CASES:
        dest=OUT/name;dest.mkdir(exist_ok=True)
        env=os.environ.copy();env.update(settings)
        env.update({'CUDA_VISIBLE_DEVICES':gpu,'MS_GRID_SIZE':'6','MS_BANDWIDTH_M':bw,
                    'V40_FEATURE_CACHE_ROOT':str(dest/'cache'),
                    'V40_GT_FRONT18_ORACLE':'0','V40_GT_FRONT18_PATCH_ORACLE':'0',
                    'V40_GT_FINAL_MS_ORACLE_CENTER':'0','V40_GT_FINAL_MS_PATCH_ORACLE':'0',
                    'V40_GT_JITTER_EVAL_M':'0','V40_GT_JITTER_PATCH_SELECT':'0',
                    'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        cmd=[sys.executable,'-u',str(HERE/'evaluate_official96_validation.py'),
             '--checkpoint',original['checkpoint'],'--city','citya',
             '--tta-rotations','0,1,2,3','--output',str(dest)]
        log=(dest/'console.log').open('w')
        p=subprocess.Popen(cmd,cwd=HERE.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((name,bw,gpu,p,log,dest))
        print('[START]',name,'GPU',gpu,'PID',p.pid,flush=True)
    summary=[]
    for name,bw,gpu,p,log,dest in jobs:
        status=p.wait();log.close()
        if status:raise RuntimeError(f'{name} failed ({status}); see {dest/"console.log"}')
        result=json.loads((dest/'report.json').read_text())
        saved=json.loads((dest/'run_settings.json').read_text())['environment']
        assert result['valid_no_GT_inference_result'] and not result['current_frame_GT_model_input']
        assert saved['MS_GRID_SIZE']=='6' and saved['MS_BANDWIDTH_M']==bw
        frames=list(csv.DictReader(next((dest/'citya').glob('*frames.csv')).open()))
        worst=sorted(frames,key=lambda row:float(row['error_final_m']),reverse=True)[:3]
        row={'name':name,'GPU':gpu,'final_ms_bandwidth_m':float(bw),'MLE_m':result['MLE_m'],
             'P90_m':result['P90_m'],'LSR_at_15_pct':result['LSR@15_pct'],
             'worst_frames':[(x['frame_id'],round(float(x['error_final_m']),3)) for x in worst],
             'report':str(dest/'report.json')}
        summary.append(row);print('[RESULT]',json.dumps(row),flush=True)
    (OUT/'summary.json').write_text(json.dumps({'baseline_MLE_m':original['MLE_m'],
         'baseline_report':str(SOURCE/'report.json'),'cases':summary},indent=2)+'\n')
    print('[DONE]',OUT/'summary.json',flush=True)

if __name__=='__main__':main()
