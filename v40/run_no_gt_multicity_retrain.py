#!/usr/bin/env python3
"""Train city-specific GT-loss visual models and a shared causal temporal model."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'output/no_gt_multicity_retrain_20261009'
SETTINGS=HERE/'output/no_gt_6x6_20260927_a/cross_prior_sweep/cross_prior_0.00/run_settings.json'
CITYA_VISUAL=HERE/'output/no_gt_6x6_20260927_a/visual_search/continue_seed2061/train/checkpoints/citya/visual_retrieval_A_only.pt'


def env(gpu,visual_root=None,cache_root=None):
 e=os.environ.copy();e.update(json.loads(SETTINGS.read_text())['environment'])
 e.update({'CUDA_VISIBLE_DEVICES':str(gpu),'MS_GRID_SIZE':'6','V40_GT_FRONT18_ORACLE':'0',
           'V40_GT_FRONT18_PATCH_ORACLE':'0','V40_GT_FINAL_MS_ORACLE_CENTER':'0',
           'V40_GT_FINAL_MS_PATCH_ORACLE':'0','V40_GT_JITTER_EVAL_M':'0',
           'V40_GT_JITTER_PATCH_SELECT':'0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
 e.pop('V40_VISUAL_CHECKPOINT_FILE',None)
 e.pop('V40_VISUAL_CHECKPOINT_ROOT',None)
 if visual_root:e['V40_VISUAL_CHECKPOINT_ROOT']=str(visual_root)
 if cache_root:e['V40_FEATURE_CACHE_ROOT']=str(cache_root)
 return e


def run(label,gpu,cmd,out,environment):
 out.mkdir(parents=True,exist_ok=True)
 with (out/'run.log').open('w') as log:
  process=subprocess.Popen([sys.executable,'-u',str(HERE/cmd[0]),*cmd[1:]],cwd=HERE,env=environment,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  for line in process.stdout:
   print(f'[{label} GPU{gpu}] {line}',end='',flush=True);log.write(line);log.flush()
  code=process.wait()
 if code:raise RuntimeError(f'{label} failed with code {code}; see {out}/run.log')


def visual_stage():
 jobs=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for city,gpu in [('cityb',0),('cityc',5),('cityd',6)]:
   out=OUT/'visual_base'/city
   cmd=['train_visual_global_gtloss.py','--city',city,'--samples','18000','--epochs','10',
        '--batch-size','128','--lr','0.0003','--target-sigma-m','4',
        '--distance-loss-weight','5','--local-loss-weight','3','--local-topk-windows','3',
        '--official96-train-only','--cache-root',str(OUT/'visual_cache'/city),
        '--output',str(out),'--seed','2033']
   jobs.append(pool.submit(run,city,gpu,cmd,out,env(gpu)))
  for job in as_completed(jobs):job.result()


def backbone_stage():
 jobs=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for city,gpu in [('cityb',0),('cityc',5),('cityd',6)]:
   out=OUT/'visual_finetune'/city
   source=OUT/'visual_base'/city/'checkpoints'/city/'visual_retrieval_A_only.pt'
   assert source.is_file()
   cmd=['train_visual_backbone_gtloss.py','--city',city,'--source-checkpoint',str(source),
        '--samples','18000','--epochs','1','--batch-size','48',
        '--backbone-lr','5e-6','--head-lr','3e-5','--unfreeze-last-blocks','8',
        '--target-sigma-m','4','--distance-loss-weight','10',
        '--live-sat-topk','16','--live-sat-random','16','--uav-quarter-turn-augment',
        '--seed','2061','--output',str(out)]
   jobs.append(pool.submit(run,city,gpu,cmd,out,env(gpu)))
  for job in as_completed(jobs):job.result()


def temporal_stage():
 root=OUT/'visual_checkpoints';root.mkdir(parents=True,exist_ok=True)
 sources={'citya':CITYA_VISUAL}
 for city in ('cityb','cityc','cityd'):
  sources[city]=OUT/'visual_finetune'/city/'checkpoints'/city/'visual_retrieval_A_only.pt'
 for city,source in sources.items():
  assert source.is_file(),source
  dst=root/city/'visual_retrieval_A_only.pt';dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():dst.symlink_to(source.resolve())
 out=OUT/'temporal'
 cmd=['train_multicity_routes.py','--train-cities','citya','cityb','cityc','cityd',
      '--eval-cities','citya','cityb','cityc','cityd',
      '--supervision','gt_train','--init','random_temporal','--epochs','10',
      '--max-train-frames','136','--lr','5e-6','--dense-offbank-loss',
      '--acquisition-loss-weight','1','--acquisition-target-sigma-m','4',
      '--validate-every-epochs','1','--early-stop-patience','4','--skip-test','--seed','2033',
      '--output',str(out)]
 run('temporal',0,cmd,out,env(0,root,OUT/'temporal_feature_cache'))


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 visual_stage();backbone_stage();temporal_stage()
 print('[TRAINING COMPLETE]',OUT,flush=True)

if __name__=='__main__':main()
