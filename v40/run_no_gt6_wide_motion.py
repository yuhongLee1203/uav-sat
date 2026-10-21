"""Warm-start two GT-loss-only causal 6x6 temporal models on train-only wide-step routes."""
import json,os,sys
from pathlib import Path
from run_fused_temporal_citya import launch,wait_all,say

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'output/no_gt_6x6_20260927_a/cross_prior_sweep/cross_prior_0.00'
VISUAL=HERE/'output/no_gt_6x6_20260927_a/visual_search/continue_seed2061/train/checkpoints/citya/visual_retrieval_A_only.pt'
VISUAL_ROOT=VISUAL.parent.parent
WIDE=HERE/'data/official96_fixed_train_grid_wide'
OUT=HERE/'output/no_gt_6x6_20261008_wide_motion'
CASES=[('acq1_sigma4','0','1','4'),('acq2_sigma4','5','2','4')]
BASE=json.loads((SOURCE/'run_settings.json').read_text())['environment']
CHECKPOINT=Path(json.loads((SOURCE/'report.json').read_text())['checkpoint'])

def environment(gpu,dest,train):
 env=os.environ.copy();env.update(BASE)
 env.update({'CUDA_VISIBLE_DEVICES':gpu,'MS_GRID_SIZE':'6','MS_BANDWIDTH_M':'18',
  'V40_TEMPORAL_FUSION_MLP':'1','V40_IMAGE_GUIDED_EXTRA_HYPOTHESIS':'1',
  'V40_VISUAL_CHECKPOINT_ROOT':str(VISUAL_ROOT),'V40_VISUAL_CHECKPOINT_FILE':str(VISUAL),
  'V40_FEATURE_CACHE_ROOT':str(dest/('train_cache' if train else 'validation_cache')),
  'V40_GT_FRONT18_ORACLE':'0','V40_GT_FRONT18_PATCH_ORACLE':'0',
  'V40_GT_FINAL_MS_ORACLE_CENTER':'0','V40_GT_FINAL_MS_PATCH_ORACLE':'0',
  'V40_GT_JITTER_EVAL_M':'0','V40_GT_JITTER_PATCH_SELECT':'0',
  'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
 return env

def main():
 if not VISUAL.is_file() or not CHECKPOINT.is_file():raise FileNotFoundError('missing source checkpoint')
 audit=json.loads((WIDE/'citya/audit.json').read_text())
 assert audit['no_cross_split_samples'] and audit['no_duplicate_extra_rows']
 OUT.mkdir(parents=True,exist_ok=True)
 jobs=[]
 for name,gpu,weight,sigma in CASES:
  dest=OUT/name;cmd=[sys.executable,'-u',str(HERE/'train_multicity_routes.py'),
   '--train-cities','citya','--eval-cities','citya','--supervision','gt_train',
   '--init','random_temporal','--epochs','4','--max-train-frames','136','--lr','1e-5',
   '--dense-offbank-loss','--acquisition-loss-weight',weight,
   '--acquisition-target-sigma-m',sigma,'--extra-train-routes-dir',str(WIDE),
   '--validate-every-epochs','1','--early-stop-patience','3','--skip-test',
   '--seed','2033','--init-temporal-checkpoint',str(CHECKPOINT),
   '--output',str(dest/'train')]
  label=f'{name} GPU{gpu} TRAIN';jobs.append((label,*launch(cmd,environment(gpu,dest,True),label,dest/'train.log',4*8)))
 wait_all(jobs)
 jobs=[]
 for name,gpu,weight,sigma in CASES:
  dest=OUT/name;ck=dest/'train/best_temporal_interim.pt'
  if not ck.is_file():raise FileNotFoundError(ck)
  cmd=[sys.executable,'-u',str(HERE/'evaluate_official96_validation.py'),
   '--checkpoint',str(ck),'--city','citya','--tta-rotations','0,1,2,3',
   '--output',str(dest/'validation')]
  label=f'{name} GPU{gpu} VAL';jobs.append((label,*launch(cmd,environment(gpu,dest,False),label,dest/'validation.log')))
 wait_all(jobs)
 summary=[]
 for name,gpu,weight,sigma in CASES:
  dest=OUT/name;report=json.loads((dest/'validation/report.json').read_text());settings=json.loads((dest/'validation/run_settings.json').read_text())
  assert report['valid_no_GT_inference_result'] and not report['current_frame_GT_model_input']
  assert settings['environment']['MS_GRID_SIZE']=='6'
  summary.append({'name':name,'gpu':gpu,'acquisition_loss_weight':float(weight),'target_sigma_m':float(sigma),
   'MLE_m':report['MLE_m'],'P90_m':report['P90_m'],'LSR_at_15_pct':report['LSR@15_pct'],
   'selected_capture_pct':report['SelectedCandidateCapture_pct'],
   'checkpoint':str(dest/'train/best_temporal_interim.pt'),'report':str(dest/'validation/report.json')})
 summary.sort(key=lambda x:x['MLE_m'])
 (OUT/'summary.json').write_text(json.dumps({'baseline_MLE_m':json.loads((SOURCE/'report.json').read_text())['MLE_m'],
  'train_episode_count':len(audit['episodes']),'train_episode_frames':sum(v['frames'] for v in audit['episodes'].values()),
  'cases':summary},indent=2)+'\n')
 say('[DONE] '+str(OUT/'summary.json'))
 for row in summary:say('[RESULT] '+json.dumps(row))

if __name__=='__main__':main()
