import csv,json,hashlib
from pathlib import Path
import numpy as np
R=Path('/yh/study/uav-sat');S=R/'v39_otherdata/output/original3759_inference_20260925_224113_1642368';O=R/'report/1016'
a=np.load(R/'report/add_more/thesis_experiments/raw/final_decoder_predictions.npz');files=[];index=0;max_score_diff=0.;max_pos_diff=0.
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for c in 'abcd':
 for route in ('test_01','test_02'):
  rows={}
  for variant in ('full','decoder_probe'):
   folder=S/f'city{c}/variants'/variant
   summary=folder/'bearing_v39_summary.json';f=folder/Path(json.loads(summary.read_text())[route]['CSV']).name
   with f.open(newline='') as handle:rows[variant]=list(csv.DictReader(handle))
   files.append({'path':str(f),'sha256':sha(f)})
  assert len(rows['full'])==len(rows['decoder_probe'])
  for full,probe in zip(rows['full'],rows['decoder_probe']):
   assert str(a['sample_key'][index])==f'{c}/{route}/{full["frame_id"]}'
   assert full['frame_id']==probe['frame_id'] and full['image_path']==probe['image_path']
   gt=[float(full['gt_x']),float(full['gt_y'])];p=[float(full['final_x']),float(full['final_y'])]
   assert np.max(np.abs(a['gt_xy'][index]-gt))<1e-6
   max_pos_diff=max(max_pos_diff,float(np.linalg.norm(a['mean_shift_xy'][index]-p)))
   assert np.max(np.abs(a['candidate_centers_xy'][index]-json.loads(probe['decoder_candidate_centers_json'])))<1e-6
   d=np.max(np.abs(a['candidate_regularized_logits'][index]-json.loads(probe['decoder_candidate_logits_json'])))
   max_score_diff=max(max_score_diff,float(d));index+=1
assert index==1482 and max_pos_diff<1e-3 and max_score_diff<1e-6
payload={'samples':index,'source_suite':str(S),'max_full_vs_saved_probe_position_difference_m':max_pos_diff,
         'max_saved_vs_source_logits_difference':max_score_diff,'source_files':files,
         'candidate_archive_sha256':sha(R/'report/add_more/thesis_experiments/raw/final_decoder_predictions.npz'),
         'gpu_allowed':'0 only','model_retrained':False,'sensitivity_scope':'fixed-candidate final decoder only'}
(O/'source_audit.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({k:v for k,v in payload.items() if k!='source_files'},indent=2))
