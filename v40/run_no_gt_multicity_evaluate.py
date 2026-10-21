#!/usr/bin/env python3
"""Evaluate the retrained four-city causal model, then draw satellite result maps."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import run_no_gt_multicity_retrain as train
HERE=train.HERE
RUN=train.OUT
OUT=HERE.parent/'report/add_more/picture_noGT/retrained'
VISUAL_ROOT=RUN/'visual_checkpoints'
TEMPORAL=RUN/'temporal/best_temporal_interim.pt'


def evaluate(city,gpu):
    dest=OUT/'inference'/city
    dest.mkdir(parents=True,exist_ok=True)
    visual=VISUAL_ROOT/city/'visual_retrieval_A_only.pt'
    assert visual.is_file() and TEMPORAL.is_file()
    e=train.env(gpu,VISUAL_ROOT,OUT/'feature_cache'/city)
    for kind,script in [('validation','evaluate_official96_validation.py'),
                        ('test','evaluate_official96_fixed_test.py')]:
        target=(OUT/'validation'/city if kind=='validation' else dest)
        target.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-u',str(HERE/script),'--checkpoint',str(TEMPORAL),
             '--city',city,'--tta-rotations','0,1,2,3','--output',str(target)]
        with (target/'run.log').open('w') as log:
            process=subprocess.Popen(cmd,env=e,cwd=HERE,stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(f'[{city} GPU{gpu} {kind}] {line}',end='',flush=True)
                log.write(line);log.flush()
            code=process.wait()
        if code:raise RuntimeError(f'{city} {kind} failed; see {target}/run.log')
        report=json.loads((target/'report.json').read_text())
        assert report['current_frame_GT_model_input'] is False
        if kind=='test':
            assert report['visual_checkpoint']==str(visual.resolve())
            for route in ('test_01','test_02'):
                item=report['results'][route]
                assert item['ValidNoGTInferenceResult'] and item['MS_GridSize']==6
                assert not any(item[k] for k in ('GTFront18Oracle','GTFront18PatchOracle','GTFinalMSCenterOracle','GTFinalMSPatchOracle'))
    return city


def main():
    assert TEMPORAL.is_file(),'Run run_no_gt_multicity_retrain.py first'
    OUT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(evaluate,c,g) for c,g in [('citya',0),('cityb',5),('cityc',6)]]
        for f in as_completed(futures):f.result()
    evaluate('cityd',0)
    e=os.environ.copy()
    e['NO_GT_PICTURE_ROOT']=str(OUT)
    e['NO_GT_RETRAINED']='1'
    subprocess.run([sys.executable,str(HERE/'plot_no_gt_test_pictures.py')],env=e,check=True)
    overview={'training_root':str(RUN),'selected_temporal_checkpoint':str(TEMPORAL),
              'visual_root':str(VISUAL_ROOT),'validation':{},'test':{}}
    for city in ('citya','cityb','cityc','cityd'):
        val=json.loads((OUT/'validation'/city/'report.json').read_text())
        test=json.loads((OUT/'inference'/city/'report.json').read_text())
        overview['validation'][city]={'MLE_m':val['MLE_m'],'P90_m':val['P90_m']}
        overview['test'][city]={r:{'MLE_m':test['results'][r]['MLE_m'],
                                   'LSR@15_pct':test['results'][r]['LSR@15_pct']}
                                for r in ('test_01','test_02')}
    (OUT/'training_validation_test_provenance.json').write_text(json.dumps(overview,indent=2)+'\n')
    top=OUT.parent/'README.md'
    top.write_text('# 無 GT 四城市結果\n\n正式重新訓練、四城市 validation 選 checkpoint 後的結果與衛星底圖在 [retrained/README.md](retrained/README.md)。先前 City A 權重直接跨城市的圖片是探索性 transfer pilot，已由此結果取代。\n')
    print(json.dumps(overview,indent=2),flush=True)

if __name__=='__main__':main()
