#!/usr/bin/env python3
"""Cross-city held-out replay of the fixed 6x6 GT-free CityA model.

The learned visual/temporal weights are unchanged. Each city's satellite gallery
is rebuilt with that city's map and the same visual backbone. GT is used only
for post-inference metrics/plots, never for acquisition or final decoding.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'report/add_more/picture_noGT'
SOURCE_SETTINGS=HERE/'output/no_gt_6x6_20260927_a/cross_prior_sweep/cross_prior_0.00/run_settings.json'
VISUAL=HERE/'output/no_gt_6x6_20260927_a/visual_search/continue_seed2061/train/checkpoints/citya/visual_retrieval_A_only.pt'
TEMPORAL=HERE/'output/no_gt_6x6_20260927_a/temporal_visual2061/raw10_sigma4/train/best_temporal_interim.pt'
PREPARED=HERE/'data/official96_fixed_routes'
CITIES=('citya','cityb','cityc','cityd')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def manifest_paths(city,route):
    with (PREPARED/city/'prepared/routes'/route/'manifest.csv').open(newline='') as f:
        return {r['image_path'] for r in csv.DictReader(f)}


def setup_env(gpu, output):
    e=os.environ.copy()
    e.update(json.loads(SOURCE_SETTINGS.read_text())['environment'])
    e.update({'CUDA_VISIBLE_DEVICES':str(gpu),'V40_FEATURE_CACHE_ROOT':str(output/'feature_cache'),
              'V40_VISUAL_CHECKPOINT_FILE':str(VISUAL),'V40_GT_FRONT18_ORACLE':'0',
              'V40_GT_FRONT18_PATCH_ORACLE':'0','V40_GT_FINAL_MS_ORACLE_CENTER':'0',
              'V40_GT_FINAL_MS_PATCH_ORACLE':'0','V40_GT_JITTER_EVAL_M':'0',
              'V40_GT_JITTER_PATCH_SELECT':'0','MS_GRID_SIZE':'6',
              'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    return e


def worker(city,output):
    import torch
    import image_only_citya as exp
    import visual_localizer as vl
    from train_multicity_routes import evaluate_city
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type=='cuda', 'GPU is required for this replay'
    output.mkdir(parents=True,exist_ok=True)
    assert VISUAL.is_file() and TEMPORAL.is_file()
    source=torch.load(VISUAL,map_location='cpu',weights_only=False)
    assert source.get('gallery_city')=='citya'
    assert source.get('training_protocol')=='global_gallery_gt_loss_only'
    assert source.get('previous_task_checkpoint_loaded') is False
    assert source.get('official96_train_only') is True
    citya_test=manifest_paths('citya','test_01')|manifest_paths('citya','test_02')
    assert source['excluded_test_manifest_sha256']==hashlib.sha256('\n'.join(sorted(citya_test)).encode()).hexdigest()
    train=manifest_paths(city,'train_01')
    test1=manifest_paths(city,'test_01')
    test2=manifest_paths(city,'test_02')
    assert train and test1 and test2 and not(train&test1 or train&test2 or test1&test2)
    if city!='citya':
        assert not any(p.startswith(f'/yh/study/cvpr_data/Bearing_UAV_90K/citya/') for p in test1|test2)
        exp.configure(city)
        visual=exp.FrozenVisualLocalizer(device)
        first=next(csv.DictReader((PREPARED/city/'prepared/routes/train_01/manifest.csv').open(newline='')))
        origin_lat=float(first['y_m']);origin_lon=float(first['x_m'])
        gallery=vl._build_satellite_backbone_gallery(visual.model,origin_lat,origin_lon,device)
        adapted=dict(source)
        adapted['gallery']=gallery
        adapted['origin_lat']=origin_lat
        adapted['origin_lon']=origin_lon
        adapted['gallery_city']=city
        adapted['cross_city_transfer_from']='citya'
        adapted['transfer_note']='Same CityA-trained weights; city satellite gallery rebuilt without test GT.'
        adapted['source_visual_checkpoint_sha256']=digest(VISUAL)
        adapted['satellite_image_sha256']=digest(exp.config.SAT_IMAGE)
        adapted_path=output/'visual_city_gallery.pt'
        torch.save(adapted,adapted_path)
        del visual,adapted,gallery
        torch.cuda.empty_cache()
        os.environ['V40_VISUAL_CHECKPOINT_FILE']=str(adapted_path)
        exp.configure(city)
    else:
        exp.configure(city)
        exp.verify_visual_checkpoint_split()
        adapted_path=VISUAL
    checkpoint=torch.load(TEMPORAL,map_location='cpu',weights_only=False)
    exp.verify_temporal_correction_bounds(checkpoint)
    assert checkpoint['architecture']==exp.config.ARCHITECTURE_NAME
    assert exp.config.NO_GT_INFERENCE and exp.config.ROUTE_REFERENCE_ONLY
    assert not any((exp.config.GT_FRONT18_ORACLE,exp.config.GT_FRONT18_PATCH_ORACLE,
                    exp.config.GT_FINAL_MS_ORACLE_CENTER,exp.config.GT_FINAL_MS_PATCH_ORACLE))
    assert int(os.environ['MS_GRID_SIZE'])==6
    model=exp.ThreeFrameRouteStateGRU().to(device)
    model.load_state_dict(checkpoint['model'],strict=True)
    original=exp.FrozenVisualLocalizer.encode_uav_clip
    def encode_with_tta(self,image):
        return torch.stack([original(self,torch.rot90(image,k,(-2,-1))) for k in (0,1,2,3)]).mean(dim=0)
    exp.FrozenVisualLocalizer.encode_uav_clip=encode_with_tta
    results=evaluate_city(city,model,device,output)
    for route in ('test_01','test_02'):
        item=results[route]
        assert item['ValidNoGTInferenceResult'] and item['MS_GridSize']==6
        assert not any(item[k] for k in ('GTFront18Oracle','GTFront18PatchOracle','GTFinalMSCenterOracle','GTFinalMSPatchOracle'))
        assert Path(item['CSV']).is_file()
    report={'city':city,'protocol':'CityA-trained GT-loss-only model, GT-free inference, cross-city transfer' if city!='citya' else 'CityA-trained GT-loss-only model, GT-free inference',
            'citya_validation_only_mle_m':4.6677,'test_replay_not_model_selection':True,
            'visual_checkpoint_source':str(VISUAL),'visual_checkpoint_source_sha256':digest(VISUAL),
            'visual_checkpoint_with_city_gallery':str(adapted_path),
            'temporal_checkpoint':str(TEMPORAL),'temporal_checkpoint_sha256':digest(TEMPORAL),
            'gallery_city':city,'test_image_overlap_with_training_route':False,
            'tta_rotations':[0,1,2,3],'final_ms_grid_size':6,'current_frame_gt_model_input':False,
            'uses_supplied_route_reference':True,'results':results}
    (output/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print('[DONE]',city,[(r,round(results[r]['MLE_m'],3)) for r in ('test_01','test_02')],flush=True)


def launch(city,gpu):
    path=OUT/'inference'/city
    path.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',city,'--output',str(path)]
    with (path/'run.log').open('w') as log:
        p=subprocess.Popen(cmd,env=setup_env(gpu,path),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:
            print(f'[{city} GPU{gpu}] {line}',end='',flush=True)
            log.write(line);log.flush()
        code=p.wait()
    if code:raise RuntimeError(f'{city} GPU{gpu} failed with exit {code}; see {path}/run.log')
    return city


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',choices=CITIES)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.worker:
        if args.output is None:parser.error('--output is required with --worker')
        worker(args.worker,args.output)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(launch,c,g) for c,g in [('citya',0),('cityb',5),('cityc',6)]]
        for f in as_completed(futures):f.result()
    launch('cityd',0)
    from plot_no_gt_test_pictures import main as plot
    plot()


if __name__=='__main__':main()
