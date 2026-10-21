"""Rebuild publication figures from the original 1,482-frame v13 GT-assisted replay."""
from pathlib import Path
import csv
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'v39_otherdata/output/original3759_inference_20260925_220434_1615534'
PAPER_URL = 'https://arxiv.org/abs/2603.22153'
CITIES = ['citya', 'cityb', 'cityc', 'cityd']
COLORS = {'Reference baseline':'#8d99ae','w/o GRU':'#e78f49',
          'w/o Final Refinement':'#b56576','Full Model':'#277da1'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':130,'savefig.dpi':220})


def save(fig, relative):
    path = OUT / 'figures' / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def metric(errors, threshold):
    return float(np.mean(np.asarray(errors) < threshold) * 100)


def load_variant(variant):
    rows = []
    for city in CITIES:
        for route in ('route_B', 'route_C'):
            files = list((SOURCE / city / 'variants' / variant).glob(route + '*frames.csv'))
            if len(files) != 1:
                raise RuntimeError('expected one frame CSV for %s %s %s: %s' %
                                   (city, variant, route, files))
            frame = pd.read_csv(files[0])
            frame.insert(0,'city',city)
            frame.insert(1,'route',route)
            rows.append(frame)
    data = pd.concat(rows, ignore_index=True)
    if len(data) != 1482 or data.image_path.duplicated().any():
        raise RuntimeError('%s has wrong frame count or duplicate images' % variant)
    if not (data.reference_protocol == 'controlled_gt_jitter').all():
        raise RuntimeError('unexpected protocol for '+variant)
    data['error_recomputed_m'] = np.hypot(data.final_x-data.gt_x,data.final_y-data.gt_y)
    return data


def bar(ax, names, vals, title, ylabel, color, ylim=None, fmt='%.2f'):
    bars = ax.bar(range(len(names)),vals,color=color,width=.68)
    ax.set_xticks(range(len(names)),names,rotation=26,ha='right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y',alpha=.25)
    ax.set_axisbelow(True)
    if ylim: ax.set_ylim(*ylim)
    pad = (ax.get_ylim()[1]-ax.get_ylim()[0])*.018
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+pad,fmt%v,
                ha='center',va='bottom',fontsize=8)


def main():
    for name in ('data','tables'):
        (OUT/name).mkdir(parents=True,exist_ok=True)
    variants = {name:load_variant(name) for name in
                ('full','no_gru','no_kalman','no_ms','frames1','frames2',
                 'grid4','grid5','grid7','grid8')}
    full = variants['full']
    for name,data in variants.items():
        if set(data.image_path) != set(full.image_path):
            raise RuntimeError('unpaired frames in '+name)
    base = full[['city','route','frame_id','image_path','gt_x','gt_y',
                 'prior_center_x','prior_center_y','final_x','final_y','ms_latency_ms']].copy()
    base['reference_error_m'] = np.hypot(base.prior_center_x-base.gt_x,
                                          base.prior_center_y-base.gt_y)
    base['full_error_m'] = np.hypot(base.final_x-base.gt_x,base.final_y-base.gt_y)
    for variant in ('no_gru','no_ms'):
        frame=variants[variant].set_index('image_path')
        base[variant+'_error_m']=base.image_path.map(frame.error_recomputed_m)
        if base[variant+'_error_m'].isna().any():raise RuntimeError('missing '+variant)
    base.to_csv(OUT/'data/frame_errors_and_trajectory.csv',index=False)

    # Paper Table 2 UAV-view values, from Bearing-UAV CVPR 2026, not our replay.
    published = [
      ('University-1652',33.15,15.11),('SUES-200',30.83,15.76),
      ('DenseUAV',28.79,16.54),('GTA-UAV',28.43,27.96),
      ('Bearing-UAV',8.61,89.36),('Bearing-UAV + WA',7.48,92.88)]
    main_rows = [{'Method':n,'MLE_m':m,'LSR@15_pct':l,
                  'protocol':'Bearing-UAV CVPR 2026 Table 2 UAV-view',
                  'source':PAPER_URL} for n,m,l in published]
    main_rows.append({'Method':'Our Method','MLE_m':base.full_error_m.mean(),
                      'LSR@15_pct':metric(base.full_error_m,15),
                      'protocol':'v13 synthetic routes; per-frame GT-assisted reference/acquisition',
                      'source':str(SOURCE/'original_3759_all_metrics.json')})
    main_df=pd.DataFrame(main_rows)
    main_df.to_csv(OUT/'tables/main_comparison.csv',index=False)
    names=main_df.Method.tolist()
    colors=['#7b8794']*6+['#277da1']
    for field,title,ylabel,stem in [
        ('MLE_m','Localization error across reported protocols','MLE (m) ↓','main/main_mle'),
        ('LSR@15_pct','Localization success across reported protocols','LSR@15 (%) ↑','main/main_lsr15')]:
        fig,ax=plt.subplots(figsize=(11,5.5))
        bar(ax,names,main_df[field].tolist(),title,ylabel,colors,
            (0,max(main_df[field])*1.15),'%.2f')
        fig.text(.5,-.02,'Published methods: Bearing-UAV Table 2 UAV view. Our Method: GT-assisted local/path reference; protocols are not directly comparable.',
                 ha='center',fontsize=8,color='#9b3d3d')
        save(fig,stem)

    labels=[('no_gru','w/o GRU'),('no_kalman','w/o Kalman'),
            ('no_ms','w/o Final Refinement'),('full','Full Model')]
    ab=[]
    for key,label in labels:
        e=variants[key].error_recomputed_m
        ab.append({'Variant':label,'frames':len(e),'MLE_m':e.mean(),
                   'LSR@3_pct':metric(e,3),'LSR@5_pct':metric(e,5),
                   'LSR@15_pct':metric(e,15)})
    ab=pd.DataFrame(ab)
    ab.to_csv(OUT/'tables/component_ablation.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    bar(axes[0],ab.Variant,ab.MLE_m,'Component removal: error','MLE (m) ↓','#277da1',(0,8))
    bar(axes[1],ab.Variant,ab['LSR@3_pct'],'Component removal: success','LSR@3 (%) ↑','#e9a23b',(0,52))
    fig.suptitle('GT-assisted v13 replay · 1,482 frames',fontsize=12)
    fig.tight_layout();save(fig,'ablation/component_ablation')

    tm=[]
    for k,label,count in [('frames1','1 frame',1),('frames2','2 frames',2),('full','3 frames',3)]:
        e=variants[k].error_recomputed_m
        tm.append({'Input':label,'frame_count':count,'frames':len(e),
                   'MLE_m':e.mean(),'LSR@3_pct':metric(e,3),'LSR@5_pct':metric(e,5)})
    tm=pd.DataFrame(tm);tm.to_csv(OUT/'tables/temporal_window.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,col,ylabel,ylim in [(axes[0],'MLE_m','MLE (m) ↓',(3.65,4.02)),
                              (axes[1],'LSR@3_pct','LSR@3 (%) ↑',(37,44))]:
        ax.plot(tm.frame_count,tm[col],marker='o',markersize=8,color='#277da1',linewidth=2)
        ax.set_xticks([1,2,3],['1 frame','2 frames','3 frames'])
        ax.set_ylim(*ylim);ax.set_ylabel(ylabel);ax.grid(alpha=.25)
        for x,y in zip(tm.frame_count,tm[col]):ax.annotate('%.2f'%y,(x,y),xytext=(0,8),textcoords='offset points',ha='center')
    fig.suptitle('Explicit temporal input with recurrent state in every variant')
    fig.tight_layout();save(fig,'ablation/temporal_window')

    latency_summary=pd.read_csv(SOURCE/'paper_ablation_results.csv').set_index('variant')
    sizes=[]
    for key,grid in [('grid4',4),('grid5',5),('full',6),('grid7',7),('grid8',8)]:
        data=variants[key];e=data.error_recomputed_m
        sizes.append({'window':grid,'candidates':grid*grid,'frames':len(e),
                      'MLE_m':e.mean(),'LSR@3_pct':metric(e,3),
                      'mean_MS_time_ms':float(latency_summary.loc[key,'MS_Latency_ms']),
                      'frame_weighted_MS_time_ms':data.ms_latency_ms.mean()})
    sizes=pd.DataFrame(sizes);sizes.to_csv(OUT/'tables/final_neighborhood.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,col,ylabel,ylim in [(axes[0],'MLE_m','MLE (m) ↓',(3.65,4.25)),
                              (axes[1],'LSR@3_pct','LSR@3 (%) ↑',(34,45))]:
        ax.plot(sizes.window,sizes[col],marker='o',color='#277da1',linewidth=2)
        ax.scatter([6],[float(sizes.loc[sizes.window==6,col].iloc[0])],color='#e76f51',s=90,zorder=3)
        ax.set_xticks(sizes.window,[f'{v}×{v}' for v in sizes.window]);ax.set_ylabel(ylabel);ax.set_ylim(*ylim);ax.grid(alpha=.25)
    fig.suptitle('Final neighborhood accuracy · 6×6 is the Full configuration')
    fig.tight_layout();save(fig,'ablation/final_neighborhood_accuracy')
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(sizes.window,sizes.mean_MS_time_ms,marker='o',color='#935a96',linewidth=2)
    ax.scatter([6],[float(sizes.loc[sizes.window==6,'mean_MS_time_ms'].iloc[0])],color='#e76f51',s=90,zorder=3)
    ax.set_xticks(sizes.window,[f'{v}×{v}' for v in sizes.window]);ax.set_ylabel('Mean Shift time (ms) ↓')
    ax.set_title('Final neighborhood runtime');ax.grid(alpha=.25)
    for x,y in zip(sizes.window,sizes.mean_MS_time_ms):ax.annotate('%.2f'%y,(x,y),xytext=(0,8),textcoords='offset points',ha='center')
    fig.tight_layout();save(fig,'ablation/final_neighborhood_runtime')

    city_rows=[];route_rows=[]
    for city in CITIES:
        sub=base[base.city==city]
        city_rows.append({'City':city.title(),'frames':len(sub),
            'Reference_MLE_m':sub.reference_error_m.mean(),
            'Reference_LSR@3_pct':metric(sub.reference_error_m,3),
            'Reference_LSR@5_pct':metric(sub.reference_error_m,5),
            'Full_MLE_m':sub.full_error_m.mean(),
            'Full_P90_m':np.percentile(sub.full_error_m,90),
            'Full_LSR@3_pct':metric(sub.full_error_m,3),
            'Full_LSR@5_pct':metric(sub.full_error_m,5),
            'Full_LSR@10_pct':metric(sub.full_error_m,10),
            'Full_LSR@15_pct':metric(sub.full_error_m,15)})
        for route in ('route_B','route_C'):
            r=sub[sub.route==route]
            route_rows.append({'City':city.title(),'Route':route,'frames':len(r),
                'Reference_MLE_m':r.reference_error_m.mean(),
                'Reference_LSR@5_pct':metric(r.reference_error_m,5),
                'Full_MLE_m':r.full_error_m.mean(),
                'Full_LSR@5_pct':metric(r.full_error_m,5)})
        fig,axes=plt.subplots(1,2,figsize=(11,4.8))
        for ax,route in zip(axes,('route_B','route_C')):
            r=sub[sub.route==route].sort_values('frame_id')
            ax.plot(r.prior_center_x,r.prior_center_y,color='#8d99ae',alpha=.65,linewidth=1.2,label='GT-assisted reference center')
            ax.plot(r.gt_x,r.gt_y,color='#2a9d8f',linewidth=2,label='GT')
            ax.plot(r.final_x,r.final_y,color='#277da1',linewidth=1.5,label='Full prediction')
            ax.scatter(r.gt_x.iloc[0],r.gt_y.iloc[0],c='#2a9d8f',marker='o',s=45,zorder=4)
            ax.scatter(r.gt_x.iloc[-1],r.gt_y.iloc[-1],c='#2a9d8f',marker='X',s=50,zorder=4)
            ax.set_title('%s · %s · %d frames'%(city.title(),route,len(r)))
            ax.set_xlabel('Local x (m)');ax.set_ylabel('Local y (m)')
            ax.axis('equal');ax.grid(alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,1.02))
        fig.tight_layout(rect=(0,0,1,.94));save(fig,'trajectory/'+city+'_routes')
    city_df=pd.DataFrame(city_rows);city_df.to_csv(OUT/'tables/per_city.csv',index=False)
    pd.DataFrame(route_rows).to_csv(OUT/'tables/per_route_reference_improvement.csv',index=False)
    for col,ylabel,stem,limit in [
        ('Full_MLE_m','MLE (m) ↓','per_city_mle',(0,5)),
        ('Full_LSR@5_pct','LSR@5 (%) ↑','per_city_lsr5',(0,100)),
        ('Full_P90_m','P90 error (m) ↓','per_city_p90',(0,9))]:
        fig,ax=plt.subplots(figsize=(6.8,4))
        bar(ax,city_df.City,city_df[col],stem.replace('_',' ').title(),ylabel,
            ['#669bbc','#277da1','#e76f51','#7a9e7e'],limit)
        fig.tight_layout();save(fig,'city/'+stem)
    for ref_col,model_col,ylabel,stem,ylim in [
        ('Reference_MLE_m','Full_MLE_m','MLE (m) ↓','per_city_baseline_mle',(0,7)),
        ('Reference_LSR@5_pct','Full_LSR@5_pct','LSR@5 (%) ↑','per_city_baseline_lsr5',(0,100))]:
        fig,ax=plt.subplots(figsize=(7,4))
        x=np.arange(4);width=.34
        ax.bar(x-width/2,city_df[ref_col],width,color=COLORS['Reference baseline'],label='Reference baseline')
        ax.bar(x+width/2,city_df[model_col],width,color=COLORS['Full Model'],label='Full model')
        ax.set_xticks(x,city_df.City);ax.set_ylabel(ylabel);ax.set_ylim(*ylim)
        ax.set_title('GT-assisted reference vs Full · per city');ax.legend();ax.grid(axis='y',alpha=.2)
        fig.tight_layout();save(fig,'baseline/'+stem)

    threshold=np.linspace(0,20,401)
    curves={'Reference baseline':base.reference_error_m,
            'w/o GRU':base.no_gru_error_m,
            'w/o Final Refinement':base.no_ms_error_m,
            'Full Model':base.full_error_m}
    cdf=pd.DataFrame({'threshold_m':threshold})
    fig,ax=plt.subplots(figsize=(7.7,4.8))
    for name,values in curves.items():
        y=np.array([metric(values,t) for t in threshold])
        cdf[name]=y
        ax.plot(threshold,y,label=name,color=COLORS[name],linewidth=2)
    cdf.to_csv(OUT/'data/error_cdf.csv',index=False)
    ax.set(xlim=(0,20),ylim=(0,103),xlabel='Localization error threshold (m)',
           ylabel='Cumulative success rate (%)',title='Paired error CDF · 1,482 GT-assisted frames')
    ax.grid(alpha=.2);ax.legend(loc='lower right')
    fig.tight_layout();save(fig,'baseline/error_cdf')

    checks={'full_mle':base.full_error_m.mean(),'full_lsr15':metric(base.full_error_m,15),
            'reference_mle':base.reference_error_m.mean(),'frame_count':len(base)}
    if not (abs(checks['full_mle']-3.7591276968535423)<1e-6 and
            abs(checks['reference_mle']-5.372479122924439)<1e-6 and
            checks['full_lsr15']==100.0 and len(base)==1482):
        raise RuntimeError('source replay consistency check failed: '+repr(checks))
    (OUT/'data/recomputation_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (OUT/'data/source_manifest.json').write_text(json.dumps({
        'original_replay':str(SOURCE),'original_protocol_file':str(SOURCE/'PROTOCOL.txt'),
        'primary_published_benchmark_paper':PAPER_URL,
        'published_values':{r['Method']:{'MLE_m':r['MLE_m'],'LSR@15_pct':r['LSR@15_pct']} for r in main_rows[:-1]},
        'all_our_metric_sources':'original replay per-frame CSV; recomputed here',
        'n_frames':1482,'n_routes':8,'cities':CITIES,
        'warning':'Our Method uses current-frame GT-assisted local/path reference; published Table 2 UAV-view methods use different reference protocol.'
    },indent=2)+'\n')
    print('Generated',len(list((OUT/'figures').rglob('*.pdf'))),'figures and',len(list((OUT/'tables').glob('*.csv'))),'tables; checks',checks)


if __name__=='__main__':main()
