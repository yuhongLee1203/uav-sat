#!/usr/bin/env python3
"""Audit and render thesis results from the immutable v39 3.759 m replay.
No training, checkpoint writes, prediction smoothing, or result tuning.
"""
from __future__ import annotations
import csv, json, math, platform, hashlib, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image

ROOT=Path('/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368')
OUT=Path('/yh/study/uav-sat/report/add_more/thesis_experiments')
T=OUT/'tables'; F=OUT/'figures'; R=OUT/'raw'
CITIES='abcd'; ROUTES=('test_01','test_02'); VARIANTS=('full','decoder_probe','no_gru','no_kalman','no_ms','frames1','frames2','grid4','grid5','grid7','grid8')
for d in (T,F,R): d.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8,'savefig.dpi':320,'font.family':'DejaVu Sans'})
COLORS={'FieldAnchor':'#bd392f','Ground Truth':'#11834e','Supplied Reference':'#555f6e','w/o GRU':'#2477a9','w/o Final Refinement':'#a56b2a','Argmax':'#8047a5','Weighted Average':'#1d83a6','Mean Shift':'#b73136'}

def readcsv(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def writecsv(p,cols,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def f2(x):return f'{float(x):.2f}'
def xy(rows,p):return np.asarray([[float(r[p+'_x']),float(r[p+'_y'])] for r in rows])
def error(p,g):return np.linalg.norm(p-g,axis=-1)
def metrics(e):
 e=np.asarray(e,dtype=float);assert len(e)>0 and np.isfinite(e).all()
 return {'N':len(e),'MLE (m)':e.mean(),'Median Error (m)':np.median(e),'LSR@3 (%)':100*np.mean(e<=3),'LSR@15 (%)':100*np.mean(e<=15)}
def row_metrics(e):
 m=metrics(e);return {k:(int(v) if k=='N' else f2(v)) for k,v in m.items()}
def savefig(fig,stem,svg=False):
 fig.savefig(F/(stem+'.png'),dpi=320,bbox_inches='tight')
 fig.savefig(F/(stem+'.pdf'),bbox_inches='tight')
 if svg:fig.savefig(F/(stem+'.svg'),bbox_inches='tight')
 plt.close(fig)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load():
 data={}; source={}
 for c in CITIES:
  for v in VARIANTS:
   folder=ROOT/('city'+c)/'variants'/v
   if not folder.exists():continue
   summary=json.loads((folder/'bearing_v39_summary.json').read_text())
   for route in ROUTES:
    p=folder/Path(summary[route]['CSV']).name
    rows=readcsv(p);data[(c,route,v)]=rows;source[(c,route,v)]=str(p)
 for c in CITIES:
  for route in ROUTES:
   base=data[(c,route,'full')]
   for v in VARIANTS:
    rows=data[(c,route,v)]
    assert len(rows)==len(base),(c,route,v)
    for a,b in zip(base,rows):
     assert a['frame_id']==b['frame_id'] and a['image_path']==b['image_path']
     assert abs(float(a['gt_x'])-float(b['gt_x']))<1e-6 and abs(float(a['gt_y'])-float(b['gt_y']))<1e-6
 return data,source

def route_points(c,route):
 prep=ROOT/('city'+c)/'prepared'
 way=json.loads((prep/'routes'/route/'waypoints.json').read_text())
 train=readcsv(prep/'routes/train_01/manifest.csv')
 origin=np.array([float(train[0]['x_m']),float(train[0]['y_m'])])
 mpp=float(way['mpp'])
 return np.asarray([[float(w['pixel_x'])*mpp,float(w['pixel_y'])*mpp] for w in sorted(way['waypoints'],key=lambda x:int(x['waypoint_order']))])-origin

def xy_from_se(points,s,e,radius=36.0):
 delta=np.diff(points,axis=0);length=np.linalg.norm(delta,axis=1);unit=delta/length[:,None]
 cum=np.r_[0,np.cumsum(length)];s=float(np.clip(s,0,cum[-1]));leg=int(np.clip(np.searchsorted(cum,s,side='right')-1,0,len(length)-1))
 along=float(np.clip(s-cum[leg],0,length[leg]));center=points[leg]+along*unit[leg]
 cross=np.array([-unit[leg,1],unit[leg,0]])
 if leg>0 and s-cum[leg]<radius:
  alpha=float(np.clip((s-cum[leg])/radius,0,1));alpha=alpha*alpha*(3-2*alpha)
  blended=(1-alpha)*unit[leg-1]+alpha*unit[leg]
  n=np.linalg.norm(blended)
  if n>1e-9:
   blended=blended/n;cross=np.array([-blended[1],blended[0]])
 return center+e*cross

def stage_analysis(data):
 names=['Supplied Local Reference','Front Mean Shift','Corrected Visual Measurement','Kalman Posterior','Final Mean Shift']
 places={k:[] for k in names};gt_all=[];route_id=[];frames=[]
 for c in CITIES:
  for route in ROUTES:
   rows=data[(c,route,'full')];gt=xy(rows,'gt');points=route_points(c,route)
   positions={names[0]:xy(rows,'prior_center'),names[1]:xy(rows,'softms'),names[2]:np.asarray([xy_from_se(points,float(r['measurement_s']),float(r['measurement_e'])) for r in rows]),names[3]:xy(rows,'kalman'),names[4]:xy(rows,'final')}
   assert np.max(np.abs(positions[names[1]]-xy(rows,'visual_anchor')))<1e-4
   for name in names:places[name].append(positions[name])
   gt_all.append(gt);route_id.extend([c+'/'+route]*len(rows));frames.extend([int(r['frame_id']) for r in rows])
 gt=np.vstack(gt_all);out={'gt_xy':gt,'route_id':np.asarray(route_id),'frame_id':np.asarray(frames,dtype=int)}
 tab=[]
 for name in names:
  arr=np.vstack(places[name]);out[name.lower().replace(' ','_')+'_xy']=arr
  tab.append({'Estimation Stage':name,**{k:v for k,v in row_metrics(error(arr,gt)).items() if k!='N'}})
 np.savez_compressed(R/'stage_wise_predictions.npz',**out)
 writecsv(T/'stage_wise_results.csv',list(tab[0]),tab)
 return out,tab

def route_and_overall(data):
 tab=[];errors=[];route_n=[]
 for c in CITIES:
  for route in ROUTES:
   rows=data[(c,route,'full')];gt=xy(rows,'gt');pred=xy(rows,'final');e=error(pred,gt)
   assert np.max(np.abs(e-np.asarray([float(r['error_final_m']) for r in rows])))<1e-4
   m=row_metrics(e);tab.append({'City':c.upper(),'Route':route,'Samples':len(rows),'MLE (m)':m['MLE (m)'],'Median Error (m)':m['Median Error (m)'],'LSR@3 (%)':m['LSR@3 (%)'],'LSR@15 (%)':m['LSR@15 (%)']})
   errors.extend(e);route_n.append(len(rows))
 assert sum(route_n)==1482
 writecsv(T/'route_wise_results.csv',list(tab[0]),tab)
 (T/'route_wise_results.json').write_text(json.dumps(tab,indent=2)+'\n')
 m=metrics(errors)
 assert abs(m['MLE (m)']-3.7591276968535423)<1e-6 and round(m['LSR@3 (%)'],2)==41.90 and m['LSR@15 (%)']==100.0
 (OUT/'overall_results.json').write_text(json.dumps({'protocol':'GT-assisted v39 Full replay','samples':1482,'routes':8,'MLE_m':float(m['MLE (m)']),'Median_Error_m':float(m['Median Error (m)']),'LSR_at_3_pct':float(m['LSR@3 (%)']),'LSR_at_15_pct':float(m['LSR@15 (%)'])},indent=2)+'\n')
 return tab,m

def decoder(data):
 keys=[];gt=[];kal=[];centers=[];scores=[];weights=[];arg=[];wa=[];ms=[];route_ids=[]
 tau=.30
 for c in CITIES:
  for route in ROUTES:
   probe=data[(c,route,'decoder_probe')];base=data[(c,route,'full')]
   for r,b in zip(probe,base):
    C=np.asarray(json.loads(r['decoder_candidate_centers_json']),dtype=float);S=np.asarray(json.loads(r['decoder_candidate_logits_json']),dtype=float)
    assert C.shape==(36,2) and S.shape==(36,)
    W=np.exp((S-S.max())/tau);W=W/W.sum();A=C[int(np.argmax(S))];B=(W[:,None]*C).sum(axis=0);M=np.array([float(r['final_x']),float(r['final_y'])]);
    assert np.linalg.norm(B-np.array([float(r['decoder_weighted_x']),float(r['decoder_weighted_y'])]))<2e-4
    assert np.linalg.norm(M-np.array([float(b['final_x']),float(b['final_y'])]))<1e-3
    keys.append(f'{c}/{route}/{r["frame_id"]}');route_ids.append(f'{c}/{route}');gt.append([float(r['gt_x']),float(r['gt_y'])]);kal.append([float(r['kalman_x']),float(r['kalman_y'])]);centers.append(C);scores.append(S);weights.append(W);arg.append(A);wa.append(B);ms.append(M)
 arrays={'sample_key':np.asarray(keys),'route_id':np.asarray(route_ids),'gt_xy':np.asarray(gt),'kalman_xy':np.asarray(kal),'candidate_centers_xy':np.asarray(centers),'candidate_regularized_logits':np.asarray(scores),'candidate_weights':np.asarray(weights),'argmax_xy':np.asarray(arg),'weighted_average_xy':np.asarray(wa),'mean_shift_xy':np.asarray(ms)}
 assert len(keys)==1482
 np.savez_compressed(R/'final_decoder_predictions.npz',**arrays)
 tab=[]
 for name,pos in [('Argmax','argmax_xy'),('Weighted Average','weighted_average_xy'),('Mean Shift','mean_shift_xy')]:
  m=row_metrics(error(arrays[pos],arrays['gt_xy']));tab.append({'Final Decoder':name,**{k:v for k,v in m.items() if k!='N'}})
 writecsv(T/'final_decoder_comparison.csv',list(tab[0]),tab)
 assert abs(float(tab[-1]['MLE (m)'])-3.76)<.011
 return arrays,tab

def reference_improvement(data):
 tab=[]
 for c in CITIES:
  for route in ROUTES:
   rows=data[(c,route,'full')];g=xy(rows,'gt');ref=error(xy(rows,'prior_center'),g).mean();full=error(xy(rows,'final'),g).mean()
   tab.append({'City':c.upper(),'Route':route,'Reference MLE (m)':f2(ref),'FieldAnchor MLE (m)':f2(full),'Improvement (m)':f2(ref-full),'Improvement (%)':f2(100*(ref-full)/ref)})
 writecsv(T/'route_reference_improvement.csv',list(tab[0]),tab)
 x=np.arange(8);fig,ax=plt.subplots(figsize=(10,4.4),layout='constrained')
 ax.bar(x-.18,[float(r['Reference MLE (m)']) for r in tab],.36,label='Supplied Reference',color=COLORS['Supplied Reference'])
 ax.bar(x+.18,[float(r['FieldAnchor MLE (m)']) for r in tab],.36,label='FieldAnchor',color=COLORS['FieldAnchor'])
 ax.set_xticks(x,[f"{r['City']} {r['Route'][-2:]}" for r in tab]);ax.set_ylabel('Mean Localization Error (m)');ax.set_xlabel('Held-out Route');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 savefig(fig,'route_reference_improvement')
 return tab

def distribution(data):
 methods=[('Supplied Reference','full','prior_center'),('FieldAnchor','full','final'),('w/o GRU','no_gru','final'),('w/o Final Refinement','no_ms','final')]
 tab=[];curves={}
 for name,v,p in methods:
  vals=[]
  for c in CITIES:
   for route in ROUTES:
    rows=data[(c,route,v)];vals.extend(error(xy(rows,p),xy(rows,'gt')))
  e=np.asarray(vals);curves[name]=e
  tab.append({'Method':name,'Mean (m)':f2(e.mean()),'Median (m)':f2(np.median(e)),'P75 (m)':f2(np.quantile(e,.75)),'P90 (m)':f2(np.quantile(e,.90)),'P95 (m)':f2(np.quantile(e,.95)),'Max (m)':f2(e.max())})
 writecsv(T/'localization_error_distribution.csv',list(tab[0]),tab)
 fig,ax=plt.subplots(figsize=(7.6,5),layout='constrained')
 for name,e in curves.items():
  s=np.sort(e);ax.step(np.r_[0,s],np.r_[0,np.arange(1,len(s)+1)/len(s)],where='post',label=name,color=COLORS[name],lw=1.8)
 ax.set_xlim(0,max(v.max() for v in curves.values())*1.02);ax.set_ylim(0,1.01);ax.set_xlabel('Localization Error (m)');ax.set_ylabel('Cumulative Fraction');ax.legend(loc='lower right',frameon=True);ax.grid(alpha=.2)
 savefig(fig,'localization_error_cdf')
 return tab

def trajectories(data,route_table):
 """Plot the full satellite map and straight waypoint legs for every route."""
 maps={};audit={'background':'complete city satellite image; no route crop',
                 'gt_line':'saved route start/end and waypoints with an absolute direction change of at least 30 degrees, joined by straight segments; display only',
                 'prediction_line':'raw per-frame final_x/final_y in frame order',
                 'metrics_gt':'raw per-frame gt_x/gt_y, not the nominal waypoint line',
                 'routes':{}}
 for c in CITIES:
  prep=ROOT/('city'+c)/'prepared'
  experiment=json.loads((prep/'experiment.json').read_text())
  satellite=Path(experiment['satellite_image']);mpp=float(experiment['mpp'])
  train=readcsv(prep/'routes/train_01/manifest.csv');ox=float(train[0]['x_m']);oy=float(train[0]['y_m'])
  with Image.open(satellite) as image:
   source_size=image.size
   image.thumbnail((2048,2048),Image.Resampling.LANCZOS)
   rendered=np.asarray(image.convert('RGB'))
  extent=(0*mpp-ox,source_size[0]*mpp-ox,source_size[1]*mpp-oy,0*mpp-oy)
  maps[c]=(rendered,extent,str(satellite),source_size)
 assert len({v[3] for v in maps.values()})==1
 assert len({round(v[1][1]-v[1][0],6) for v in maps.values()})==1
 def panel(ax,c,route):
  rows=data[(c,route,'full')]
  gt=xy(rows,'gt');pred=xy(rows,'final');waypoint=route_points(c,route)
  legs=np.diff(waypoint,axis=0)
  turns=np.degrees(np.arctan2(legs[:-1,0]*legs[1:,1]-legs[:-1,1]*legs[1:,0],(legs[:-1]*legs[1:]).sum(axis=1)))
  keep=[0]+[i+1 for i,turn in enumerate(turns) if abs(turn)>=30.0]+[len(waypoint)-1]
  display_route=waypoint[keep]
  m=metrics(error(pred,gt));background,extent,path,source_size=maps[c]
  pts=np.vstack([gt,pred,waypoint]);inside=bool(np.all((pts[:,0]>=extent[0])&(pts[:,0]<=extent[1])&(pts[:,1]>=extent[3])&(pts[:,1]<=extent[2])))
  if not inside:raise RuntimeError(f'{c}/{route}: positions outside full satellite image')
  ax.imshow(background,extent=extent,origin='upper',interpolation='nearest',zorder=0)
  ground,=ax.plot(display_route[:,0],display_route[:,1],color=COLORS['Ground Truth'],lw=2.0,
                  label='Ground Truth Route (major turns)',zorder=3,solid_joinstyle='miter')
  ground.set_path_effects([pe.Stroke(linewidth=4.2,foreground='white',alpha=.9),pe.Normal()])
  predicted,=ax.plot(pred[:,0],pred[:,1],color=COLORS['FieldAnchor'],lw=1.4,
                     label='FieldAnchor',zorder=4)
  predicted.set_path_effects([pe.Stroke(linewidth=3.0,foreground='white',alpha=.65),pe.Normal()])
  ax.set_xlim(extent[0],extent[1]);ax.set_ylim(extent[2],extent[3])
  ax.set_aspect('equal',adjustable='box');ax.set_xlabel('Map X (m)');ax.set_ylabel('Map Y (m)')
  audit['routes'][f'{c}/{route}']={'satellite_image':path,'source_size_px':list(source_size),
   'rendered_size_px':list(background.shape[1::-1]),'map_extent_m':list(extent),
   'waypoint_count':len(waypoint),'display_waypoint_indices':keep,'display_waypoint_count':len(keep),
   'display_turn_threshold_deg':30.0,'prediction_count':len(pred),'all_positions_inside':inside}
  return m
 fig,axes=plt.subplots(2,2,figsize=(11,11),layout='constrained')
 for ax,c in zip(axes.flat,CITIES):
  panel(ax,c,'test_01');ax.set_title(f'({c}) City {c.upper()} · test_01')
 handles,labels=axes.flat[0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.005),frameon=False)
 savefig(fig,'representative_trajectories',svg=True)
 fig,axes=plt.subplots(4,2,figsize=(11,22),layout='constrained')
 for i,c in enumerate(CITIES):
  for j,route in enumerate(ROUTES):
   m=panel(axes[i,j],c,route)
   axes[i,j].set_title(f'City {c.upper()} · {route} · MLE {m["MLE (m)"]:.2f} m')
 handles,labels=axes.flat[0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.005),frameon=False)
 savefig(fig,'all_eight_routes_appendix')
 (R/'trajectory_map_audit.json').write_text(json.dumps(audit,indent=2)+'\n')

def route_progress(data):
 methods=[('FieldAnchor','full'),('w/o GRU','no_gru'),('w/o Final Refinement','no_ms')]
 fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 for ax,c in zip(axes.flat,CITIES):
  for name,v in methods:
   rows=data[(c,'test_01',v)];e=error(xy(rows,'final'),xy(rows,'gt'));ax.plot(np.arange(len(e)),e,lw=1.2,label=name,color=COLORS[name])
  ax.set_title(f'City {c.upper()} · test_01');ax.set_xlabel('Observation Index');ax.set_ylabel('Localization Error (m)');ax.grid(alpha=.2)
 handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.02),frameon=False)
 savefig(fig,'localization_error_over_sequence')
 bins=np.linspace(0,100,101);curves={}
 for name,v in methods[:2]:
  all_interp=[]
  for c in CITIES:
   for route in ROUTES:
    rows=data[(c,route,v)];e=error(xy(rows,'final'),xy(rows,'gt'))
    all_interp.append(np.interp(bins,np.linspace(0,100,len(e)),e))
  a=np.asarray(all_interp);curves[name]={'median':np.median(a,axis=0),'p25':np.quantile(a,.25,axis=0),'p75':np.quantile(a,.75,axis=0)}
 csvrows=[]
 for name,d in curves.items():
  for i,progress in enumerate(bins):csvrows.append({'Method':name,'Route Progress (%)':int(progress),'Median Error (m)':f2(d['median'][i]),'P25 (m)':f2(d['p25'][i]),'P75 (m)':f2(d['p75'][i])})
 writecsv(R/'error_vs_route_progress.csv',list(csvrows[0]),csvrows)
 fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
 for name,d in curves.items():
  ax.plot(bins,d['median'],color=COLORS[name],lw=2,label=name)
  ax.fill_between(bins,d['p25'],d['p75'],color=COLORS[name],alpha=.15)
 ax.set_xlim(0,100);ax.set_xlabel('Normalized Route Progress (%)');ax.set_ylabel('Localization Error (m)');ax.legend(frameon=False);ax.grid(alpha=.2)
 savefig(fig,'error_vs_route_progress')

def response_figure(a):
 S=a['candidate_regularized_logits'];W=a['candidate_weights'];C=a['candidate_centers_xy'];K=a['kalman_xy'];G=a['gt_xy'];A=a['argmax_xy'];B=a['weighted_average_xy'];M=a['mean_shift_xy']
 peaks=[]
 for i,w in enumerate(W):
  grid=w.reshape(6,6);local=[]
  for y in range(6):
   for x in range(6):
    neighbors=[grid[yy,xx] for yy in range(max(0,y-1),min(6,y+2)) for xx in range(max(0,x-1),min(6,x+2)) if (yy,xx)!=(y,x)]
    if grid[y,x]>max(neighbors):local.append((float(grid[y,x]),y,x))
  local.sort(reverse=True);peaks.append(local)
 # Categories are chosen algorithmically from the untouched responses. A local
 # peak in the 6x6 lattice is not a distinct converged Mean Shift mode.
 single=max((i for i,p in enumerate(peaks) if len(p)==1),key=lambda i:float(W[i].max()),default=int(np.argmax(W.max(axis=1))))
 multiple=max((i for i,p in enumerate(peaks) if len(p)>=2),key=lambda i:peaks[i][1][0]/peaks[i][0][0],default=None)
 chosen=[single,multiple,int(np.argmax(np.linalg.norm(A-M,axis=1))),int(np.argmax(np.linalg.norm(B-M,axis=1)))]
 if multiple is None:chosen[1]=int(np.argmax(np.partition(W,-2,axis=1)[:,-2]/np.max(W,axis=1)))
 # Retain distinct samples when possible without changing their category score.
 for j in range(1,4):
  if chosen[j] in chosen[:j]:
   if j==1:ranking=sorted(range(len(W)),key=lambda i:peaks[i][1][0]/peaks[i][0][0] if len(peaks[i])>=2 else -1,reverse=True)
   elif j==2:ranking=np.argsort(-np.linalg.norm(A-M,axis=1))
   else:ranking=np.argsort(-np.linalg.norm(B-M,axis=1))
   chosen[j]=next((int(i) for i in ranking if int(i) not in chosen[:j]),chosen[j])
 labels=['Strong single lattice peak','Multiple local lattice peaks' if multiple is not None else 'Second-strongest response','Largest Argmax/Mean Shift gap','Largest Weighted Average/Mean Shift gap']
 keys=['sample_key','gt_xy','kalman_xy','candidate_centers_xy','candidate_regularized_logits','candidate_weights','argmax_xy','weighted_average_xy','mean_shift_xy']
 np.savez_compressed(R/'final_response_visualization_samples.npz',**{k:a[k][chosen] for k in keys},selection_labels=np.asarray(labels),lattice_peak_counts=np.asarray([len(peaks[i]) for i in chosen]))
 # Equal 36 m x 36 m spatial window about each Kalman posterior.
 # Verified to contain every one of the selected 36 candidate centers.
 radius=18.0
 assert np.max(np.abs(C[chosen]-K[chosen,None,:]))<radius
 fig,axes=plt.subplots(2,2,figsize=(10.5,10.5))
 fig.subplots_adjust(left=.09,right=.98,top=.96,bottom=.16,hspace=.48,wspace=.29)
 for ax,idx,label in zip(axes.flat,chosen,labels):
  center=K[idx];rel=C[idx]-center
  ax.scatter(rel[:,0],rel[:,1],c=W[idx],s=35+700*W[idx],cmap='viridis',vmin=0,vmax=float(W[chosen].max()),edgecolor='black',linewidth=.35,zorder=2,label='Candidate Response')
  for point,name,marker,color,size in [(G[idx],'Ground Truth','*','#198b52',160),(K[idx],'Kalman Posterior','s','#333333',70),(A[idx],'Argmax','^',COLORS['Argmax'],75),(B[idx],'Weighted Average','D',COLORS['Weighted Average'],65),(M[idx],'Mean Shift','o',COLORS['Mean Shift'],70)]:
   v=point-center;ax.scatter([v[0]],[v[1]],marker=marker,c=color,s=size,edgecolors='white',linewidths=.8,zorder=4,label=name)
  ax.set_xlim(-radius,radius);ax.set_ylim(-radius,radius);ax.set_aspect('equal');ax.set_title(f'{label}\n{a["sample_key"][idx]} · {len(peaks[idx])} lattice peaks');ax.set_xlabel('Δ Map X from Kalman (m)');ax.set_ylabel('Δ Map Y from Kalman (m)');ax.grid(alpha=.16)
 handles,leglabels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,leglabels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.025),frameon=False)
 savefig(fig,'final_response_visualization')
 return [{'category':label,'sample':str(a['sample_key'][i]),'lattice_peaks':len(peaks[i]),'argmax_ms_gap_m':float(np.linalg.norm(A[i]-M[i])),'weighted_ms_gap_m':float(np.linalg.norm(B[i]-M[i]))} for label,i in zip(labels,chosen)]

def split_stats(data):
 tab=[]
 for c in CITIES:
  prep=ROOT/('city'+c)/'prepared';n=len(readcsv(prep/'routes/train_01/manifest.csv'))
  # Exact v39 robust_tracker.split_ranges: 70%/15%, 16-frame guard,
  # with one validation observation minimum on these short train routes.
  train_end=int(n*.70);val_end=int(n*.85);guard=16
  train_count=max(1,train_end-guard)
  val_start=min(n-1,train_end+guard)
  val_stop=max(min(n,val_end-guard),val_start+1)
  val_count=val_stop-val_start
  t1=len(data[(c,'test_01','full')]);t2=len(data[(c,'test_02','full')])
  tab.append({'City':c.upper(),'Train Sequence':'train_01','Train Samples':train_count,'Validation Samples':val_count,'Test_01 Samples':t1,'Test_02 Samples':t2,'Train Route Total':n,'Unused Guard/Tail Samples':n-train_count-val_count})
 assert sum(x['Test_01 Samples']+x['Test_02 Samples'] for x in tab)==1482
 writecsv(T/'dataset_split_statistics.csv',list(tab[0]),tab)
 return tab

def ablation(data):
 def all_err(v):
  return np.concatenate([error(xy(data[(c,route,v)],'final'),xy(data[(c,route,v)],'gt')) for c in CITIES for route in ROUTES])
 comp=[]
 for name,v,g,k,r in [('w/o GRU','no_gru','No','Yes','Yes'),('w/o Kalman','no_kalman','Yes','No','Yes'),('w/o Final Refinement','no_ms','Yes','Yes','No'),('Full','full','Yes','Yes','Yes')]:
  m=metrics(all_err(v));comp.append({'Variant':name,'GRU':g,'Kalman':k,'Final Refinement':r,'LSR@3 (%)':f2(m['LSR@3 (%)']),'LSR@15 (%)':f2(m['LSR@15 (%)']),'MLE (m)':f2(m['MLE (m)'])})
 writecsv(T/'component_ablation.csv',list(comp[0]),comp)
 temp=[]
 for n,v in [(1,'frames1'),(2,'frames2'),(3,'full')]:
  m=metrics(all_err(v));temp.append({'Input Length (frames)':n,'LSR@3 (%)':f2(m['LSR@3 (%)']),'LSR@15 (%)':f2(m['LSR@15 (%)']),'MLE (m)':f2(m['MLE (m)'])})
 writecsv(T/'temporal_context_ablation.csv',list(temp[0]),temp)
 sizes=[]
 for n,v in [(4,'grid4'),(5,'grid5'),(6,'full'),(7,'grid7'),(8,'grid8')]:
  m=metrics(all_err(v));times=[]
  for c in CITIES:
   for route in ROUTES:times.extend(float(r['ms_latency_ms']) for r in data[(c,route,v)])
  # Historical paper latency used a separate benchmark; report this replay's
  # per-frame measured mean, without copying 4.04/5.00/7.52/... by hand.
  sizes.append({'Window':f'{n}x{n}','Candidates':n*n,'LSR@3 (%)':f2(m['LSR@3 (%)']),'LSR@15 (%)':f2(m['LSR@15 (%)']),'MLE (m)':f2(m['MLE (m)']),'Measured Mean Shift Time (ms)':f2(np.mean(times))})
 writecsv(T/'final_neighborhood_ablation.csv',list(sizes[0]),sizes)
 assert comp[-1]['MLE (m)']==sizes[2]['MLE (m)'] and comp[-1]['LSR@3 (%)']==sizes[2]['LSR@3 (%)']
 return comp,temp,sizes

def runtime(data):
 import torch
 from torchvision import models
 times=[];raw=[]
 for c in CITIES:
  for route in ROUTES:
   rows=data[(c,route,'full')]
   assert len(rows)>50
   for i,r in enumerate(rows):
    latency=float(r['end_to_end_latency_ms'])
    assert math.isfinite(latency) and latency>0
    if i>=50:
     raw.append({'City':c.upper(),'Route':route,'Frame ID':r['frame_id'],'Latency (ms)':f'{latency:.6f}'})
     times.append(latency)
 writecsv(R/'runtime_raw_ms.csv',list(raw[0]),raw)
 assert len(times)==1482-50*8
 t=np.asarray(times)
 ckroot=ROOT/'citya/train_frames3/checkpoints'
 learned=torch.load(ckroot/'controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt',map_location='cpu',weights_only=False)['model']
 visual=torch.load(ckroot/'visual_retrieval_A_only.pt',map_location='cpu',weights_only=False)['model']
 temporal_n=sum(x.numel() for x in learned.values() if torch.is_tensor(x));visual_n=sum(x.numel() for x in visual.values() if torch.is_tensor(x))
 backbone=models.mobilenet_v3_small(weights=None);backbone.classifier=torch.nn.Identity()
 backbone_n=sum(p.numel() for p in backbone.parameters())
 trainable=temporal_n+visual_n;total=trainable+backbone_n
 result=[('Trainable Parameters',str(trainable)),('Total Parameters',str(total)),('Mean End-to-End Latency (ms)',f2(t.mean())),('Median End-to-End Latency (ms)',f2(np.median(t))),('Latency Standard Deviation (ms)',f2(t.std(ddof=0))),('FPS',f2(1000/t.mean())),('GPU','NVIDIA GeForce RTX 3090 (original replay distributed across GPU 0/5/6)'),('Precision','FP32 compute; some precomputed feature caches stored as FP16'),('Batch Size','1'),('UAV Input Resolution','256x256'),('Timing Observations',str(len(t))),('Warm-up Observations per Route','50')]
 writecsv(T/'runtime_results.csv',['Metric','Value'],[{'Metric':k,'Value':v} for k,v in result])
 meta=[f'Original replay root: {ROOT}',f'CPU: {next((x.split(":",1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("model name")),platform.processor())}',f'GPU: NVIDIA GeForce RTX 3090; city evaluations ran on GPU 0/5/6',f'Current PyTorch used to analyze checkpoints: {torch.__version__}',f'Current PyTorch CUDA build: {torch.version.cuda}',f'Original timing source: full variant per-frame end_to_end_latency_ms fields',f'Original timing implementation: full/runtime/robust_tracker.py run_route_inference',f'Timing includes 256x256 UAV backbone feature extraction, cross-view scoring, front Mean Shift, GRU, Kalman, final candidate scoring, and final Mean Shift.',f'Timing excludes image disk I/O, tensor preprocessing, model loading, satellite gallery construction.',f'CUDA synchronization occurs before timer start and after final prediction. First 50 observations from each route excluded in analysis; no new inference was run.',f'Original replay timing 8 routes x (route length minus 50) = {len(t)} measured observations.',f'Architecture parameter breakdown: frozen MobileNetV3-small={backbone_n}, visual heads={visual_n}, temporal model={temporal_n}. The visual backbone is frozen during training.',f'Precision: no autocast detected in replay code; model operations use FP32; cached feature storage can be FP16.']
 (OUT/'environment.txt').write_text('\n'.join(meta)+'\n')
 return result

def markdown_table(rows):
 if not rows:return ''
 cols=list(rows[0]);return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join('---' for _ in cols)+' |']+['| '+' | '.join(str(row[k]) for k in cols)+' |' for row in rows])

def main():
 data,sources=load();route_tab,overall=route_and_overall(data);stages,stage_tab=stage_analysis(data);dec,decoder_tab=decoder(data)
 improvement=reference_improvement(data);dist=distribution(data);trajectories(data,route_tab);from audit_and_redraw_routes import run as redraw_routes;redraw_routes(data);route_progress(data);cases=response_figure(dec);splits=split_stats(data);comp,temp,sizes=ablation(data);rt=runtime(data)
 shared=[sha(ROOT/('city'+c)/'train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt') for c in CITIES]
 visual=[sha(ROOT/('city'+c)/'train_frames3/checkpoints/visual_retrieval_A_only.pt') for c in CITIES]
 assert len(set(shared))==1 and len(set(visual))==4
 # Explicit consistency checks and provenance; no evaluation state is replayed.
 report={
  'n_routes':8,'n_test_observations':1482,'overall_MLE_m':float(overall['MLE (m)']),'overall_LSR_at_3_pct':float(overall['LSR@3 (%)']),'overall_LSR_at_15_pct':float(overall['LSR@15 (%)']),
  'decoder_mean_shift_matches_full':bool(np.max(np.linalg.norm(dec['mean_shift_xy']-stages['final_mean_shift_xy'],axis=1))<1e-3),
  'same_temporal_checkpoint_all_cities':len(set(shared))==1,'city_specific_visual_checkpoints':len(set(visual))==4,
  'route_state_reset_source':'run_route_inference initializes RouteKalman and hidden=None before each route loop; independent route CSVs were evaluated.',
  'candidate_count':36,'gt_assisted':True,'training_performed':False,'selected_response_samples':cases,
  'source_csv_count':len(sources),'source_root':str(ROOT)
 }
 assert report['decoder_mean_shift_matches_full']
 (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
 stage_note='Corrected Visual Measurement x/y is reconstructed from saved measurement_s/e and the unmodified WaypointRoute.xy_from_se mapping (36 m post-turn route-frame smoothing). All listed stages are map-meter positions. Front Mean Shift uses saved softms_x/y. Supplied Reference uses prior_center_x/y, not frame_reference_x/y (the latter is a separate GT-progress centerline prior used inside final scoring).'
 lines=['# FieldAnchor supplementary thesis experiments','',
  '## Scope and provenance','',
  f'- Source: `{ROOT}`. This run analyzes the existing v39 Full replay and paired ablation/probe CSVs; **no checkpoint was retrained or modified**.',
  '- Protocol: **GT-assisted current-frame reference and hypothesis selection**. It is not a no-GT localization evaluation. Routes are synthetic sequences assembled from individually geotagged Bearing-UAV-90K images, rather than continuous UAV video.',
  '- The four city evaluations share one temporal checkpoint (identical SHA-256) but use four distinct city visual checkpoints. Thus “one complete checkpoint for all cities” would be inaccurate.',
  '- Every test route is independent. The source `run_route_inference` creates a new Kalman state and sets `hidden=None` before processing each route. Stored CSVs cannot independently prove execution-state behavior beyond this source audit.',
  '- Numerical tables below are recomputed from the original per-frame predictions, not copied from thesis prose. MLE/median are in meters; LSR is percentage of frames with Euclidean error at most the threshold.',
  '', '## 1. Route-wise localization — Chapter 4, stability across routes','',markdown_table(route_tab),'',
  f'Pooled sample-level aggregate is saved separately in [overall_results.json](overall_results.json): MLE {overall["MLE (m)"]:.3f} m, LSR@3 {overall["LSR@3 (%)"]:.2f}%, LSR@15 {overall["LSR@15 (%)"]:.2f}%. No Overall row appears in the route table. [CSV](tables/route_wise_results.csv) · [JSON](tables/route_wise_results.json)',
  '', '## 2. Final decoder — Chapter 4, what final Mean Shift itself contributes','',markdown_table(decoder_tab),'',
  'All three decoders use the same 36 final candidate positions and **regularized logits** from the exact same UAV/front Mean Shift/GRU/Kalman output. The logits include visual similarity plus Kalman/current-GT-derived reference priors; raw visual-only similarities were not logged and are not mislabeled as such. Argmax selects the highest logit; Weighted Average uses softmax(logit/0.30); Mean Shift is the original recorded decoder. Mean Shift output matches Full frame by frame. The comparison should be reported as measured even if another decoder wins. [CSV](tables/final_decoder_comparison.csv) · [raw NPZ](raw/final_decoder_predictions.npz)',
  '', '## 3. Stage-wise localization — Chapter 4, stage contributions','',markdown_table(stage_tab),'',stage_note,'',
  'These stages need not improve monotonically: the Kalman output can be worse than the supplied reference while the final GT-conditioned refinement recovers accuracy. [CSV](tables/stage_wise_results.csv) · [raw NPZ](raw/stage_wise_predictions.npz)',
  '', '## 4. Computational cost — Chapter 4, efficiency','',markdown_table([{'Metric':k,'Value':v} for k,v in rt]),'',
  'The original per-frame timer synchronizes CUDA before and after the complete image-tensor-to-coordinate inference, including fresh UAV backbone extraction. Analysis excludes the first 50 observations of each route, leaving 1,082 timed frames. Timing excludes disk I/O, preprocessing, model loading, and satellite gallery construction; concurrent GPU load in the replay was uncontrolled. Hardware/software details and parameter breakdown are in [environment.txt](environment.txt). [CSV](tables/runtime_results.csv) · [raw ms](raw/runtime_raw_ms.csv)',
  '', '## 5–6. Localization trajectories — Chapter 4 result figure and appendix','',
  '[(Four-city PNG)](figures/representative_trajectories.png) · [(PDF)](figures/representative_trajectories.pdf) · [(SVG)](figures/representative_trajectories.svg) · [(Eight-route PNG)](figures/all_eight_routes_appendix.png) · [(PDF)](figures/all_eight_routes_appendix.pdf). Both use each city’s **complete satellite map** as background (original 4096 × 4096 pixels, 1024 × 1024 m; rendered at 2048 pixels for manageable paper files). The green Ground Truth Route joins the saved start/end and only waypoints with an absolute direction change ≥30° using straight segments (a display-only major-turn simplification); the red FieldAnchor line joins **raw, unsmoothed per-frame predictions**. Per-frame GT coordinates remain the source of all numeric localization metrics and are intentionally not drawn as a jagged line or scattered points. The four-city figure uses test_01 for every city; the eight-route figure reports each route MLE. The four official *_50 files have identical XY waypoint arrays, so test_01 route shape is intentionally repeated; each panel uses a different city satellite, UAV observations, predictions, and per-frame error trace. [Original map audit](raw/trajectory_map_audit.json) · [duplicate-route and result-source audit](raw/trajectory_duplicate_audit.json).',
  '', '## 7. Real final responses — Chapter 4 decoder interpretation','',
  '[(PNG)](figures/final_response_visualization.png) · [(PDF)](figures/final_response_visualization.pdf) · [four raw samples](raw/final_response_visualization_samples.npz). Panels are algorithmically selected from unedited 6×6 response logits. “Multiple local lattice peaks” means separate local maxima on the discrete 6×6 grid; it does **not** mean multiple converged Mean Shift modes. All 1,482 recorded final outputs had one active Mean Shift mode. Every panel uses the same 36 × 36 m map-coordinate window about its Kalman center.',
  '', '## 8. Error along route — Chapter 4, sequential behavior','',
  '[(Raw route PNG)](figures/localization_error_over_sequence.png) · [(PDF)](figures/localization_error_over_sequence.pdf) · [(Normalized progress PNG)](figures/error_vs_route_progress.png) · [(PDF)](figures/error_vs_route_progress.pdf) · [101-bin data](raw/error_vs_route_progress.csv). Each route is independently interpolated to 0–100% progress only for the aggregate progress plot; original predictions themselves are never interpolated or smoothed. Shading is route-level IQR.',
  '', '## 9. Improvement over supplied reference — Chapter 4, reference-conditioned refinement','',markdown_table(improvement),'',
  '[(CSV)](tables/route_reference_improvement.csv) · [(PNG)](figures/route_reference_improvement.png) · [(PDF)](figures/route_reference_improvement.pdf). The reference is the saved `prior_center_x/y`, matching the publication audit reference MLE 5.372 m.',
  '', '## 10. Localization error distribution — Chapter 4, robustness','',markdown_table(dist),'',
  '[(CSV)](tables/localization_error_distribution.csv) · [(CDF PNG)](figures/localization_error_cdf.png) · [(PDF)](figures/localization_error_cdf.pdf). CDF x-axis includes all observed errors; w/o Final Refinement is the **whole final refinement stage** removal.',
  '', '## 11. Dataset sequence statistics — Experimental setup','',markdown_table(splits),'',
  'Train/validation counts above use the **temporal trainer** `split_ranges` on each saved `train_01` route: 70%/15% chronological cut points, 16-frame guards, with a one-frame minimum temporal validation interval. The remaining train-route observations are guards/tail, not a new split. The visual checkpoints are reused per city; their original visual train/validation observation counts cannot be reconstructed reliably from these test manifests and are not claimed here. [CSV](tables/dataset_split_statistics.csv)',
  '', '## 12. Existing ablations exported as machine-readable data','',
  '### Components','',markdown_table(comp),'',
  '`w/o Final Refinement` removes final-neighborhood construction, candidate rescoring, and Mean Shift decoding together. It is not a single-function Mean Shift ablation. [CSV](tables/component_ablation.csv)',
  '', '### Temporal context','',markdown_table(temp),'','[CSV](tables/temporal_context_ablation.csv)',
  '', '### Final neighborhood','',markdown_table(sizes),'',
  'Latency here is the per-frame final Mean Shift latency recorded in this v39 replay, recomputed from CSV; it may differ from a separately timed manuscript table. [CSV](tables/final_neighborhood_ablation.csv)',
  '', '### Explicit discrepancy audit against the numbers supplied with this request','',
  '| Row / Metric | Supplied manuscript value | Recomputed original v39 replay | Difference | Likely explanation |',
  '| --- | ---: | ---: | ---: | --- |',
  '| 5×5 MLE (m) | 3.77 | 3.76 (exact 3.762507) | −0.01 displayed | The saved replay rounds to 3.76; the manuscript value may come from another run or a different rounding source. The original manuscript run cannot be reconstructed from this CSV alone. |',
  '| 4×4 Mean Shift time (ms) | 4.04 | 3.83 | −0.21 | Timing run/environment differs; these are actual saved per-frame replay timings. |',
  '| 5×5 Mean Shift time (ms) | 5.00 | 5.61 | +0.61 | Same timing provenance difference. |',
  '| 6×6 Mean Shift time (ms) | 7.52 | 7.21 | −0.31 | Same timing provenance difference. |',
  '| 7×7 Mean Shift time (ms) | 9.50 | 8.59 | −0.91 | Same timing provenance difference. |',
  '| 8×8 Mean Shift time (ms) | 10.27 | 10.96 | +0.69 | Same timing provenance difference. |',
  '', 'The ablation MLE/LSR values otherwise agree with the supplied table at two decimals. Do not silently substitute one timing series for the other in the thesis. Source: per-route frame CSVs under the original v39 replay; evaluator: this script `ablation()`. Exact hardware load of the manuscript timing run is not available.',
  '', '## Consistency and limits','',
  '- Eight routes total 1,482 frames. Full and all ablations align on frame ID, UAV image path, and GT x/y; decoder probe Mean Shift equals original Full for every frame.',
  '- The 6×6 row equals Full by construction from the same original Full predictions.',
  '- The saved CSVs and source code show separate route invocations with fresh recurrent and Kalman state. No model was retrained. Figures use raw saved predictions and no clipped errors.',
  '- The 3.759 m result depends on current-frame GT-informed reference/hypothesis and cannot support a no-GT claim. Direct comparison against published full retrieval methods requires protocol qualification.',
  '- Original paper numbers sometimes used different rounding/timing runs; compare original decimal and recomputed columns before replacing thesis tables. [Machine audit](audit.json).',
  '', 'Recreate all outputs with `python3 report/add_more/generate_thesis_experiments.py` from `/yh/study/uav-sat`.', '']
 (OUT/'README.md').write_text('\n'.join(lines))
 print(json.dumps({'out':str(OUT),'aggregate_mle':float(overall['MLE (m)']),'decoder':decoder_tab,'runtime_mean_ms':dict(rt)['Mean End-to-End Latency (ms)'],'files':len(list(OUT.rglob('*')))},indent=2))

if __name__=='__main__':main()
