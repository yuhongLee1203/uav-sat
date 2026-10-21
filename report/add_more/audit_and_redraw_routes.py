#!/usr/bin/env python3
"""Audit the city routes and redraw route maps with city-specific error traces."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image

import generate_thesis_experiments as thesis


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(data=None):
    if data is None:
        data, _ = thesis.load()
    root, out = thesis.ROOT, thesis.OUT
    maps, audit, official_xy, error_hashes, image_sets = {}, {}, {}, {}, {}
    for city in thesis.CITIES:
        prep = root / f'city{city}' / 'prepared'
        experiment = json.loads((prep / 'experiment.json').read_text())
        sat = Path(experiment['satellite_image'])
        mpp = float(experiment['mpp'])
        train = thesis.readcsv(prep / 'routes/train_01/manifest.csv')
        ox, oy = float(train[0]['x_m']), float(train[0]['y_m'])
        with Image.open(sat) as im:
            size = im.size
            im.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            background = np.asarray(im.convert('RGB'))
        extent = (-ox, size[0]*mpp-ox, size[1]*mpp-oy, -oy)
        maps[city] = (background, extent)
        official = Path('/yh/study/bearinguav_lastver/loc2traj/traj_wps_gcs') / f'wps{dict(zip("abcd",(34,36,37,38)))[city]}bc_50.json'
        original = json.loads(official.read_text())
        official_xy[city] = np.asarray([original[str(i)]['xy'] for i in range(len(original))], dtype=float)
        prepared = json.loads((prep / 'routes/test_01/waypoints.json').read_text())
        prepared_xy = np.asarray([[x['pixel_x'],x['pixel_y']] for x in prepared['waypoints']])
        assert np.array_equal(prepared_xy, official_xy[city]), f'{city}: prepared/official waypoint mismatch'
        for route in thesis.ROUTES:
            rows = data[(city, route, 'full')]
            gt, pred = thesis.xy(rows, 'gt'), thesis.xy(rows, 'final')
            e = thesis.error(pred, gt)
            saved = np.asarray([float(r['error_final_m']) for r in rows])
            assert np.max(np.abs(e-saved)) < 1e-4, f'{city}/{route}: reported error mismatch'
            assert np.isfinite(e).all()
            image_paths = {r['image_path'] for r in rows}
            assert all(Path(p).is_file() for p in image_paths), f'{city}/{route}: missing UAV images'
            key = f'{city}/{route}'
            audit[key] = {'satellite_image':str(sat), 'satellite_sha256':digest(sat),
                          'official_waypoints':str(official) if route == 'test_01' else None,
                          'uav_first_image':rows[0]['image_path'], 'observations':len(rows),
                          'frame_error_sha256':hashlib.sha256(e.tobytes()).hexdigest(),
                          'mle_m':float(e.mean()), 'lsr3_pct':float(100*np.mean(e<=3))}
            if route == 'test_01':
                error_hashes[city] = audit[key]['frame_error_sha256']
                image_sets[city] = image_paths
    assert all(np.array_equal(official_xy['a'], official_xy[c]) for c in 'bcd'), 'Official test_01 geometry changed'
    assert len({audit[f'{c}/test_01']['satellite_sha256'] for c in 'abcd'}) == 4
    assert len(set(error_hashes.values())) == 4
    assert all(not image_sets[a].intersection(image_sets[b]) for i,a in enumerate('abcd') for b in 'abcd'[i+1:])
    audit_payload = {'finding':'Official city-specific *_50 files have identical pixel XY waypoint arrays. Distinct satellites, UAV image paths, per-frame errors and metrics were verified.',
                     'shared_test01_xy':True, 'official_test01_xy_sha256':hashlib.sha256(official_xy['a'].tobytes()).hexdigest(),
                     'test01_distinct_satellite_sha256':4, 'test01_distinct_error_sha256':4,
                     'test01_uav_image_path_overlap':0, 'routes':audit}
    (out/'raw/trajectory_duplicate_audit.json').write_text(json.dumps(audit_payload, indent=2)+'\n')

    def panel(ax, curve, city, route):
        rows = data[(city,route,'full')]
        gt, pred = thesis.xy(rows,'gt'), thesis.xy(rows,'final')
        e = thesis.error(pred,gt)
        way = thesis.route_points(city,route)
        legs = np.diff(way,axis=0)
        angles = np.degrees(np.arctan2(legs[:-1,0]*legs[1:,1]-legs[:-1,1]*legs[1:,0],
                                       np.sum(legs[:-1]*legs[1:],axis=1)))
        keep = [0]+[i+1 for i,t in enumerate(angles) if abs(t)>=30]+[len(way)-1]
        line = way[keep]
        background, extent = maps[city]
        assert np.all((np.vstack([gt,pred,line])[:,0]>=extent[0]) & (np.vstack([gt,pred,line])[:,0]<=extent[1]))
        ax.imshow(background, extent=extent, origin='upper', interpolation='nearest')
        g, = ax.plot(line[:,0],line[:,1],color='#0b894f',lw=2.0,label='GT route (major turns)')
        p, = ax.plot(pred[:,0],pred[:,1],color='#d03430',lw=1.3,label='FieldAnchor (raw predictions)')
        g.set_path_effects([pe.Stroke(linewidth=3.8,foreground='white',alpha=.85),pe.Normal()])
        p.set_path_effects([pe.Stroke(linewidth=2.6,foreground='white',alpha=.6),pe.Normal()])
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel='Map X (m)',ylabel='Map Y (m)',
               title=f'City {city.upper()} · {route} · MLE {e.mean():.2f} m')
        ax.set_aspect('equal')
        curve.plot(np.arange(len(e)),e,color='#d03430',lw=.9)
        curve.axhline(e.mean(),color='#333333',ls='--',lw=.9)
        curve.set(xlabel='Observation',ylabel='Error (m)',title=f'{city.upper()} {route} · per-frame error')
        curve.set_ylim(0,max(12,float(e.max())*1.08))
        curve.grid(alpha=.2)

    def plot(routes, stem, figsize):
        n=len(routes)
        fig,axes=plt.subplots(n,2,figsize=figsize,layout='constrained',gridspec_kw={'width_ratios':[1,.72]})
        if n==1: axes=np.asarray([axes])
        for i,(city,route) in enumerate(routes): panel(axes[i,0],axes[i,1],city,route)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.008))
        fig.suptitle('Official test_01 shares one waypoint shape across cities; satellite scenes and frame errors differ',fontsize=12)
        fig.savefig(out/f'figures/{stem}.png',dpi=220,bbox_inches='tight')
        fig.savefig(out/f'figures/{stem}.pdf',bbox_inches='tight')
        if stem=='representative_trajectories': fig.savefig(out/f'figures/{stem}.svg',bbox_inches='tight')
        plt.close(fig)
    plot([(c,'test_01') for c in thesis.CITIES],'representative_trajectories',(11,20))
    plot([(c,r) for c in thesis.CITIES for r in thesis.ROUTES],'all_eight_routes_appendix',(11,39))
    print(json.dumps({'figures':[str(out/'figures/representative_trajectories.png'),str(out/'figures/all_eight_routes_appendix.png')],
                      'audit':str(out/'raw/trajectory_duplicate_audit.json'),
                      'test01_mle_m':{c:audit[f'{c}/test_01']['mle_m'] for c in thesis.CITIES}},indent=2))

if __name__=='__main__': run()
