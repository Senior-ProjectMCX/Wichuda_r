"""Regenerate MCX detected-photon histories for the two-layer DCS phantom.

Run from dcs_simulator/: python simulation.py --dry-run
                     python simulation.py --name trial_01
                     python simulation.py --name trial_01 --activate

--activate replaces the GUI's detp_core_v1.npy only after making a backup.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from parameters import NX, NY, VOXEL_SIZE_MM, SDS_MM
from phantom import create_two_layer_phantom, create_mcx_properties, validate_phantom

HERE = Path(__file__).resolve().parent
OUT = HERE / 'output'


def make_config(nphoton=10_000_000, radius_voxels=3.0, gpu_id=1):
    volume = create_two_layer_phantom()
    validate_phantom(volume)
    srcpos = [NX / 2.0, NY / 2.0, 0.0]
    detpos = [[srcpos[0] + float(sds) / VOXEL_SIZE_MM,
               srcpos[1], -0.5, float(radius_voxels)] for sds in SDS_MM]
    if any(not (0 <= d[0] < NX and 0 <= d[1] < NY) for d in detpos):
        raise ValueError('Detector outside phantom: check SDS and geometry')
    return {
        'nphoton': int(nphoton), 'vol': volume,
        'prop': create_mcx_properties(), 'srcpos': srcpos,
        'srcdir': [0.0, 0.0, 1.0], 'detpos': detpos,
        'issavedet': 1, 'savedetflag': 'DPM',
        'unitinmm': VOXEL_SIZE_MM,
        'tstart': 0.0, 'tend': 5e-9, 'tstep': 5e-9,
        'autopilot': 1, 'gpuid': int(gpu_id),
    }


def run(nphoton=10_000_000, radius_voxels=3.0, gpu_id=1,
        name='new_run', dry_run=False, activate=False):
    if not name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Name must contain letters, digits, hyphens, or underscores')
    cfg = make_config(nphoton, radius_voxels, gpu_id)
    meta = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'nphoton': cfg['nphoton'], 'unitinmm': cfg['unitinmm'],
        'volume_shape': list(cfg['vol'].shape),
        'prop': np.asarray(cfg['prop']).tolist(),
        'srcpos': cfg['srcpos'], 'srcdir': cfg['srcdir'],
        'detpos': cfg['detpos'], 'savedetflag': cfg['savedetflag'],
        'gpu_id': cfg['gpuid'], 'sds_mm': np.asarray(SDS_MM).tolist(),
        'note': 'MCX configuration. Check partial-pathlength units before DCS processing.',
    }
    print(json.dumps(meta, indent=2))
    if dry_run:
        print('Dry run only: MCX not executed.')
        return None
    import pmcx  # keep dry-run usable without CUDA/PMCX
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f'detp_{name}.npy'
    if target.exists():
        raise FileExistsError(f'Output exists: {target}; choose another --name')
    result = pmcx.run(cfg)
    if 'detp' not in result:
        raise RuntimeError(f'MCX did not return detp; keys={list(result)}')
    detp = np.asarray(result['detp'])
    if detp.ndim != 2 or detp.shape[0] != 5:
        raise ValueError(f'GUI expects detp shape (5,N), got {detp.shape}')
    if not np.all(np.isfinite(detp)):
        raise ValueError('detp contains nonfinite values')
    np.save(target, detp)
    meta['detected_photons'] = int(detp.shape[1])
    (OUT / f'detp_{name}.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Saved {target}; detected photons: {detp.shape[1]}')
    if activate:
        import shutil
        active = OUT / 'detp_core_v1.npy'
        if active.exists():
            backup = OUT / f'detp_core_v1_backup_{datetime.now():%Y%m%d_%H%M%S}.npy'
            shutil.copy2(active, backup)
            print(f'Backed up existing GUI data: {backup}')
        shutil.copy2(target, active)
        print(f'Activated for GUI: {active}')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nphoton', type=int, default=10_000_000)
    parser.add_argument('--radius-voxels', type=float, default=3.0)
    parser.add_argument('--gpu-id', type=int, default=1)
    parser.add_argument('--name', default='new_run')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args()
    if args.nphoton < 1 or args.radius_voxels <= 0:
        parser.error('nphoton and radius-voxels must be positive')
    run(**vars(args))
