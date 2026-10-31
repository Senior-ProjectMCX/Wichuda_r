"""Evaluate Model B against its original raw physical phantom reference.
Place this file in Wichuda_r-main/dcs_simulator/ and run python check_model_b_metrics.py.
No MCX rerun required; uses output/detp_core_v1.npy.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.io import loadmat
from gui import (TAU_SECONDS, fit_temporary_physical_reference,
                 calculate_dynamic_g1, calculate_g2, flow_to_alpha_db)

BASE=Path(__file__).resolve().parent
mat=loadmat(BASE/'reference_data'/'Dataset_phantom-intra40.mat')
meta=mat['Dataset_set_intra40']
g2=mat['ave_gtau_intra40']
detp=np.load(BASE/'output'/'detp_core_v1.npy')

def r2_ss(ref, pred):
    den=np.sum((ref-ref.mean())**2)
    return 1-np.sum((ref-pred)**2)/den if den>0 else np.nan

def r2_corr(ref,pred):
    if np.std(ref)==0 or np.std(pred)==0:return np.nan
    return float(np.corrcoef(ref,pred)[0,1]**2)

conditions=sorted(set(zip(meta[0],meta[1])))
rows=[]
for k,(shallow,deep) in enumerate(conditions,1):
    if shallow==0 and deep==0: continue
    indices=np.where((meta[0]==shallow)&(meta[1]==deep))[0]
    # Same intended closest-to-preset rule as the slide: min absolute flow deviation sum.
    # Exact tie-breaking may differ from teammate's script.
    distances=np.abs(meta[2,indices]-shallow)+np.abs(meta[3,indices]-deep)
    closest=indices[np.argmin(distances)]
    for sds in (1,2,3):
        raw=g2[sds-1,:,indices].T
        fitted,_,_=fit_temporary_physical_reference(raw)
        beta=float(fitted[0]-1)
        g1=calculate_dynamic_g1(detp,TAU_SECONDS,sds,
              flow_to_alpha_db(shallow),flow_to_alpha_db(deep))
        pred=calculate_g2(g1,beta)
        for idx in indices:
            ref=g2[sds-1,:,idx]
            if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(pred))):continue
            rows.append(dict(condition_shallow=shallow,condition_deep=deep,
                trial_index=int(idx),closest=int(idx==closest),sds=sds,
                rmse=float(np.sqrt(np.mean((pred-ref)**2))),
                r2_ss=r2_ss(ref,pred),r2_corr=r2_corr(ref,pred),
                beta=beta))
    if k%20==0: print(f'Processed {k}/{len(conditions)} conditions',flush=True)

out=BASE/'output'/'evaluation'/'model_b_recheck.csv'
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for subset in ('closest','all'):
    print('\n'+subset.upper())
    for sds in (1,2,3):
        r=[x for x in rows if x['sds']==sds and (subset=='all' or x['closest'])]
        rmse=np.array([x['rmse'] for x in r]);rs=np.array([x['r2_ss'] for x in r]);rc=np.array([x['r2_corr'] for x in r])
        print(f'SDS{sds}: n={len(r)}, pooled_RMSE={np.sqrt(np.mean(rmse**2)):.5f}, '
              f'R2_SS_median={np.nanmedian(rs):.4f}, R2_SS_mean={np.nanmean(rs):.4f}, '
              f'R2_corr_squared_median={np.nanmedian(rc):.4f}')
print('Saved:',out)