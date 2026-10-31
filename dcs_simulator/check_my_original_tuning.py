"""Summarize the original Model B evaluation metrics from model_b_recheck.csv.
Run after check_model_b_metrics.py, in the same dcs_simulator folder.
No simulation rerun and no parameter changes.
"""
from pathlib import Path
import csv
import math
import statistics

base = Path(__file__).resolve().parent
path = base / 'output' / 'evaluation' / 'model_b_recheck.csv'
if not path.exists():
    raise SystemExit('Missing model_b_recheck.csv. Run: python check_model_b_metrics.py first')
with path.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print('Reference: raw ave_gtau_intra40 in Dataset_phantom-intra40.mat')
print('Model: current Model B configuration; no parameter tuning')
print('R2_corr = Pearson correlation squared (original evaluation.py metric)')
print('R2_SS = 1 - SSE/SST (slide metric)')

for selection in ('all', 'closest'):
    print('\n===', selection.upper(), '===')
    for sds in (1, 2, 3):
        subset = [r for r in rows if int(r['sds']) == sds and (selection == 'all' or int(r['closest']) == 1)]
        def nums(key):
            return [float(r[key]) for r in subset if math.isfinite(float(r[key]))]
        rmse = nums('rmse')
        corr = nums('r2_corr')
        ss = nums('r2_ss')
        print(f'SDS {sds}: n={len(subset)}, RMSE_pooled={math.sqrt(statistics.mean(x*x for x in rmse)):.5f}, '
              f'R2_corr_median={statistics.median(corr):.4f}, R2_corr_mean={statistics.mean(corr):.4f}, '
              f'R2_SS_median={statistics.median(ss):.4f}, R2_SS_mean={statistics.mean(ss):.4f}')

print('\nNOTE: This is NOT a comparison against a different reference dataset.')
print('It shows how the SAME predictions score under two different R2 definitions.')