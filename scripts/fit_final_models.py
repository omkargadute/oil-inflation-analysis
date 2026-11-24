"""Populate the content-addressed final-model cache; the notebook verifies and uses it."""
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.data import prepare_panel, COUNTRIES
from analysis.robust_model import fit_cached, simulation_data

panel,_=prepare_panel()
log=[]
def record(fit,label):
    d=fit['diagnostics']
    log.append({'label':label,'cache_key':fit['key'],'cache_hit':fit['cache_hit'],
                'sampling_pass':bool(d.sampling_pass.all()),'max_Rhat':d.r_hat.max(),
                'min_bulk_ESS':d.ess_bulk.min(),'min_tail_ESS':d.ess_tail.min()})
    folder=ROOT/'results/final_project';folder.mkdir(exist_ok=True)
    pd.DataFrame(log).to_csv(folder/'fit_manifest.csv',index=False)

for fi,family in enumerate(['gaussian','student']):
    frame,truth=simulation_data(family)
    for adjusted in [False,True]:
        label=f'simulation/{family}/adjusted={adjusted}'
        record(fit_cached(frame,adjusted=adjusted,family=family,seed=3000+fi*100+int(adjusted),label=label),label)
for ci,iso in enumerate(COUNTRIES):
    frame=panel[panel.iso3.eq(iso)].reset_index(drop=True)
    for fi,family in enumerate(['gaussian','student']):
        for adjusted in [False,True]:
            label=f'full/{iso}/{family}/adjusted={adjusted}'
            record(fit_cached(frame,adjusted=adjusted,family=family,seed=10000+ci*1000+fi*100+int(adjusted),label=label),label)
    train=frame[frame.date<'2020-01-01']
    for fi,family in enumerate(['gaussian','student']):
        for oil in ([True] if family=='gaussian' else [True,False]):
            label=f'training/{iso}/{family}/oil={oil}'
            record(fit_cached(train,family=family,oil=oil,seed=20000+ci*1000+fi*100+int(oil),label=label),label)
if not all(row['sampling_pass'] for row in log):
    raise RuntimeError('Inspect fit_manifest.csv: some sampling diagnostics did not pass')
print('Completed and checked',len(log),'fits.')
