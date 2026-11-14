"""Download/verify OECD country CPI snapshots and audit 2000-2024 coverage.

Run with --refresh to fetch sources; otherwise reproduce from the saved snapshots.
This checks availability and transformations, not full national-definition harmonization.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
NAMES = {'USA':'United States','CHN':'China','JPN':'Japan','CAN':'Canada',
         'GBR':'United Kingdom','DEU':'Germany','FRA':'France','ITA':'Italy',
         'ESP':'Spain','KOR':'South Korea','IND':'India','BRA':'Brazil',
         'MEX':'Mexico','ZAF':'South Africa','AUS':'Australia','NOR':'Norway','CHE':'Switzerland'}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--refresh', action='store_true')
args = parser.parse_args()
manifest_path = RAW / 'country_manifest.json'
if args.refresh:
    manifest = []
    for flow, filename in [('DSD_PRICES@DF_PRICES_ALL','oecd_country_cpi_1999.csv'),
                           ('DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL','oecd_country_cpi_2018.csv')]:
        url = ('https://sdmx.oecd.org/public/rest/v1/data/OECD.SDD.TPS,' + flow + ',1.0/'
               + '+'.join(NAMES) + '.M.N.CPI.IX._T.N+S._Z'
               + '?startPeriod=1999-01&endPeriod=2024-12&format=csvfile')
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        if not response.text.startswith('DATAFLOW,'):
            raise ValueError('Unexpected response format')
        (RAW / filename).write_bytes(response.content)
        manifest.append({'file':filename,'flow':flow,'url':url,
                         'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                         'sha256':hashlib.sha256(response.content).hexdigest()})
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
frames = []
for item in manifest:
    path = RAW / item['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    frame = pd.read_csv(path)
    assert not frame.duplicated(['REF_AREA','ADJUSTMENT','TIME_PERIOD']).any()
    assert (frame.OBS_VALUE.dropna() > 0).all()
    assert frame.FREQ.eq('M').all() and frame.UNIT_MEASURE.eq('IX').all()
    frame['flow'] = item['flow']; frame['date'] = pd.to_datetime(frame.TIME_PERIOD)
    frames.append(frame)
all_data = pd.concat(frames,ignore_index=True)
calendar = pd.DataFrame({'date':pd.date_range('1999-01-01','2024-12-01',freq='MS')})
oil = pd.read_csv(RAW/'brent_direct.csv',parse_dates=[0]); oil.columns=['date','oil_usd']
oil['oil_change_lag1'] = oil.oil_usd.pct_change(fill_method=None).shift(1)*100
demand = pd.read_csv(RAW/'igrea.csv',parse_dates=[0]); demand.columns=['date','global_activity']
demand['global_activity_lag2'] = demand.global_activity.shift(2)
detail = []
series = {}
for (area,flow,adjustment),g in all_data.groupby(['REF_AREA','flow','ADJUSTMENT']):
    d = calendar.merge(g[['date','OBS_VALUE']],on='date',how='left',validate='one_to_one')
    d = d.rename(columns={'OBS_VALUE':'cpi'})
    d['inflation_mom'] = d.cpi.pct_change(fill_method=None)*100
    d['inflation_yoy'] = d.cpi.pct_change(12,fill_method=None)*100
    d = d.merge(oil[['date','oil_change_lag1']],on='date',how='left',validate='one_to_one')
    d = d.merge(demand[['date','global_activity_lag2']],on='date',how='left',validate='one_to_one')
    d = d[d.date.ge('2000-01-01')].copy()
    valid = d.cpi.notna()
    joint = d[['inflation_mom','oil_change_lag1','global_activity_lag2']].notna().all(axis=1)
    detail.append({'country':NAMES[area],'iso3':area,'flow':flow,'adjustment':adjustment,
                   'cpi_months':int(valid.sum()),'mom_months':int(d.inflation_mom.notna().sum()),
                   'yoy_months':int(d.inflation_yoy.notna().sum()),'aligned_mom_rows':int(joint.sum()),
                   'first_cpi':d.loc[valid,'date'].min().strftime('%Y-%m'),
                   'last_cpi':d.loc[valid,'date'].max().strftime('%Y-%m')})
    series[(area,flow,adjustment)] = d
details = pd.DataFrame(detail)
summary=[];panels=[]
for area,name in NAMES.items():
    candidate = details[(details.iso3==area)&(details.adjustment=='N')].sort_values(['aligned_mom_rows','flow'],ascending=False)
    best = candidate.iloc[0]
    sa = details[(details.iso3==area)&(details.adjustment=='S')]
    summary.append({'country':name,'iso3':area,'unadjusted_cpi_months':int(best.cpi_months),
                    'unadjusted_aligned_mom_rows':int(best.aligned_mom_rows),
                    'unadjusted_yoy_months':int(best.yoy_months),'selected_flow':best.flow,
                    'full_period_official_SA':bool((sa.aligned_mom_rows==300).any()),
                    'last_unadjusted_month':best.last_cpi})
    d=series[(area,best.flow,'N')].copy();d['country']=name;d['iso3']=area
    d['adjustment']='N';d['flow']=best.flow;panels.append(d)
details.to_csv(ROOT/'results/country_series_coverage.csv',index=False)
summary=pd.DataFrame(summary);summary.to_csv(ROOT/'results/country_coverage.csv',index=False)
pd.concat(panels,ignore_index=True).to_csv(ROOT/'data/processed/country_cpi_candidate_panel.csv',index=False)
print(summary.to_string(index=False))
print('\nDetailed Japan coverage:')
print(details[details.iso3.eq('JPN')].to_string(index=False))
print('\nComplete unadjusted country series:',int(summary.unadjusted_aligned_mom_rows.eq(300).sum()),'of',len(NAMES),'checked')
