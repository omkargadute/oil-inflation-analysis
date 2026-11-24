"""Verify current snapshots and preserve calendar gaps before computing growth."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from analysis.data import COUNTRIES,FLOWS,prepare_panel

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/latest'
OUT=ROOT/'results/latest'

def prepare_latest():
    manifest=json.loads((RAW/'manifest.json').read_text())
    frames=[];audit=[]
    for item in manifest:
        path=RAW/item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Current snapshot hash mismatch: '+item['file'])
        audit.append({'file':item['file'],'verified':True,'retrieved_utc':item['retrieved_utc'],'url':item['url']})
        if 'flow' in item:
            f=pd.read_csv(path);f['flow']=item['flow'];frames.append(f)
    cpi=pd.concat(frames,ignore_index=True)
    cap=pd.Timestamp(datetime.now(timezone.utc).date()).replace(day=1)-pd.DateOffset(months=1)
    calendar=pd.date_range('1999-01-01',cap,freq='MS')
    base=pd.DataFrame({'date':calendar});coverage=[]
    for file,column in [('brent_direct.csv','oil_usd'),('igrea.csv','global_activity')]:
        d=pd.read_csv(RAW/file,parse_dates=[0]);d.columns=['date',column]
        if d.date.duplicated().any():raise ValueError('Duplicate predictor date')
        d[column]=pd.to_numeric(d[column],errors='coerce')
        coverage.append({'series':column,'latest_observed_month':d.loc[d[column].notna(),'date'].max()})
        base=base.merge(d,on='date',how='left',validate='one_to_one')
    base['oil_pct']=100*base.oil_usd.pct_change(fill_method=None)
    base['T']=base.oil_pct.shift(1)/10;base['C']=base.global_activity.shift(2)/100
    panels=[];missing=[];last=[]
    for iso,name in COUNTRIES.items():
        d=cpi[cpi.REF_AREA.eq(iso)&cpi.flow.eq(FLOWS[iso])&cpi.ADJUSTMENT.eq('N')].copy()
        for key,value in [('FREQ','M'),('MEASURE','CPI'),('UNIT_MEASURE','IX'),
                          ('METHODOLOGY','N'),('EXPENDITURE','_T'),('TRANSFORMATION','_Z')]:
            if d.empty or not d[key].eq(value).all():raise ValueError(iso+': unexpected CPI definition '+key)
        d['date']=pd.to_datetime(d.TIME_PERIOD)
        if d.date.duplicated().any():raise ValueError(iso+': duplicated CPI month')
        d['cpi']=pd.to_numeric(d.OBS_VALUE,errors='raise')
        if not (d.cpi.dropna()>0).all():raise ValueError('Nonpositive CPI')
        end=min(cap,d.loc[d.cpi.notna(),'date'].max());last.append(end)
        coverage.append({'series':iso+' CPI','latest_observed_month':end})
        g=base.merge(d[['date','cpi']],on='date',how='left',validate='one_to_one')
        # Reindex first, then calculate growth. November cannot bridge a missing October.
        g['monthly']=100*g.cpi.pct_change(fill_method=None)
        g['yearly']=100*g.cpi.pct_change(12,fill_method=None)
        g['month']=g.date.dt.month;g['iso3']=iso;g['country']=name
        g['flow']=FLOWS[iso];g['adjustment']='N'
        for date in g.loc[g.date.between('2000-01-01',end)&g.cpi.isna(),'date']:
            missing.append({'iso3':iso,'date':date,'variable':'cpi','action':'missing; not imputed'})
        panels.append(g)
    end=min(last+[base.loc[base[['T','C']].notna().all(axis=1),'date'].max()])
    panel=pd.concat([g[g.date.between('2000-01-01',end)] for g in panels],ignore_index=True)
    if panel[['T','C']].isna().any().any() or panel.duplicated(['iso3','date']).any():
        raise ValueError('Invalid common predictor support or panel keys')
    expected_missing={('USA',pd.Timestamp('2025-10-01'))}
    actual_missing={(r['iso3'],r['date']) for r in missing if r['date']<=end}
    if actual_missing-expected_missing:raise ValueError('New CPI gaps require explicit review: '+str(actual_missing))
    daily=pd.read_csv(RAW/'daily_brent.csv',parse_dates=[0]);daily.columns=['date','oil_usd']
    daily['oil_usd']=pd.to_numeric(daily.oil_usd,errors='coerce')
    today=pd.Timestamp(datetime.now(timezone.utc).date())
    daily=daily[daily.date.le(today)&daily.oil_usd.notna()].reset_index(drop=True)
    return panel,pd.DataFrame(audit),pd.DataFrame(coverage),pd.DataFrame(missing),daily

def write_data_audit():
    OUT.mkdir(parents=True,exist_ok=True)
    panel,audit,coverage,missing,daily=prepare_latest()
    old,_=prepare_panel();matched=old.merge(panel,on=['iso3','date'],suffixes=('_old','_new'),validate='one_to_one')
    revisions=[]
    for iso in COUNTRIES:
        g=matched[matched.iso3.eq(iso)]
        for col in ['cpi','monthly','T','C']:
            delta=g[col+'_new']-g[col+'_old']
            revisions.append({'iso3':iso,'variable':col,'overlap_n':len(g),'changed_n':int((delta.abs()>1e-8).sum()),
                              'max_absolute_revision':delta.abs().max()})
    for name,table in [('source_verification',audit),('availability',coverage),('missing_observations',missing),
                       ('historical_revisions',pd.DataFrame(revisions)),('daily_brent',daily)]:
        table.to_csv(OUT/(name+'.csv'),index=False)
    panel.to_csv(ROOT/'data/processed/multicountry_monthly_latest.csv',index=False)
    return panel,daily
