"""Verified raw-snapshot preparation shared by the final report notebook."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COUNTRIES = {'USA': 'United States', 'CHN': 'China', 'JPN': 'Japan', 'CAN': 'Canada'}
FLOWS = {
    'USA': 'DSD_PRICES@DF_PRICES_ALL',
    'CHN': 'DSD_PRICES@DF_PRICES_ALL',
    'JPN': 'DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL',
    'CAN': 'DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL',
}

def prepare_panel(root=ROOT):
    """Return a complete four-country panel and source verification table.

    Reference-month dates are calendar-aligned before any lag is constructed.
    Original snapshots are never revised, interpolated or silently downloaded.
    """
    root = Path(root)
    frames, audit = [], []
    for manifest_name in ['manifest.json', 'country_manifest.json']:
        manifest = json.loads((root/'data/raw'/manifest_name).read_text(encoding='utf-8'))
        for item in manifest:
            path = root/'data/raw'/item['file']
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item['sha256']:
                raise ValueError(f"Source checksum mismatch: {item['file']}")
            audit.append({'file': item['file'], 'verified': True,
                          'recorded_retrieval_utc': item['retrieved_utc'], 'url': item['url']})
            if manifest_name == 'country_manifest.json':
                frame = pd.read_csv(path)
                frame['flow'] = item['flow']
                frames.append(frame)
    cpi = pd.concat(frames, ignore_index=True)
    if cpi.duplicated(['REF_AREA', 'flow', 'ADJUSTMENT', 'TIME_PERIOD']).any():
        raise ValueError('Duplicate CPI country/flow/adjustment/month keys')
    base = pd.DataFrame({'date': pd.date_range('1999-01-01', '2024-12-01', freq='MS')})
    for filename, column in [('brent_direct.csv', 'oil_usd'), ('igrea.csv', 'global_activity')]:
        source = pd.read_csv(root/'data/raw'/filename, parse_dates=[0])
        source.columns = ['date', column]
        base = base.merge(source, on='date', how='left', validate='one_to_one')
    base['oil_pct'] = base.oil_usd.pct_change(fill_method=None)*100
    base['T'] = base.oil_pct.shift(1)/10
    base['C'] = base.global_activity.shift(2)/100
    panels = []
    for iso, name in COUNTRIES.items():
        selected = cpi[(cpi.REF_AREA == iso) & (cpi.flow == FLOWS[iso]) & (cpi.ADJUSTMENT == 'N')].copy()
        for column, expected in [('FREQ','M'), ('MEASURE','CPI'), ('UNIT_MEASURE','IX'),
                                 ('METHODOLOGY','N'), ('EXPENDITURE','_T'), ('TRANSFORMATION','_Z')]:
            if selected.empty or not selected[column].eq(expected).all():
                raise ValueError(f'{iso}: unexpected CPI definition ({column})')
        selected['date'] = pd.to_datetime(selected.TIME_PERIOD)
        d = base.merge(selected[['date','OBS_VALUE']].rename(columns={'OBS_VALUE':'cpi'}),
                       on='date', how='left', validate='one_to_one')
        d['monthly'] = d.cpi.pct_change(fill_method=None)*100
        d['yearly'] = d.cpi.pct_change(12, fill_method=None)*100
        d['last_monthly'] = d.monthly.shift(1)
        d['last_year_same_month'] = d.monthly.shift(12)
        d['month'] = d.date.dt.month
        d['country'], d['iso3'], d['adjustment'], d['flow'] = name, iso, 'N', FLOWS[iso]
        d = d[d.date.ge('2000-01-01')].reset_index(drop=True)
        if len(d) != 300 or not np.isfinite(d[['cpi','monthly','yearly','T','C']].to_numpy()).all():
            raise ValueError(f'{iso}: incomplete or non-finite core analysis series')
        if not (d.cpi > 0).all():
            raise ValueError(f'{iso}: non-positive CPI level')
        panels.append(d)
    panel = pd.concat(panels, ignore_index=True)
    if panel.duplicated(['iso3','date']).any() or len(panel) != 1200:
        raise ValueError('Unexpected analysis panel keys or size')
    return panel, pd.DataFrame(audit)
