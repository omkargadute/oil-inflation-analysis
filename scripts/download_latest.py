"""Download separately versioned current snapshots; retain the original study inputs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,io
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/raw/latest'

def retrieve(item):
    response=requests.get(item['url'],timeout=120)
    response.raise_for_status()
    frame=pd.read_csv(io.BytesIO(response.content))
    required=['TIME_PERIOD','OBS_VALUE','REF_AREA'] if 'flow' in item else ['observation_date']
    if not set(required)<=set(frame.columns) or frame.empty:
        raise ValueError('Unexpected or empty source: '+item['file'])
    return item,response.content

def main():
    sources=[]
    for manifest in ['manifest.json','country_manifest.json']:
        for item in json.loads((ROOT/'data/raw'/manifest).read_text()):
            item={k:v for k,v in item.items() if k in ['file','url','flow']}
            item['url']=item['url'].replace('&endPeriod=2024-12','')
            sources.append(item)
    sources.append({'file':'daily_brent.csv','url':'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU'})
    with ThreadPoolExecutor(max_workers=5) as pool:
        downloaded=list(pool.map(retrieve,sources))
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for item,content in downloaded:
        (OUT/item['file']).write_bytes(content)
        item.update(retrieved_utc=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(content).hexdigest())
        manifest.append(item)
        print('Downloaded and parsed',item['file'],len(content),'bytes',flush=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
