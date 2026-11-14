"""Verify included snapshots by default; explicitly opt into downloading revisions."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
import requests

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--verify', action='store_true')
mode.add_argument('--refresh', action='store_true')
args = parser.parse_args()
for manifest_name in ['manifest.json', 'country_manifest.json']:
    path = root / 'data/raw' / manifest_name
    manifest = json.loads(path.read_text(encoding='utf-8'))
    for item in manifest:
        dest = root / 'data/raw' / item['file']
        if args.refresh:
            response = requests.get(item['url'], timeout=120)
            response.raise_for_status()
            if not response.text.startswith(('observation_date,', 'DATAFLOW,')):
                raise ValueError(f"Unexpected source format: {item['file']}")
            dest.write_bytes(response.content)
            item['retrieved_utc'] = datetime.now(timezone.utc).isoformat()
            item['sha256'] = hashlib.sha256(response.content).hexdigest()
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise ValueError(f"Checksum mismatch: {item['file']}")
        print(f"Verified {item['file']}")
    if args.refresh:
        path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
if args.refresh:
    print('Snapshots refreshed. Rebuild the country audit and analysis to update results.')
