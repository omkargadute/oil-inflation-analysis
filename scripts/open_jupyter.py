"""Open the active report in a local authenticated background JupyterLab server."""
from pathlib import Path
import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from urllib.parse import urlencode, urljoin, urlparse
import requests
from jupyter_server.serverapp import list_running_servers

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT/'.local-archive/jupyter'
LOGS.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--no-browser', action='store_true')
args = parser.parse_args()

def matching_server():
    for info in list_running_servers():
        if (Path(info.get('root_dir', '.')).resolve() == ROOT
                and urlparse(info['url']).hostname in ['127.0.0.1', 'localhost']):
            return info
    return None

server = matching_server()
if server is None:
    for port in range(8888,8900):
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1',port))
                break
            except OSError:
                continue
    else:
        raise RuntimeError('No free Jupyter port between 8888 and 8899')
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
    with (LOGS/'server.log').open('a',encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable,'-m','jupyterlab','--no-browser','--ip=127.0.0.1',
                                    f'--port={port}','--ServerApp.port_retries=0',
                                    '--ServerApp.root_dir='+str(ROOT),
                                    'notebooks/final_project.ipynb'],cwd=ROOT,
                                   stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
    (LOGS/'server.pid').write_text(str(process.pid))
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError('JupyterLab exited during startup; inspect the ignored local server log')
        server = matching_server()
        if server is not None:
            break
        time.sleep(.5)
    else:
        raise RuntimeError('JupyterLab startup timed out; inspect the ignored local server log')

auth = {'token':server['token']} if server.get('token') else {}
response = requests.get(urljoin(server['url'],'api/contents/notebooks/final_project.ipynb'),
                        params={**auth,'content':'0'},timeout=15)
response.raise_for_status()
if response.json()['type'] != 'notebook':
    raise RuntimeError('Jupyter did not return the expected report notebook')
kernels = requests.get(urljoin(server['url'],'api/kernelspecs'),params=auth,timeout=15)
kernels.raise_for_status()
if 'oil-inflation' not in kernels.json()['kernelspecs']:
    raise RuntimeError('The project kernel is not visible to JupyterLab')
target = urljoin(server['url'],'lab/tree/notebooks/final_project.ipynb')
if not args.no_browser:
    webbrowser.open(target + ('?' + urlencode(auth) if auth else ''))
# Authentication tokens remain in Jupyter's protected runtime files/browser, not terminal output.
print('JupyterLab and project kernel verified:',target)
print('Stop the server with: .venv\\Scripts\\python.exe -m jupyter server stop',server['port'])
