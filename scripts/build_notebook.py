"""Build and execute the notebook from the percent-format analysis source."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
source = (root / 'analysis/oil_inflation.py').read_text(encoding='utf-8')
cells = []
for part in source.split('# %%'):
    if not part.strip():
        continue
    if part.startswith(' [markdown]'):
        lines = part.splitlines()[1:]
        text = '\n'.join(line[2:] if line.startswith('# ') else '' if line == '#' else line for line in lines)
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(part.strip()))
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
})
path = root / 'notebooks/oil_inflation.ipynb'
try:
    NotebookClient(notebook, timeout=600, kernel_name='python3',
                   resources={'metadata': {'path': str(root)}}).execute()
finally:
    nbformat.write(notebook, path)
nbformat.validate(notebook)
print(f'Executed and validated {path}')
