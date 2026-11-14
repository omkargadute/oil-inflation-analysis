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
# Include the numerical implementation so an exported notebook contains the model code.
cells[2:2] = [
    nbformat.v4.new_markdown_cell('## Numerical model implementation\n\nShared by all four country fits; no pooling of observations.'),
    nbformat.v4.new_code_cell((root / 'analysis/model.py').read_text(encoding='utf-8')),
]
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
})
path = root / 'notebooks/oil_inflation.ipynb'
def report_cell(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Executing code cell {cell_index + 1}/{len(notebook.cells)}', flush=True)
try:
    NotebookClient(notebook, timeout=1200, kernel_name='python3',
                   resources={'metadata': {'path': str(root)}},
                   on_cell_start=report_cell).execute()
finally:
    nbformat.write(notebook, path)
nbformat.validate(notebook)
print(f'Executed and validated {path}')
