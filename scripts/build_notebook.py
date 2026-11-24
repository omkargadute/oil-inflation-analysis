"""Build and execute the notebook from the percent-format analysis source."""
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--target', choices=['final', 'exploratory'], default='final')
parser.add_argument('--kernel', default='oil-inflation')
parser.add_argument('--build-only', action='store_true')
args = parser.parse_args()
filename = 'final_project' if args.target == 'final' else 'oil_inflation'
source = (root / 'analysis' / (filename + '.py')).read_text(encoding='utf-8')
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
if args.target == 'exploratory':
    cells[2:2] = [
        nbformat.v4.new_markdown_cell('## Numerical model implementation\n\nShared by all four country fits; no pooling of observations.'),
        nbformat.v4.new_code_cell((root / 'analysis/model.py').read_text(encoding='utf-8')),
    ]
else:
    cells.append(nbformat.v4.new_markdown_cell(
        '## Appendix. Computational implementation\n\n'
        'The following source is included for code-inclusive exports and review. '
        'It is imported by the executed analysis above; these listings do not refit models.'))
    for module in ['analysis/data.py','analysis/model.py','analysis/robust_model.py',
                   'analysis/extension_model.py','analysis/compiled_sampler.py','analysis/latest_data.py',
                   'analysis/combined_sampler.py','analysis/combined_model.py','analysis/combined_analysis.py']:
        cells.append(nbformat.v4.new_markdown_cell(
            '### '+module+'\n\n```python\n'+(root/module).read_text(encoding='utf-8')+'\n```',
            metadata={'tags':['implementation-appendix']}))
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python (Oil Inflation Project)', 'language': 'python', 'name': args.kernel}
})
path = root / 'notebooks' / (filename + '.ipynb')
def report_cell(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Executing code cell {cell_index + 1}/{len(notebook.cells)}', flush=True)
try:
    if not args.build_only:
        NotebookClient(notebook, timeout=1200, kernel_name=args.kernel,
                       resources={'metadata': {'path': str(root)}},
                       on_cell_start=report_cell).execute()
finally:
    nbformat.write(notebook, path)
nbformat.validate(notebook)
print(f'{"Built" if args.build_only else "Executed"} and validated {path}')
