# Jupyter environment and development workflow

The project uses an isolated Python 3.11 environment in `.venv/`. It does not inherit global site packages. JupyterLab and the dedicated **Python (Oil Inflation Project)** kernel live in this environment. The virtual environment, runtime credentials and caches are not committed.

## Open the notebook

From the project folder in PowerShell:

```powershell
.\scripts\start_jupyter.ps1
```

This launches JupyterLab bound to `127.0.0.1`, with normal token authentication, and opens the browser. Open **notebooks/final_project.ipynb** and select **Python (Oil Inflation Project)**. Stop the foreground server with Ctrl+C. Use `-NoBrowser` if you prefer to open the displayed local link yourself, or `-Port 8889` if 8888 is occupied.

For a background server, run `.\.venv\Scripts\python.exe scripts/open_jupyter.py`. It opens the report, verifies that the dedicated kernel is available, and stores server logs in the ignored local archive. It reuses a matching local project server if one is already running. Stop a background server with `.\.venv\Scripts\python.exe -m jupyter server stop 8888` (use the port printed by the launcher).

Activation is optional: these scripts invoke `.venv\Scripts\python.exe` directly. If PowerShell blocks a script, use the equivalent command without changing the machine execution policy:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab --ip=127.0.0.1
```

## Recreate the environment

```powershell
.\scripts\setup_notebook.ps1
```

This creates `.venv` if needed, installs `requirements-lock.txt` when available, checks dependency consistency, and registers the local kernel. If intentionally updating dependencies, use `-RefreshDependencies` to resolve `requirements-dev.txt` and refresh the lock file. The core analysis dependencies remain in `requirements.txt`.

The lock file captures this Windows/Python 3.11 environment, including platform-specific packages. On another operating system, resolve `requirements-dev.txt` instead of assuming the Windows lock is portable.

## Edit and rerun

The report's source of truth is `analysis/final_project.py` in percent-cell format. Rebuild and execute it with:

```powershell
.\.venv\Scripts\python.exe scripts/build_notebook.py
```

It writes `notebooks/final_project.ipynb` and development outputs in `results/final_project/`. All data come from saved, checksum-verified snapshots. No downloads occur during notebook execution. Editing generated notebook cells directly is convenient for experiments, but copy accepted changes back into the source before rebuilding, which replaces the notebook.

To reproduce the earlier full exploratory analysis instead:

```powershell
.\.venv\Scripts\python.exe scripts/build_notebook.py --target exploratory
```

That run is slower and updates the existing exploratory results. `--build-only` builds notebook cells without executing them; `--kernel python3` selects another already configured kernel if required.

## Current progress

**Current final notebook:** the main analysis uses January 2000–October 2026 with Bayesian integration of missing US monthly outcomes. Precompute with `.\.venv\Scripts\python.exe -m analysis.combined_analysis`; outputs are in `results/combined_project/`, caches in `.local-archive/combined-model-cache/`. See [COMBINED_NOTEBOOK.md](COMBINED_NOTEBOOK.md). The baseline/extension notes below describe retained historical implementations.

The notebook includes the [current-release extension](LATEST_DATA.md), with saved monthly data through October 2026. To explicitly refresh its separately versioned inputs, use `.\.venv\Scripts\python.exe scripts/download_latest.py`; to precompute its 12 fits and recent evaluation, use `.\.venv\Scripts\python.exe -m analysis.latest_analysis`. Notebook execution itself makes no download requests. The extension handles the missing US CPI month explicitly and has its own `.local-archive/latest-model-cache/`.

The final notebook also runs the five experiment families documented in [STRENGTHENING.md](STRENGTHENING.md). Precompute them with `.\.venv\Scripts\python.exe scripts/run_strengthening.py`. Their independent cache is `.local-archive/strengthening-cache/`; model code, data, settings and Numba backend version determine fit keys. Initial JIT compilation can take about a minute, followed by the experiment fits. Saved tables and plots are in `results/strengthening/`. Unchanged completed suites read saved results after verifying source/data signatures.

All nine computational report sections are implemented and executed: research question, data, EDA, causal hypothesis, priors, separate synthetic recovery, four-country Gaussian/Student-t estimation, posterior evaluation, and findings. The 32 fits include adjusted/unadjusted real-data models, simulations and training-period forecasts, including a matched no-oil comparison. All sampling checks pass; predictive and causal limitations are discussed rather than treated as resolved.

The first clean run can take several minutes. Fits are cached under `.local-archive/final-model-cache/` using hashes of data, model source and sampler settings; unchanged rebuilds transparently reuse results. `results/final_project/fit_provenance.csv` records full-data cache keys, and `fit_manifest.csv` records precomputation diagnostics. `.venv\Scripts\python.exe scripts/fit_final_models.py` can precompute all fits. Independent numerical-density tests run with `.venv\Scripts\python.exe -m unittest discover -s tests -v`.

Gaussian sigma is an innovation SD; Student-t sigma is a scale with SD sigma*sqrt(5/3). The Student-t initial residual has a variance-matched scaled-t distribution, not an asserted exact stationary t-AR marginal. The notebook appendix includes numerical/data source listings for code-inclusive export. The earlier exploratory notebook remains separate.

The final deliverable will need a code-inclusive, 12-point PDF of at most 20 pages. Installing `nbconvert` does not install a TeX/browser PDF engine; PDF export and pagination will be configured when the report is complete. The proposal's separate renderer additionally needs `reportlab` and `pypdf`.
