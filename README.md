# Multivariate analysis of OFDR data and regression models for simultaneous estimation of temperature and hydrostatic pressure changes in K-SHF fibers


## Download and run the project:
1. Clone the repository:
   ```
    git clone https://github.com/szybec31/OFDR-ML-TempPressure.git
    cd OFDR-ML-TempPressure
   ```
2. Create a virtual environment:
    ```
    python -m venv myvenv
    myvenv\Scripts\activate       # Windows
    source myvenv/bin/activate    # Linux/Mac 
    ```
3. Install requirements:
   ```
   pip install -r requirements.txt
   ```
4. Create folders: Output_files and PAKA_AI with data

5. Prepare the data (first run only, or after any change in preprocessing):
   ```
   python main.py setup
   ```
   `setup` = `prepare` + `prepare_broken` + `info`. It writes `inventory.csv`, `training_dataset.csv`
   and `paired_features.csv` (plus `*_broken` versions with uncorrected 11 / 0.01 MPa labels) to `Output_files/`.

6. Run experiments. Every experiment is defined by a JSON file in `configs/`:
   ```
   python main.py run                                   # configs/main_temperature.json
   python main.py ablations                             # configs/ablations.json
   python main.py run_condition                         # configs/main_condition.json
   python main.py sem1                                  # configs/sem1_full.json (all models from the semester 1 report)
   python main.py experiment configs/my_experiment.json # any config
   ```
   Several commands can be chained, e.g. `python main.py setup run ablations`. `python main.py help` lists all options.

   Each run creates `Output_files/runs/<date>_<name>/` with **all** results of that run
   (`Output_files/runs/LATEST.txt` points to the newest one):
   - `config.json`, `config_resolved.json`, `run_meta.json` (git commit, dirty flag, library versions, data hash),
   - `folds.csv`, `predictions.csv` (test predictions and out-of-fold predictions for `zero_end`), `fold_status.*`,
   - `results_table.*` - main table, mean (std) over the common folds,
   - `summary_common_folds.*`, `summary_all_valid_folds.*`, `pooled_metrics.*` (pooled R^2),
   - `paired_tests.*` (Wilcoxon, paired t-test, Cohen's d vs baseline), `gain_vs_baseline.*` (gain G),
   - `repeatability_*` (signal drift zero_end - zero_start incl. common/differential parts, Residual Drift),
   - `physical_coefficients.*`, `split_diagnostics.*`, plots (`*.png`).
   Every table is saved as `.csv`, `.md` and `.tex`, so report tables can be pasted without retyping numbers.

   Conventions:
   - a model that cannot be calibrated on the training data of a fold (e.g. AN-BL without dT = 0)
     gets status `not_identifiable` - no hidden default coefficients,
   - the main comparison, statistical tests and G are computed on the **common folds**
     (folds where all models in the config are valid); `n_folds` is reported in every table,
   - per-fold R^2 is reported only where the target varies inside the fold; pooled R^2 is a separate table,
   - `zero_end` samples are never used for training (unless a config sets `include_zero_end_train`)
     and never enter the main test metrics.

7. Run jupyterlab (*optional)
   ```
   python -m jupyterlab
   ```

## Project files description:

Python files:
- `analyze_files.py` - reads raw OFDR files, computes ROI features per X/Y pair, checks channel swap
- `build_df.py` - builds `inventory.csv` from the directory structure (labels, pairs, references, role flags)
- `utils.py` - DT16 folder fix, file reader, ROI cleaning
- `main.py` - command line entry point (data preparation and experiments)
- `pipeline/` - experiment pipeline:
  - `runner.py` - config loading, run directory, metadata, variants
  - `cv.py` - outer leave-one-group-out loop and inner GridSearchCV (GroupKFold by series)
  - `splits.py` - split schemes: `temperature`, `series`, `pressure`, `condition`
  - `models.py` - model registry (builders + hyperparameter grids)
  - `metrics.py` - MAE, RMSE, MaxAE, bias, R^2
  - `report.py` - all tables, statistical tests, gain G, repeatability, plots
- `baselines/models/` - model definitions (`physical_model.py` = analytical model from the paper)
- `configs/` - experiment definitions

Notebooks:
- `01_preprocessing.ipynb` - EDA
- `DisplayDF.ipynb` - displays the inventory table
- `ML_results.ipynb` - per-temperature analysis of the latest run's predictions

Directories:
- `PAKA_AI` - folder and file structure with measurement data
- `Output_files` - processed data files and experiment runs
- `other_output` - channel and pressure mapping files

Other:
- `requirements.txt` - contains required libraries
- `.gitignore` - prevents selected data from being uploaded to GitHub

## Authors: