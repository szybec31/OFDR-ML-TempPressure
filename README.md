# Multivariate analysis of OFDR data and regression models for simultaneous estimation of temperature and hydrostatic pressure changes in K-SHF fibers


New to the project? Start with the guide (in Polish): [docs/przewodnik_po_projekcie.md](docs/przewodnik_po_projekcie.md).

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
   python main.py run                                   # configs/main_condition.json (main: leave-one-condition-out)
   python main.py run_temp                              # configs/main_temperature.json (leave-one-temperature-out)
   python main.py ablations                             # configs/ablations.json
   python main.py run_series                            # configs/leave_series_out.json (leave-one-series-out)
   python main.py run_pressure                          # configs/leave_pressure_out.json (leave-one-pressure-level-out)
   python main.py run_block                             # configs/block_holdout.json (held-out regions of the T, p space)
   python main.py run_reduced                           # configs/reduced_calibration.json (fewer calibration states, task 6)
   python main.py run_spatial                           # configs/spatial_windows.json (spatial windows along the ROI, task 7)
   python main.py run_roi                               # configs/roi_windows.json (ROI width: 1-100 cm windows as training samples)
   python main.py experiment configs/analytical_dT_vs_Tp.json  # analytical models, dT vs Tp, 4 validation schemes (~30 s)
   python main.py experiment configs/my_experiment.json # any config
   ```
   Several commands can be chained, e.g. `python main.py setup run ablations`. `python main.py help` lists all options.

   Each run creates `Output_files/runs/<date>_<name>/` with **all** results of that run
   (`Output_files/runs/LATEST.txt` points to the newest one):
   - `config.json`, `config_resolved.json`, `run_meta.json` (git commit, dirty flag, library versions, data hash),
   - `results_per_fold.*` - results of every model in every fold,
   - `folds.csv`, `predictions.csv` (test predictions and out-of-fold predictions for `zero_end`), `fold_status.*`,
   - `results_table.*` - main table, mean (std) over the common folds,
   - `summary_common_folds.*`, `summary_all_valid_folds.*`, `pooled_metrics.*` (pooled R^2),
   - `paired_tests.*` (Wilcoxon, paired t-test, Cohen's d vs baseline), `gain_vs_baseline.*` (gain G),
   - `repeatability_*` (signal drift zero_end - zero_start incl. common/differential parts, Residual Drift),
   - `physical_coefficients.*`, `split_diagnostics.*`, plots (`*.png`).
   Every table is saved as `.csv`, `.md` and `.tex`, so report tables can be pasted without retyping numbers.

   `run_reduced` (`"experiment": "reduced_calibration"`) writes instead: `curve.*` (RMSE vs number of
   calibration states for every model and selection method, mean/std over random repeats), `min_states.*`
   (minimal number of states with RMSE worse by at most 5/10/20 % than full calibration), `designs.csv`,
   `results_raw.csv`, `rmse_vs_states_<model>.png`, `models_<method>.png`, `designs_examples.png`.

   `run_spatial` (`"experiment": "spatial"`) writes: `rmse_vs_L.*` and `rmse_vs_L*.png` (RMSE vs window length,
   local vs global calibration), `min_window_length.*`, `window_rmse_profiles.csv`, `window_rmse_L*cm.png`,
   `outlier_windows.*`, `sensitivity_L*cm.*` and `sensitivity_homogeneity.*` (local sensitivities along the fiber).
   The first run reads all raw files once and caches 1 cm bins in `Output_files/spatial_bins_1cm.npz`.

   `run_roi` (key `"roi_window_cm"`, any CV config): the ROI is split into non-overlapping windows of length L
   (1, 2, 5, 10, 25, 50, 100 cm; `"full"` = whole ROI = `paired_features.csv`). Every window gives one row with the
   same features (mu, std, IQR, m, d), so each X/Y sample yields many training rows; all windows of a sample share
   its labels and groups, so they never end up on both sides of a split. Extra outputs: `results_table_sample_level.*`
   (prediction of a sample = mean over its windows, comparable with the full ROI), `variants_table_sample_level.*`
   and `rmse_vs_roi_window.png`; predictions are saved as `predictions.csv.gz`. Window features are cached in
   `Output_files/roi_windows/L<L>cm.npz` (built once, ~20 s). Optional: `"train_windows_per_sample": N` keeps
   N random windows per training sample (for slow models), `"roi_window_min_coverage"` (default 0.8),
   and the column `z_center_m` (window position) can be added to `features`.

   Temperature target (any config):
   ```
   "temperature_target": "dT",        // dT = deltaT_label (default) | Tp = T_plate_label
   "temperature_reference": "auto"    // T_ref for the physical models; auto: 0 for dT, Tp of the reference series (23) for Tp
   ```
   The chosen column is used everywhere: as the model target, for fold labels (`Tp=50`, `Tp=50_p=5.0`), for
   calibration states and in all output names (`Tp_rmse`, `Tp_pred`, ...), so every table shows which target was used.
   Physical models (AN-BL, AN-JOINT) compute in `T - T_ref`, because their formula has no intercept (signal = 0 in the
   reference state). With `Tp`, blocks are defined with `T_min` / `T_max` (in Tp units) and `exclude_test_groups`
   uses `Tp=...` labels; the old `dT_*` names are rejected with an error instead of being silently ignored.

   Conventions:
   - the main validation scheme is **leave-one-condition-out**: each fold tests all repetitions of one
     (dT, p) combination, the remaining 118 combinations are used for training; hyperparameters are tuned
     inside the training part with GroupKFold over conditions (`inner_cv.group = condition_id`).
     This is an interpolation test - the same temperature and pressure levels appear in training in other
     combinations. Generalisation to unseen temperatures / pressures / regions: `run_temp`, `run_series`,
     `run_pressure`, `run_block`,
   - a model that cannot be calibrated on the training data of a fold (e.g. AN-BL without dT = 0)
     gets status `not_identifiable` - no hidden default coefficients,
   - `"baseline"` sets the reference model (paired tests, gain G). `"common_folds"` chooses the folds of the
     main table: `"baseline"` (default) - folds where the reference model is valid, `"all"` - folds where every
     model is valid, or a list of model names. Example: with `"baseline": "AN-JOINT"` leave-one-temperature-out
     uses all 9/9 folds, and AN-BL (N/A for dT = 0) is shown with `n_folds = 8`. Paired tests and G are computed
     per pair, on the folds where both models are valid; `n_folds` is reported in every row,
   - per-fold R^2 is reported only where the target varies inside the fold; pooled R^2 is a separate table,
   - `zero_end` samples are never used for training (unless a config sets `include_zero_end_train`)
     and never enter the main test metrics,
   - `"model_params"` overrides model constructor parameters. For AN-BL / AN-BL-NEAREST it sets the calibration
     point: `calibration_temperature` (target units, `null` = dT = 0) for C and `calibration_pressure` (MPa,
     default 0) for A, B. AN-BL-NEAREST falls back to the nearest available temperature / pressure level when
     the calibration point is missing from the training fold,
   - every run saves `rmse_per_fold/<model>.png` - RMSE of pressure and temperature in each fold, one figure per model.

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
  - `splits.py` - split schemes: `temperature`, `series`, `pressure`, `condition` (leave-one-group-out) and `block` (held-out rectangles in the T, p space, defined in the config)
  - `models.py` - model registry (builders, hyperparameter grids, fixed feature sets, minimal number of states)
  - `calibration_design.py` - selection of calibration states (stratified random, grid, min/max/mid, maximin, D-optimal)
  - `reduced_calibration.py` - task 6 experiment: accuracy vs number of calibration states
  - `spatial.py` - task 7 experiment: spatial windows, local vs global calibration, local sensitivities
  - `metrics.py` - MAE, RMSE, MaxAE, bias, R^2
  - `report.py` - all tables, statistical tests, gain G, repeatability, plots
- `baselines/models/` - model definitions (`physical_model.py` = analytical model from the paper: AN-BL with
  sequential calibration and AN-JOINT with joint least-squares calibration on any set of (T, p) states)
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