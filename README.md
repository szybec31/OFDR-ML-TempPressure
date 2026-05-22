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

5. Run Project:
- When you first time run main.py script use setup arg:

  ```
  python main.py setup
  ```
- Using script you may also use other arg: `prepare`, `prepare_broken`, `run`, `ablations`, `info`, `setup`, or shorter using first letter (`p`, `pb`, `r`, `a`, `i`, `s`)
- Option `setup` is equivalent of `prepare`, `prepare_broken` and `info`.
- Also you may use more than one option, for example:
  ```
  python main.py prepare prepare_broken run info
  ```
  Them all option all executed in a row.

6. Run jupyterlab (*optional)
   ```
   python -m jupyterlab
   ```

## Project files description:

Python files:
- `analyze_files.py` 
  - builds physical signals ΔX and ΔY 
  - checks how they change with pressure 
  - detects whether the channels behave in a physically correct way
- `build_df.py`
  - loads and extracts data from the directory structure  
  - based on this data and instructions, builds inventory.csv
- `utils.py`
  - fixes incorrect data structure (DT16)
  - loads raw OFDR files 
  - extracts and cleans ROI 
  - computes the physical signal: ΔShift (i.e., the effect of pressure/temperature)
- `main.py` - wdrożenie wszystkich funkcji razem 
  - fixes directory structure and naming 
  - initializes the creation of inventory.csv 
  - determines ROI and cleans points within ROI 
  - aggregates data to the sample level 
  - creates the dataset for ML analysis paired_features.csv
- `physical_model.py` - implementation of the physical model from the paper

Notebooks:
- `DisplayDF.ipynb` - displays the inventory table
- `analyze.ipynb` - displays the paired_features table

Directories:
- `PAKA_AI` - folder and file structure with measurement data
- `Output_files` - processed data files
- `other_output` - channel and pressure mapping files

Other:
- `requirements.txt` - contains required libraries
- `.gitignore` - prevents selected data from being uploaded to GitHub

## Authors: