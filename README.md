# WADNR Roads Project Lumped Model
**Updated:** 06/23/2025

## Summary
This repository contains code for a spatially lumped model used to estimate the erosion of forest roads.

## Repository navigation
### `rlm_data_scripts`
This folder contains four scripts used to format and plot input data for scripts in `rlm_model_runs`.

1. `dailyTimestep.py` is a script that groups rainfall data by day.
   - **Input** Raw rainfall data in a Pandas-readable format.
   - **Output** Pandas dataframe of rainfall data grouped by day.
2. `Emmett1970Data_script.py` is a script used to plot data from two different experiments discussed in Emmett (1970).
   - **Input:** Flow (q) and Manning's n data for two experiments in a Pandas-readable format. 
   - **Output:** Clean plots showing the relationship between q and n for two scenarios.
3. `groupStorms_script.py` is a script used to group rainfall data by storm.
   - **Input:** Raw rainfall data in a Pandas-readable format and a user-defined minimum threshold of time between storms.
   - **Output:** Pandas dataframe of rainfall data grouped by storm.
4. `pullRainfall.py` is a script used to pull rainfall data from rain gages near field sites using the Mesonet API. Note that
the Mesonet API is now known as the Synoptic Data API and is not free to use. This script is kept here for completeness.
   - **Input:** Station ID.
   - **Output:** Raw rainfall data in a Pandas-readable format.

### `rlm_model_runs`
This folder contains one script used to run the lumped road erosion model and a Jupyter notebook tutorial. Additionally includes an `archive` folder with old pieces of code that are out-of-date but are kept for record-keeping. 

1. `rlm_daily_time_step.py` estimates the erosion of forest roads using rainfall data pulled from the Mesonet API and stochastically 
generated truck passes.
   - **Input:** Output from `dailyTimestep.py`.
   - **Output:** Plots of road layer depths vs. time and erosion vs. time.
2. `rlm_tutorial.ipynb` estimates the erosion of forest roads using rainfall data pulled from the Mesonet API and stochastically generated truck passes in a tutorial format.
   - **Input:** Output from `dailyTimestep.py`.
   - **Output:** Plots of road layer depths vs. time and erosion vs. time.