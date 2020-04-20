# Lumped Road Erosion Model
**Updated:** 04/19/2020

## Summary
This repository contains code for a lumped model used to estimate the erosion of forest roads.

## Repository navigation
### `rlm_data_scripts`
This folder contains three scripts used to format and plot input data for scripts in `rlm_model_runs`.

1. `groupSeasons_script.py` is a script used to group rainfall data by season.
   - **Input:** Raw rainfall data in a Pandas-readable format. 
   - **Output:** Pandas dataframe of rainfall data grouped by season.
2. `groupStorms_script.py` is a script used to group rainfall data by storm.
   - **Input:** Raw rainfall data in a Pandas-readable format and the minimum threshold of time between storms.
   - **Output:** Pandas dataframe of rainfall data grouped by storm.
3. `hyetographPlots_script.py` is a script used to plot rainfall hyetographs.
   - **Input:** Raw rainfall data in a Pandas-readable format and output from `groupStorms_script.py` or `groupSeasons_script.py`.
   - **Output:** Rainfall hyetograph plots of raw rainfall data and grouped rainfall data.

### `rlm_model_runs`
This folder contains two scripts used to run the lumped road erosion model.

1. `generatedRainfall_run.py` estimates the erosion of forest roads using stochastically generated rainfall.
   - **Input:** Model run time,  average inter-storm duration for location in hours, average storm duration for location in hours, and average rainfall intensity for location in millimeters per hour.
   - **Output:** Plots of road layer depths vs. time and erosion vs. time.
2. `nehalemRainfall_run.py` estimates the erosion of forest roads using 15-minute rainfall data from Nehalem, OR.
   - **Input:** Output from `groupStorms_script.py` or `groupSeasons_script.py`.
   - **Output:** Plots of road layer depths vs. time and erosion vs. time.
