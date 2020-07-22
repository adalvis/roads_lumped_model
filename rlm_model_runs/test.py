#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import NFtoutle_module as mod

data_df = pd.read_csv('./rlm_output/groupedStorms_ElkRock_10yr.csv', index_col='date')
data_df.index = pd.to_datetime(data_df.index)
