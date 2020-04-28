import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

int_tip_df = pd.read_csv('./rlm_output/int_tip_df.csv', index_col=0)

storm_tips = int_tip_df.tips[int_tip_df.stormNo==15.0]
storm_int = int_tip_df.intensity[int_tip_df.stormNo==15.0]