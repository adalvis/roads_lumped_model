# Import libraries
import pandas as pd

test = pd.read_pickle('volcanic_data.pkl')

pnnl_precip = test['PREC_ACC_NC_hourlywrf_pnnl']
