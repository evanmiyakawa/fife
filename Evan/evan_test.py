"""Import"""

from fife.lgb_modelers import LGBSurvivalModeler
# from fife.tf_modelers import TFSurvivalModeler
from fife.tf_modelers_copy import *
from fife.processors import PanelDataProcessor
from fife.utils import make_results_reproducible
from pandas import concat, date_range, read_csv, to_datetime
import pandas as pd
import numpy as np
from ppprint import ppprint
from tests_performance.Data_Fabrication import *




##### countries data #######
### Set up basic data, subset of original ###

# dat = read_csv("Evan/REIGN_2020_7.csv")
# # data = read_csv("Evan/REIGN_2020_7.csv")
#
# countries_unique = np.unique(dat["country"])
#
# countries_sub = countries_unique[0:10]
#
# # multiple ways to do filtering
# # data.query("country in countries_sub")
# data = dat[dat["country"].isin(countries_sub)]
#
# #data = dat[1:100]
#
#
# data["country-leader"] = data["country"] + ": " + data["leader"]
# data["year-month"] = data["year"].astype(int).astype(str) + data["month"].astype(int).astype(str).str.zfill(2)
# data["year-month"] = to_datetime(data["year-month"], format="%Y%m")
# data = concat([data[["country-leader", "year-month"]],
#                data.drop(["ccode", "country-leader", "leader", "year-month"],
#                          axis=1)],
#                axis=1)
#
#
#
# total_obs = len(data)
# data = data.drop_duplicates(["country-leader", "year-month"], keep="first")
# n_duplicates = total_obs - len(data)



##### fife data gen ######

data = fabricate_data(N_PERSONS=1000, N_PERIODS=20, SEED=1234, exit_prob=0.3, dgp=1)




### process data ####

data_processor = PanelDataProcessor(data=data)
data_processor.build_processed_data()
#print(data_processor.data.head())

print(data_processor.data)

### LGB modeler ###
# modeler_LGB = LGBSurvivalModeler(data=data_processor.data);
# modeler_LGB.build_model(parallelize=False);
# forecasts_LGB = modeler_LGB.forecast()
# forecasts_LGB["50-period Survival Probability"]



### TF modeler ###


modeler_TF = TFSurvivalModeler(data = data_processor.data)

modeler_TF.n_intervals = modeler_TF.set_n_intervals()

# params = modeler_TF.hyperoptimize()

params = {'BATCH_SIZE': 42, 'DENSE_LAYERS': 1, 'DROPOUT_SHARE': 0.06, 'EMBED_EXPONENT': 0.06933413089910512, 'EMBED_L2_REG': 8.582852423132469, 'POST_FREEZE_EPOCHS': 25, 'PRE_FREEZE_EPOCHS': 6, 'NODES_PER_DENSE_LAYER': 963}


modeler_TF.build_model(params = params)

subset = pd.Series(data = np.repeat(True, len(modeler_TF.data)))
subset[500:len(modeler_TF.data)] = False


df = modeler_TF.compute_model_uncertainty(
    n_iterations = 5,
    params = params,
    percent_confidence = 0.99
)


df

modeler_TF.plot_forecast_prediction_intervals(df)


self = modeler_TF