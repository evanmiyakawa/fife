"""Import"""
import time

from fife.lgb_modelers import LGBSurvivalModeler
# from fife.tf_modelers import TFSurvivalModeler
# from fife.tf_modelers_copy import *
from fife.tf_modelers import *
from fife.processors import PanelDataProcessor
from fife.utils import make_results_reproducible
from pandas import concat, date_range, read_csv, to_datetime
import pandas as pd
import numpy as np
from ppprint import ppprint
# from tests_performance.Data_Fabrication import *
from tests_performance.Data_Fab_Copy import *




##### countries data #######
### Set up basic data, subset of original ###

# # dat = read_csv("Evan/REIGN_2020_7.csv")
# data = read_csv("Evan/REIGN_2020_7.csv")
#
# # countries_unique = np.unique(dat["country"])
# #
# # countries_sub = countries_unique[0:200] # all
# # # countries_sub = countries_unique[0:100]
# #
# # # multiple ways to do filtering
# # # data.query("country in countries_sub")
# # data = dat[dat["country"].isin(countries_sub)]
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

data = fabricate_data(N_PERSONS=1000, N_PERIODS=10, SEED=1234, exit_prob=0.3, dgp=1,
                      covariates_affect_outcome = True)

# data = fabricate_data(N_PERSONS=500, N_PERIODS=10, SEED=1234, exit_prob=0.2, dgp=1,
#                       covariates_affect = True)




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


params_relu = modeler_TF.hyperoptimize(hidden_activation = "relu")
# params_relu = modeler_TF.hyperoptimize(hidden_activation = "relu")


# params_relu = modeler_TF.hyperoptimize(n_trials = 10, max_epochs = 50,
#                                   hidden_activation = "relu")
params_relu = {'BATCH_SIZE': 95, 'DENSE_LAYERS': 1, 'DROPOUT_SHARE': 0.18594560795318357, 'EMBED_EXPONENT': 0.22823967220460073, 'EMBED_L2_REG': 0.9910799918860387, 'POST_FREEZE_EPOCHS': 49, 'PRE_FREEZE_EPOCHS': 8, 'NODES_PER_DENSE_LAYER': 816,
               "HIDDEN_ACTIVATION": "relu"}
params_relu = {'BATCH_SIZE': 1286, 'DENSE_LAYERS': 3, 'DROPOUT_SHARE': 0.4322057388929743, 'EMBED_EXPONENT': 0.1673694019994775, 'EMBED_L2_REG': 0.0007167166706998759, 'POST_FREEZE_EPOCHS': 50, 'PRE_FREEZE_EPOCHS': 73, 'NODES_PER_DENSE_LAYER': 195,
               "HIDDEN_ACTIVATION": "relu"}


# params_sigmoid = modeler_TF.hyperoptimize(hidden_activation = "sigmoid")
# params_sigmoid = modeler_TF.hyperoptimize(n_trials = 10, max_epochs = 50,
#                                   hidden_activation = "sigmoid")
params_sigmoid = {'BATCH_SIZE': 54, 'DENSE_LAYERS': 1, 'DROPOUT_SHARE': 0.12708007679200212, 'EMBED_EXPONENT': 0.14735536034349736, 'EMBED_L2_REG': 7.496258769837922, 'POST_FREEZE_EPOCHS': 117, 'PRE_FREEZE_EPOCHS': 72, 'NODES_PER_DENSE_LAYER': 571,
                  "HIDDEN_ACTIVATION": "sigmoid"}



# params_sigmoid = {'BATCH_SIZE': 32, 'DENSE_LAYERS': 3, 'DROPOUT_SHARE': 0.14194098083262702, 'EMBED_EXPONENT': 0.0010944939671413273, 'EMBED_L2_REG': 15.991527297711723, 'POST_FREEZE_EPOCHS': 45, 'PRE_FREEZE_EPOCHS': 50, 'NODES_PER_DENSE_LAYER': 1007,
#                "HIDDEN_ACTIVATION": "sigmoid"}
# params_sigmoid = {'BATCH_SIZE': 32, 'DENSE_LAYERS': 1, 'DROPOUT_SHARE': 0.14194098083262702, 'EMBED_EXPONENT': 0.0010944939671413273, 'EMBED_L2_REG': 15.991527297711723, 'POST_FREEZE_EPOCHS': 45, 'PRE_FREEZE_EPOCHS': 36, 'NODES_PER_DENSE_LAYER': 707,
#                "HIDDEN_ACTIVATION": "sigmoid"}






# modeler_TF.build_model(params = params_relu)
# modeler_TF.build_model(params = params_sigmoid)

modeler_TF.model.summary()

# subset = pd.Series(data = np.repeat(True, len(modeler_TF.data)))
# subset[500:len(modeler_TF.data)] = False

modeler_TF.build_model(params = params_relu) # loss = 1.1866, val_loss = 1.1043
df_relu = modeler_TF.compute_model_uncertainty(
    n_iterations = 20,
    params = params_relu,
    # dropout_rate = 0.1,
    percent_confidence = 0.95
)

modeler_TF.model.summary()

modeler_TF.build_model(params = params_sigmoid)
df_sigmoid = modeler_TF.compute_model_uncertainty(
    n_iterations = 20,
    params = params_sigmoid,
    # dropout_rate = 0.1,
    percent_confidence = 0.95
)

modeler_TF.build_model(params = params_relu)
df_relu_02 = modeler_TF.compute_model_uncertainty(
    n_iterations = 20,
    params = params_relu,
    # dropout_rate = 0.02,
    percent_confidence = 0.95
)

df_relu_02


df_sigmoid

df.ID[8]

modeler_TF.plot_forecast_prediction_intervals(df_relu, ID = "Sum")
modeler_TF.plot_forecast_prediction_intervals(df_sigmoid, ID = "Sum")
modeler_TF.plot_forecast_prediction_intervals(df_relu_02, ID = "Sum")

id = 658
modeler_TF.plot_forecast_prediction_intervals(df_relu, ID = id)
modeler_TF.plot_forecast_prediction_intervals(df_sigmoid, ID = id)
modeler_TF.plot_forecast_prediction_intervals(df_relu_50, ID = id)

print(*df_relu['ID'])

self = modeler_TF