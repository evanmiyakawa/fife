
from fife.base_modelers_copy import default_subset_to_all
from fife.utils import sigmoid
from fife.utils import compute_aggregation_uncertainty
# from fife.lgb_modelers_copy import LGBSurvivalModeler
from fife.lgb_modelers import LGBSurvivalModeler
# from fife.base_modelers import default_subset_to_all
from fife.tf_modelers import TFSurvivalModeler
# from fife.tf_modelers_copy import *
from fife.processors import PanelDataProcessor
from scipy.stats import norm
from matplotlib import pyplot as plt
import lightgbm as lgb

from fife.utils import make_results_reproducible
from pandas import concat, date_range, read_csv, to_datetime
import pandas as pd
import numpy as np
from random import randrange
from ppprint import ppprint
# from tests_performance.Data_Fabrication import *
from tests_performance.Data_Fab_Copy import *

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn import preprocessing


# SEED = 1235
# N_PERSONS = 1000
# N_PERIODS = 12
# exit_prob = 0.3
# ensemble_size = 20

SEED = 1235
N_PERSONS = 2000
N_PERIODS = 50
exit_prob = 0.4
ensemble_size = 20

# SEED = 1235
# N_PERSONS = 20000
# N_PERIODS = 250
# exit_prob = 0.2
# ensemble_size = 20

# ensemble_size = 5


if N_PERSONS > 5000:
    n = 0
    print("n: ", n)
    data = fabricate_data(
        N_PERSONS=5000,
        N_PERIODS=N_PERIODS,
        exit_prob=exit_prob,
        SEED=SEED,
        dgp=1,
        covariates_affect_outcome=True
    )

    n = 5000

    while n < N_PERSONS:
        print("n: ", n)
        data1 = fabricate_data(
            N_PERSONS=np.min([5000, n]),
            N_PERIODS=N_PERIODS,
            exit_prob=exit_prob,
            SEED=int(SEED + np.round(n / 5000)),
            dgp=1,
            covariates_affect_outcome=True
        )
        data1['ID'] = data1['ID'] + n

        n += 5000


        data = pd.concat([data, data1])

else:
    data = fabricate_data(
        N_PERSONS=N_PERSONS,
        N_PERIODS=N_PERIODS,
        exit_prob=exit_prob,
        SEED=SEED,
        dgp=1,
        covariates_affect_outcome=True
    )







data = data.drop("exit_type", axis = 1)
train_data = data[data['period'] < np.round(N_PERIODS * (2/3))]
train_data = train_data.drop("exit_prob", axis = 1)
test_data = data[data['period'] >= np.round(N_PERIODS * (2/3))]



# data = read_csv("https://www.dl.dropboxusercontent.com/s/3tdswu2jfgwp4xw/REIGN_2020_7.csv?dl=0")


# data = read_csv("Evan/REIGN_2020_7.csv")
#
# data["country-leader"] = data["country"] + ": " + data["leader"]
# data["year-month"] = data["year"].astype(int).astype(str) + data["month"].astype(int).astype(str).str.zfill(2)
# data["year-month"] = to_datetime(data["year-month"], format="%Y%m")
# data = concat([data[["country-leader", "year-month"]],
#                data.drop(["ccode", "country-leader", "leader", "year-month"],
#                          axis=1)],
#               axis=1)
# total_obs = len(data)
# data = data.drop_duplicates(["country-leader", "year-month"], keep="first")
#
#
# unique_months = data['year-month'].unique()
# len(unique_months)
# cutoff_time = unique_months[int(np.round(len(unique_months) * 0.8))]
#
#
# train_data = data[data['year-month'] < cutoff_time]
# test_data = data[data['year-month'] >= cutoff_time]



### KNN ###

data_processor = PanelDataProcessor(data=train_data)
data_processor.build_processed_data()
data_processor.data.columns

data_processor.is_categorical("period")

knn_data = train_data

knn_data = knn_data.drop(["ID", "period", "_predict_obs", "_test", "_validation", "_maximum_lead", "_spell", "_period", "_event_observed"], axis = 1)

knn_data["_duration"] = knn_data["_duration"].astype("float")

K = 2

def knn(knn_data, K: int = 2):
    # knn_data = knn_data.drop("exit_prob", axis = 1)

    knn_data_processor = PanelDataProcessor(data=knn_data)
    knn_data_processor.build_processed_data()

    knn_data = knn_data_processor.data

    knn_data = knn_data.drop(
        ["ID", "period", "_predict_obs", "_test", "_validation", "_maximum_lead", "_spell", "_period",
         "_event_observed"], axis=1)

    cols = knn_data.columns
    cols

    le = preprocessing.LabelEncoder()


    for c in cols:
        if knn_data_processor.is_categorical(c):
            knn_data[c] = le.fit_transform(knn_data[c])



    knn_data

    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(knn_data)
    distances, indices = nbrs.kneighbors(knn_data)
    print(distances)
    print(indices)

    distances[0]
    indices[0]
    len(knn_data)
    len(distances)


    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    weather_encoded = le.fit_transform(knn_data["X1"])
    print(weather_encoded)

