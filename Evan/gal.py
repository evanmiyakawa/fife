from fife.tf_modelers_copy import *
# from fife.tf_modelers import *

from fife.processors import PanelDataProcessor
# from fife.utils import make_results_reproducible
from pandas import concat, date_range, read_csv, to_datetime
import pandas as pd
import numpy as np
# from fife import tf_modelers
from fife.base_modelers import default_subset_to_all

# from fife.tf_modelers import *
# from fife.tf_modelers_copy import *

from typing import List, Union
import tensorflow.keras.backend as K
# from fife.tf_modelers import TFSurvivalModeler
from matplotlib import pyplot as plt
from tests_performance.Data_Fabrication import *
from tests_performance.Data_Fab_Copy import *




### Set up countries basic data, subset of original ###

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
#


### new data instead

data = fabricate_data(N_PERSONS=1000, N_PERIODS=20, SEED=1234, exit_prob=0.3, dgp=1)

data_processor = PanelDataProcessor(data=data)
data_processor.build_processed_data()
#print(data_processor.data.head())

print(data_processor.data)


### Create TF modeler ###


modeler_TF = TFSurvivalModeler(data = data_processor.data)

modeler_TF.n_intervals = modeler_TF.set_n_intervals()



# params = modeler_TF.hyperoptimize()

# params


# params = {'BATCH_SIZE': 118, 'DENSE_LAYERS': 1, 'DROPOUT_SHARE': 0.10930137242567001, 'EMBED_EXPONENT': 0.007865252225413372, 'EMBED_L2_REG': 8.73481866750174, 'POST_FREEZE_EPOCHS': 20, 'PRE_FREEZE_EPOCHS': 10, 'NODES_PER_DENSE_LAYER': 937}#

# params = {'BATCH_SIZE': 512, 'PRE_FREEZE_EPOCHS': 16, 'POST_FREEZE_EPOCHS': 16, 'DROPOUT_SHARE': 0.5}

# original
# params = {'BATCH_SIZE': 42, 'DENSE_LAYERS': 1, 'DROPOUT_SHARE': 0.05884135514717052, 'EMBED_EXPONENT': 0.06933413089910512, 'EMBED_L2_REG': 8.582852423132469, 'POST_FREEZE_EPOCHS': 99, 'PRE_FREEZE_EPOCHS': 6, 'NODES_PER_DENSE_LAYER': 963}

# edited
params = {'BATCH_SIZE': 42, 'DENSE_LAYERS': 1, 'DROPOUT_SHARE': 0.06, 'EMBED_EXPONENT': 0.06933413089910512, 'EMBED_L2_REG': 8.582852423132469, 'POST_FREEZE_EPOCHS': 25, 'PRE_FREEZE_EPOCHS': 6, 'NODES_PER_DENSE_LAYER': 963}
# params = {'BATCH_SIZE': 42, 'DENSE_LAYERS': 10, 'DROPOUT_SHARE': 0.99,  'POST_FREEZE_EPOCHS': 20, 'PRE_FREEZE_EPOCHS': 20, 'NODES_PER_DENSE_LAYER': 100}
# params = {'BATCH_SIZE': 42, 'DENSE_LAYERS': 1, 'DROPOUT_SHARE': 0.99,  'POST_FREEZE_EPOCHS': 5, 'PRE_FREEZE_EPOCHS': 5, 'NODES_PER_DENSE_LAYER': 10}

# modeler_TF.model = modeler_TF.construct_embedding_network(dense_layers = 1, nodes_per_dense_layer=10,
#                                        dropout_share = 0.99)

# modeler_TF.build_model(params = params)
# modeler_TF.build_model()
#
# modeler_TF.model.summary()


### Do one Dropout iteration ###

from random import randrange

n_iterations = 4
dropout_rate = 0.5
percent_confidence = 0.95
#
self = modeler_TF
subset = None
subset = pd.Series(data = np.repeat(True, len(modeler_TF.data)))
subset[3000:len(modeler_TF.data)] = False



def compute_model_uncertainty(
        self,
        subset: Union[None, pd.core.series.Series] = None,
        n_iterations: int = 50,
        params: dict = None,
        dropout_rate: float = None,
        percent_confidence: float = 0.95
) -> pd.DataFrame:
    """Predict with MC Dropout as proposed by Gal and Ghahramani (2015). Produces prediction intervals for future observations.
    This procedure must be used with a NN model that uses dropout.

    See https://arxiv.org/abs/1506.02142.

    Args:
        subset: A Boolean Series that is True for observations for which
            predictions will be produced. If None, default to all
            observations.
        n_iterations: Number of random dropout iterations to obtain
            predictions from.
        params: A dictionary containing custom model parameters.
        dropout_rate: A dropout rate to be used, if different than default or in params.
        percent_confidence: The percent confidence of the two-sided intervals
            defined by the computed prediction intervals.

    Returns:
        A pandas DataFrame of prediction intervals for each observation, for each future time point.
    """

    subset = default_subset_to_all(subset, self.data)

    alpha = (1 - percent_confidence) / 2

    conf = str(int(percent_confidence * 100))

    def one_dropout_prediction(self, params, DROPOUT_SHARE, subset) -> pd.core.frame.DataFrame:
        """Compute forecasts for one MC Dropout iteration."""

        dropout_model = TFSurvivalModeler(data = self.data)
        dropout_model.n_intervals = self.n_intervals


        dropout_model.config["SEED"] = randrange(1, 9999, 1)

        if params is None:
            params = dropout_model.config

        if DROPOUT_SHARE is not None:
            params["DROPOUT_SHARE"] = DROPOUT_SHARE

        if "DROPOUT_SHARE" not in params.keys():
            warn("Model must have a dropout rate specified. Specify one using dropout_rate argument",
                 UserWarning)
            return None

        dropout_model.build_model(params=params)

        forecasts = dropout_model.forecast()

        forecasts.columns = list(map(str, np.arange(1, len(forecasts.columns) + 1, 1)))

        ids_in_subset = dropout_model.data[subset]["ID"].unique()

        keep_rows = np.repeat(True, len(forecasts))

        for rw in range(len(forecasts)):
            if forecasts.index[rw] not in ids_in_subset:
                keep_rows[rw] = False

        forecasts = forecasts[keep_rows]


        return (forecasts)

    dropout_forecasts = list()

    # do MC dropout for n_iterations

    for i in range(n_iterations):
        one_forecast = one_dropout_prediction(self, params=params,
                                              DROPOUT_SHARE=dropout_rate,
                                              subset = subset)
        dropout_forecasts.append(one_forecast)

    ### get mean forecasts

    sum_forecasts = dropout_forecasts[0]

    for i in range(1, len(dropout_forecasts)):
        sum_forecasts = sum_forecasts + dropout_forecasts[i]

    mean_forecasts = sum_forecasts / len(dropout_forecasts)

    def get_forecast_variance() -> pd.DataFrame:
        """ Get variance across forecasts"""

        def get_forecast_variance_for_time(time=0):

            def get_variance_for_index_time(index, time=0):
                diff_from_mean_df = dropout_forecasts[index].iloc[:, [time]] - mean_forecasts.iloc[:, [time]]
                squared_error_df = diff_from_mean_df ** 2
                return squared_error_df

            squared_error_sum_df = get_variance_for_index_time(0, time=time)

            for i in range(1, len(dropout_forecasts)):
                squared_error_sum_df = squared_error_sum_df + get_variance_for_index_time(i, time=time)

            variance_df = squared_error_sum_df / len(dropout_forecasts)

            return variance_df

        variance_all_times_df = get_forecast_variance_for_time(time=0)

        for t in range(1, len(mean_forecasts.columns)):
            variance_all_times_df = pd.concat([variance_all_times_df, get_forecast_variance_for_time(time=t)],
                                              axis=1)

        return variance_all_times_df

    def aggregate_observation_dropout_forecasts(dropout_forecasts, rw=0) -> pd.DataFrame:
        """Get collection of forecasts from MC Dropout for one observation"""

        survival_probability = pd.Series(dtype="float32")
        period = np.empty(shape=0, dtype="int")
        iteration = np.empty(shape=0, dtype="int")

        # iter = 0
        for iter in range(len(dropout_forecasts)):
            observation_forecasts = dropout_forecasts[iter].iloc[rw]
            id = dropout_forecasts[iter].index[rw]
            survival_probability = survival_probability.append(observation_forecasts)

            keys = observation_forecasts.keys()
            period = np.append(period, observation_forecasts.keys().to_numpy(dtype="int"))
            iteration = np.append(iteration, np.repeat(iter, len(keys)))

        id = np.repeat(dropout_forecasts[0].index[rw], len(iteration))
        forecast_df = pd.DataFrame({"ID": id, "iter": iteration, "period": period,
                                    "survival_prob": survival_probability})

        return forecast_df

    def get_forecast_prediction_intervals(alpha=0.025) -> pd.DataFrame:
        """Get prediction intervals for all observations for all time points"""

        prediction_intervals_df = pd.DataFrame({
            "ID": np.empty(shape=0, dtype=type(dropout_forecasts[0].index[0])),
            "Period": np.empty(shape=0, dtype="str"),
            "Lower" + conf + "PercentBound": np.empty(shape=0, dtype="float"),
            "Median": np.empty(shape=0, dtype="float"),
            "Mean": np.empty(shape=0, dtype="float"),
            "Upper" + conf + "PercentBound": np.empty(shape=0, dtype="float")
        })

        # p = 1
        for rw in range(len(mean_forecasts.index)):

            forecast_df = aggregate_observation_dropout_forecasts(dropout_forecasts, rw)

            for p in range(len(forecast_df['period'].unique())):
                period = p + 1
                # mean_forecasts.iloc[rw, forecast_df['period'].unique() == p
                mean_forecast = mean_forecasts.iloc[rw, p]
                period_forecasts = forecast_df[forecast_df['period'] == period]['survival_prob']
                forecast_quantiles = period_forecasts.quantile([alpha, 0.5, 1 - alpha])
                df = pd.DataFrame(
                    {"ID": forecast_df["ID"][0],
                     "Period": [period],
                     # "Period": [forecast_df['period'].unique()[p]],
                     "Lower" + conf + "PercentBound": forecast_quantiles.iloc[0],
                     "Median": forecast_quantiles.iloc[1],
                     "Mean": mean_forecast,
                     "Upper" + conf + "PercentBound": forecast_quantiles.iloc[2]
                     }
                )
                prediction_intervals_df = prediction_intervals_df.append(df)

        return prediction_intervals_df

    prediction_intervals_df = get_forecast_prediction_intervals(alpha)

    def get_aggregation_prediction_intervals() -> pd.DataFrame:
        """Get prediction intervals for the expected counts for each time period."""

        aggregation_pi_df = pd.DataFrame({
            "ID": np.empty(shape=0, dtype=type(dropout_forecasts[0].index[0])),
            "Period": np.empty(shape=0, dtype="str"),
            "Lower" + conf + "PercentBound": np.empty(shape=0, dtype="float"),
            "Median": np.empty(shape=0, dtype="float"),
            "Mean": np.empty(shape=0, dtype="float"),
            "Upper" + conf + "PercentBound": np.empty(shape=0, dtype="float")
        })

        p = 0
        for p in range(len(dropout_forecasts[0].columns.unique())):
            period = p + 1
            dropout_expected_counts = []

            for d in dropout_forecasts:
                dropout_expected_counts.append(d[d.columns[p]].sum())

            forecast_sum_series = pd.Series(dropout_expected_counts)

            forecast_sum_quantiles = forecast_sum_series.quantile([alpha, 0.5, 1 - alpha])

            df = pd.DataFrame(
                {"ID": "Sum",
                 "Period": [period],
                 "Lower" + conf + "PercentBound": forecast_sum_quantiles.iloc[0],
                 "Median": forecast_sum_quantiles.iloc[1],
                 "Mean": forecast_sum_series.mean(),
                 "Upper" + conf + "PercentBound": forecast_sum_quantiles.iloc[2]
                 }
            )

            aggregation_pi_df = aggregation_pi_df.append(df)

        return aggregation_pi_df

    aggregation_prediction_intervals_df = get_aggregation_prediction_intervals()

    prediction_intervals_df = aggregation_prediction_intervals_df.append(prediction_intervals_df)

    prediction_intervals_df.index = np.arange(0, len(prediction_intervals_df.index))

    return prediction_intervals_df


# prediction_intervals_df = compute_model_uncertainty(modeler_TF, n_iterations = 50, params = params)
prediction_intervals_df = compute_model_uncertainty(modeler_TF, n_iterations = 5, params = params)






### plot forecasts ###

pi_df = prediction_intervals_df
pi_df = df
ID = "Sum"
ID = "5"


def plot_forecast_prediction_intervals(self, pi_df: pd.DataFrame, ID: str = "Sum") -> None:
    """Plot forecasts for future time periods with prediction intervals calculated by MC Dropout using compute_model_uncertainty()

    Args:
        pi_df: A DataFrame of prediction intervals for forecasts, generated by compute_model_uncertainty()
        ID: The ID of the observation for which to have forecasts plotted, or "Sum" for aggregated counts.
    """

    ID = str(ID)
    forecast_df = pi_df[pi_df['ID'] == ID]

    if len(forecast_df) == 0:
        try:
            ID = int(ID)
            forecast_df = pi_df[pi_df['ID'] == int(ID)]
        except ValueError:
            ID = ID
            print("Invalid ID.")
            return None

    forecast_df = forecast_df.assign(Period=forecast_df['Period'].tolist())

    forecast_df = forecast_df.rename({forecast_df.columns[2]: "Lower", forecast_df.columns[5]: "Upper"}, axis="columns")

    plt.clf()

    plt.scatter('Period', 'Mean', data=forecast_df, color="black")

    plt.fill_between(x='Period', y1='Lower', y2='Upper',
                     data=forecast_df, color='gray', alpha=0.3,
                     linewidth=0)

    plt.xlabel("Period")

    if ID == "Sum":
        plt.ylabel("Expected counts")
    else:
        plt.ylabel("Survival probability")

    plt.title("Forecasts with Prediction Intervals")

    plt.show()

    return None


plot_forecast_prediction_intervals(prediction_intervals_df, ID = "Sikk")