"""Import"""
# import time
import random

import pandas

from fife.base_modelers_copy import default_subset_to_all
from fife.utils import sigmoid
from fife.lgb_modelers_copy import LGBSurvivalModeler
# from fife.base_modelers import default_subset_to_all
# from fife.tf_modelers import TFSurvivalModeler
# from fife.tf_modelers_copy import *
from fife.processors import PanelDataProcessor
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
modeler_LGB = LGBSurvivalModeler(data=data_processor.data);

modeler_LGB.n_intervals = modeler_LGB.set_n_intervals()

# params = modeler_LGB.hyperoptimize()
# params = {0: {'num_iterations': 121, 'learning_rate': 0.315431867492976, 'num_leaves': 145, 'max_depth': 28, 'min_data_in_leaf': 85, 'min_sum_hessian_in_leaf': 0.24928461701833993, 'bagging_freq': 1, 'bagging_fraction': 0.6184868711057736, 'feature_fraction': 0.5379154203994021, 'lambda_l1': 0.2001363366987305, 'lambda_l2': 8.845983711876956, 'min_gain_to_split': 0.16969339573574543, 'min_data_per_group': 36, 'max_cat_threshold': 189, 'cat_l2': 25.78609617115775, 'cat_smooth': 2025.471439897878, 'max_cat_to_onehot': 64, 'max_bin': 44, 'min_data_in_bin': 64, 'objective': 'binary', 'num_class': 1}, 1: {'num_iterations': 100, 'learning_rate': 0.332960841612008, 'num_leaves': 8, 'max_depth': 21, 'min_data_in_leaf': 4, 'min_sum_hessian_in_leaf': 0.12051794591244501, 'bagging_freq': 1, 'bagging_fraction': 0.8255845560191495, 'feature_fraction': 0.8145705550858197, 'lambda_l1': 2.2546316785675855, 'lambda_l2': 2.198229043070872, 'min_gain_to_split': 0.020356864200784766, 'min_data_per_group': 162, 'max_cat_threshold': 3, 'cat_l2': 43.52449503203081, 'cat_smooth': 70.19587073090126, 'max_cat_to_onehot': 6, 'max_bin': 589, 'min_data_in_bin': 9, 'objective': 'binary', 'num_class': 1}, 2: {'num_iterations': 121, 'learning_rate': 0.315431867492976, 'num_leaves': 145, 'max_depth': 28, 'min_data_in_leaf': 85, 'min_sum_hessian_in_leaf': 0.24928461701833993, 'bagging_freq': 1, 'bagging_fraction': 0.6184868711057736, 'feature_fraction': 0.5379154203994021, 'lambda_l1': 0.2001363366987305, 'lambda_l2': 8.845983711876956, 'min_gain_to_split': 0.16969339573574543, 'min_data_per_group': 36, 'max_cat_threshold': 189, 'cat_l2': 25.78609617115775, 'cat_smooth': 2025.471439897878, 'max_cat_to_onehot': 64, 'max_bin': 44, 'min_data_in_bin': 64, 'objective': 'binary', 'num_class': 1}, 3: {'num_iterations': 126, 'learning_rate': 0.34255297606299584, 'num_leaves': 86, 'max_depth': 25, 'min_data_in_leaf': 4, 'min_sum_hessian_in_leaf': 0.24382512508056955, 'bagging_freq': 1, 'bagging_fraction': 0.5857959433133142, 'feature_fraction': 0.5025183074656011, 'lambda_l1': 0.669827471638558, 'lambda_l2': 16.963526856767526, 'min_gain_to_split': 0.17638809195202054, 'min_data_per_group': 496, 'max_cat_threshold': 167, 'cat_l2': 18.406545377325795, 'cat_smooth': 2041.474685621533, 'max_cat_to_onehot': 61, 'max_bin': 193, 'min_data_in_bin': 1, 'objective': 'binary', 'num_class': 1}, 4: {'objective': 'binary', 'num_iterations': 100, 'num_class': 1}, 5: {'num_iterations': 84, 'learning_rate': 0.23314152857487802, 'num_leaves': 200, 'max_depth': 13, 'min_data_in_leaf': 4, 'min_sum_hessian_in_leaf': 0.18899455685096556, 'bagging_freq': 1, 'bagging_fraction': 0.7683150962270708, 'feature_fraction': 0.880037145001022, 'lambda_l1': 0.07005876608070916, 'lambda_l2': 21.119875341594504, 'min_gain_to_split': 0.0402559203033382, 'min_data_per_group': 362, 'max_cat_threshold': 31, 'cat_l2': 43.780154171782165, 'cat_smooth': 826.0529599630217, 'max_cat_to_onehot': 35, 'max_bin': 900, 'min_data_in_bin': 44, 'objective': 'binary', 'num_class': 1}}
# params = {0: {'num_iterations': 121, 'learning_rate': 0.315431867492976, 'num_leaves': 145, 'max_depth': 28, 'min_data_in_leaf': 85, 'min_sum_hessian_in_leaf': 0.24928461701833993, 'bagging_freq': 1, 'bagging_fraction': 0.6184868711057736, 'feature_fraction': 0.5379154203994021, 'lambda_l1': 0.2001363366987305, 'lambda_l2': 8.845983711876956, 'min_gain_to_split': 0.16969339573574543, 'min_data_per_group': 36, 'max_cat_threshold': 189, 'cat_l2': 25.78609617115775, 'cat_smooth': 2025.471439897878, 'max_cat_to_onehot': 64, 'max_bin': 44, 'min_data_in_bin': 64, 'objective': 'binary'}, 1: {'num_iterations': 100, 'learning_rate': 0.332960841612008, 'num_leaves': 8, 'max_depth': 21, 'min_data_in_leaf': 4, 'min_sum_hessian_in_leaf': 0.12051794591244501, 'bagging_freq': 1, 'bagging_fraction': 0.8255845560191495, 'feature_fraction': 0.8145705550858197, 'lambda_l1': 2.2546316785675855, 'lambda_l2': 2.198229043070872, 'min_gain_to_split': 0.020356864200784766, 'min_data_per_group': 162, 'max_cat_threshold': 3, 'cat_l2': 43.52449503203081, 'cat_smooth': 70.19587073090126, 'max_cat_to_onehot': 6, 'max_bin': 589, 'min_data_in_bin': 9, 'objective': 'binary'}, 2: {'num_iterations': 121, 'learning_rate': 0.315431867492976, 'num_leaves': 145, 'max_depth': 28, 'min_data_in_leaf': 85, 'min_sum_hessian_in_leaf': 0.24928461701833993, 'bagging_freq': 1, 'bagging_fraction': 0.6184868711057736, 'feature_fraction': 0.5379154203994021, 'lambda_l1': 0.2001363366987305, 'lambda_l2': 8.845983711876956, 'min_gain_to_split': 0.16969339573574543, 'min_data_per_group': 36, 'max_cat_threshold': 189, 'cat_l2': 25.78609617115775, 'cat_smooth': 2025.471439897878, 'max_cat_to_onehot': 64, 'max_bin': 44, 'min_data_in_bin': 64, 'objective': 'binary'}, 3: {'num_iterations': 126, 'learning_rate': 0.34255297606299584, 'num_leaves': 86, 'max_depth': 25, 'min_data_in_leaf': 4, 'min_sum_hessian_in_leaf': 0.24382512508056955, 'bagging_freq': 1, 'bagging_fraction': 0.5857959433133142, 'feature_fraction': 0.5025183074656011, 'lambda_l1': 0.669827471638558, 'lambda_l2': 16.963526856767526, 'min_gain_to_split': 0.17638809195202054, 'min_data_per_group': 496, 'max_cat_threshold': 167, 'cat_l2': 18.406545377325795, 'cat_smooth': 2041.474685621533, 'max_cat_to_onehot': 61, 'max_bin': 193, 'min_data_in_bin': 1, 'objective': 'binary'}, 4: {'objective': 'binary', 'num_iterations': 100, 'num_class': 1}, 5: {'num_iterations': 84, 'learning_rate': 0.23314152857487802, 'num_leaves': 200, 'max_depth': 13, 'min_data_in_leaf': 4, 'min_sum_hessian_in_leaf': 0.18899455685096556, 'bagging_freq': 1, 'bagging_fraction': 0.7683150962270708, 'feature_fraction': 0.880037145001022, 'lambda_l1': 0.07005876608070916, 'lambda_l2': 21.119875341594504, 'min_gain_to_split': 0.0402559203033382, 'min_data_per_group': 362, 'max_cat_threshold': 31, 'cat_l2': 43.780154171782165, 'cat_smooth': 826.0529599630217, 'max_cat_to_onehot': 35, 'max_bin': 900, 'min_data_in_bin': 44, 'objective': 'binary'}}
params = {0: {'num_iterations': 100, 'learning_rate': 0.4271681166673791, 'num_leaves': 54, 'max_depth': 20, 'min_data_in_leaf': 4, 'min_sum_hessian_in_leaf': 0.0752411162807661, 'bagging_freq': 0, 'bagging_fraction': 0.5556632063754483, 'feature_fraction': 0.7894963315381907, 'lambda_l1': 0.06952219198723059, 'lambda_l2': 9.405474841875103, 'min_gain_to_split': 0.049673449128315755, 'min_data_per_group': 510, 'max_cat_threshold': 9, 'cat_l2': 53.59245703942317, 'cat_smooth': 1595.3103874876606, 'max_cat_to_onehot': 20, 'max_bin': 927, 'min_data_in_bin': 2, 'objective': 'binary'}, 1: {'num_iterations': 52, 'learning_rate': 0.31603944758351377, 'num_leaves': 81, 'max_depth': 28, 'min_data_in_leaf': 5, 'min_sum_hessian_in_leaf': 0.22624934416508022, 'bagging_freq': 1, 'bagging_fraction': 0.658860498281767, 'feature_fraction': 0.8857957658480419, 'lambda_l1': 1.8916812366387934, 'lambda_l2': 15.251416204468983, 'min_gain_to_split': 0.107741635087494, 'min_data_per_group': 243, 'max_cat_threshold': 130, 'cat_l2': 27.366350807930875, 'cat_smooth': 1583.2298107503727, 'max_cat_to_onehot': 50, 'max_bin': 656, 'min_data_in_bin': 16, 'objective': 'binary'}, 2: {'num_iterations': 121, 'learning_rate': 0.3262265892752121, 'num_leaves': 130, 'max_depth': 30, 'min_data_in_leaf': 46, 'min_sum_hessian_in_leaf': 0.24876600558274548, 'bagging_freq': 1, 'bagging_fraction': 0.6070653969236076, 'feature_fraction': 0.5606587406627099, 'lambda_l1': 0.24713700506009806, 'lambda_l2': 21.962932156582497, 'min_gain_to_split': 0.12559274012483687, 'min_data_per_group': 76, 'max_cat_threshold': 251, 'cat_l2': 7.5452628341183825, 'cat_smooth': 2040.7727886744697, 'max_cat_to_onehot': 60, 'max_bin': 35, 'min_data_in_bin': 60, 'objective': 'binary'}, 3: {'num_iterations': 119, 'learning_rate': 0.4778299348996808, 'num_leaves': 103, 'max_depth': 17, 'min_data_in_leaf': 5, 'min_sum_hessian_in_leaf': 0.06943101789612632, 'bagging_freq': 1, 'bagging_fraction': 0.6403806552571523, 'feature_fraction': 0.9132566024843901, 'lambda_l1': 5.298223883793813, 'lambda_l2': 0.7684973042168224, 'min_gain_to_split': 0.10077530718476888, 'min_data_per_group': 286, 'max_cat_threshold': 367, 'cat_l2': 47.95378847610971, 'cat_smooth': 970.6071880782908, 'max_cat_to_onehot': 29, 'max_bin': 936, 'min_data_in_bin': 23, 'objective': 'binary'}, 4: {'num_iterations': 55, 'learning_rate': 0.267617080229367, 'num_leaves': 194, 'max_depth': 26, 'min_data_in_leaf': 22, 'min_sum_hessian_in_leaf': 0.203344697430431, 'bagging_freq': 0, 'bagging_fraction': 0.9662303887292932, 'feature_fraction': 0.7314107798635332, 'lambda_l1': 0.44476568923203885, 'lambda_l2': 33.370399354096165, 'min_gain_to_split': 0.0011976125462795015, 'min_data_per_group': 474, 'max_cat_threshold': 158, 'cat_l2': 47.03372475735066, 'cat_smooth': 329.9729880946299, 'max_cat_to_onehot': 49, 'max_bin': 930, 'min_data_in_bin': 54, 'objective': 'binary'}, 5: {'num_iterations': 104, 'learning_rate': 0.3303922278294767, 'num_leaves': 46, 'max_depth': 20, 'min_data_in_leaf': 31, 'min_sum_hessian_in_leaf': 0.1901638300747604, 'bagging_freq': 1, 'bagging_fraction': 0.8073046122477952, 'feature_fraction': 0.7006448563103835, 'lambda_l1': 0.33886174454635487, 'lambda_l2': 21.548040485480172, 'min_gain_to_split': 0.17682631049718167, 'min_data_per_group': 87, 'max_cat_threshold': 334, 'cat_l2': 59.900890147595995, 'cat_smooth': 1468.4889838468994, 'max_cat_to_onehot': 35, 'max_bin': 586, 'min_data_in_bin': 10, 'objective': 'binary'}}
# params = modeler_LGB.hyperoptimize(n_trials = 20)

# params["seed"] = randrange(1, 9999, 1)

modeler_LGB.build_model(parallelize=False,
                        params = params);

forecasts_LGB = modeler_LGB.forecast()
forecasts_LGB["6-period Survival Probability"]

forecasts_LGB


### make SGB ensemble

self = modeler_LGB
percent_confidence = 0.95
n_models = 25
subset = pd.Series(data = np.repeat(True, len(modeler_LGB.data)))



def make_ensemble(
        self,
        n_models: int = 10,
        subset: Union[None, pd.core.series.Series] = None,
        params: dict = None,
        percent_confidence: float = 0.95):

    subset = default_subset_to_all(subset, self.data)
    alpha = (1 - percent_confidence) / 2
    conf = str(int(percent_confidence * 100))
    if params is None:
        params = self.config

    # def one_sgb_prediction(self, params, subset) -> pd.core.frame.DataFrame:
    #     """Compute forecasts for one MC Dropout iteration."""
    #
    #     one_sgb_modeler = LGBSurvivalModeler(data=self.data);
    #
    #     one_sgb_modeler.n_intervals = self.n_intervals
    #
    #     for k in params.keys():
    #         # params[k]["seed"] = randrange(1, 9999, 1)
    #         params[k]["data_random_seed"] = randrange(1, 9999, 1)
    #         params[k]["feature_fraction_seed"] = randrange(1, 9999, 1)
    #         params[k]["bagging_seed"] = randrange(1, 9999, 1)
    #
    #
    #     one_sgb_modeler.build_model(params=params, parallelize=False)
    #     forecasts = one_sgb_modeler.forecast()
    #     forecasts.columns = list(map(str, np.arange(1, len(forecasts.columns) + 1, 1)))
    #     ids_in_subset = self.data[subset]["ID"].unique()
    #     keep_rows = np.repeat(True, len(forecasts))
    #     for rw in range(len(forecasts)):
    #         if forecasts.index[rw] not in ids_in_subset:
    #             keep_rows[rw] = False
    #     forecasts = forecasts[keep_rows]
    #     return (forecasts)

    # t = self.data['_event_observed'].astype(int)
    # lgb_data = self.data

    # N = len(self.data)
    # learning_rate = params[0]['learning_rate']
    # learning_rate = params[5]['learning_rate']
    #
    # langevin_noise = 2 / (N * learning_rate)

    # self.langevin_noise = langevin_noise

    # langevin_noise
    #
    # langevin_noise = 0.5
    # langevin_noise = 0


    def bce_loss(z, lgb_data):
        t = lgb_data.get_label().astype(int)
        y = sigmoid(z)
        langevin_noise = self.langevin_noise
        # print("Langevin SD: " + str(langevin_noise))
        if langevin_noise != 0:
            e = np.random.normal(0, langevin_noise)
        else:
            e = 0
        # print("Langevin Noise: " + str(e))
        grad = y - t + e
        hess = y * (1 - y)
        return grad, hess

    def bce_eval(z, data):
        t = data.get_label().astype(int)
        loss = t * softplus(-z) + (1 - t) * softplus(z)
        return 'bce', loss.mean(), False

    self.config["fobj"] = bce_loss

    # testing

    # one_sglb_modeler = LGBSurvivalModeler(data=self.data);
    #
    # one_sglb_modeler.n_intervals = self.n_intervals
    #
    #
    # one_sglb_modeler
    #
    # for k in params.keys():
    #     params[k]["seed"] = 1
    #     params[k]["random_seed"] = 1
    #     # params[k]["objective"] = binary_objective
    #
    # # one_sglb_modeler.allow_gaps = False
    # # one_sglb_modeler.allow_gaps = True
    #
    # one_sglb_modeler.config["fobj"] = None
    # one_sglb_modeler.config["feval"] = None
    #
    # one_sglb_modeler.build_model(params=params, parallelize=False)
    #
    # forecasts = one_sglb_modeler.forecast()
    #
    #
    #
    # # one_sglb_modeler.config["fobj"] = binary_log_loss_langevin_gradient
    # # one_sglb_modeler.allow_gaps = True
    # one_sglb_modeler.config["fobj"] = bce_loss
    # # one_sglb_modeler.config["feval"] = bce_eval
    #
    #
    # one_sglb_modeler.build_model(params=params, parallelize=False)
    #
    # forecasts2 = one_sglb_modeler.forecast()
    #
    #
    # forecasts2
    # forecasts
    #
    # np.var(forecasts)
    # np.var(forecasts2)
    #




    def one_sglb_prediction(self, params, subset) -> pd.core.frame.DataFrame:
        """Compute forecasts for one MC Dropout iteration."""

        # one_sglb_modeler = LGBSurvivalModeler(data=self.data);
        #
        # one_sglb_modeler.data
        #
        # one_sglb_modeler.n_intervals = self.n_intervals

        # self.config["fobj"] = bce_loss
        # one_sglb_modeler.config["fobj"] = bce_loss

        for k in params.keys():
            params[k]["seed"] = randrange(1, 9999, 1)
            # params[k]["data_random_seed"] = randrange(1, 9999, 1)
            # params[k]["feature_fraction_seed"] = randrange(1, 9999, 1)
            # params[k]["bagging_seed"] = randrange(1, 9999, 1)


        self.build_model(params=params, parallelize=False)
        forecasts = self.forecast()
        forecasts.columns = list(map(str, np.arange(1, len(forecasts.columns) + 1, 1)))
        ids_in_subset = self.data[subset]["ID"].unique()
        keep_rows = np.repeat(True, len(forecasts))
        for rw in range(len(forecasts)):
            if forecasts.index[rw] not in ids_in_subset:
                keep_rows[rw] = False
        forecasts = forecasts[keep_rows]
        return (forecasts)


    ensemble_forecasts = list()

    # do MC dropout for n_iterations
    for i in range(n_models):
        one_forecast = one_sglb_prediction(self, params=params, subset=subset)
        ensemble_forecasts.append(one_forecast)

    ### get mean forecasts
    sum_forecasts = ensemble_forecasts[0]
    for i in range(1, len(ensemble_forecasts)):
        sum_forecasts = sum_forecasts + ensemble_forecasts[i]

    mean_forecasts = sum_forecasts / len(ensemble_forecasts)

    def get_forecast_variance() -> pd.DataFrame:
        """ Get variance across forecasts"""

        def get_forecast_variance_for_time(time=0):

            def get_variance_for_index_time(index, time=0):
                diff_from_mean_df = ensemble_forecasts[index].iloc[:, [time]] - mean_forecasts.iloc[:, [time]]
                squared_error_df = diff_from_mean_df ** 2
                return squared_error_df

            squared_error_sum_df = get_variance_for_index_time(0, time=time)
            for i in range(1, len(ensemble_forecasts)):
                squared_error_sum_df = squared_error_sum_df + get_variance_for_index_time(i, time=time)
            variance_df = squared_error_sum_df / len(ensemble_forecasts)
            return variance_df

        variance_all_times_df = get_forecast_variance_for_time(time=0)
        for t in range(1, len(mean_forecasts.columns)):
            variance_all_times_df = pd.concat([variance_all_times_df, get_forecast_variance_for_time(time=t)],
                                              axis=1)

        return variance_all_times_df

    def aggregate_observation_ensemble_forecasts(ensemble_forecasts, rw=0) -> pd.DataFrame:
        """Get collection of forecasts from MC Dropout for one observation"""

        survival_probability = pd.Series(dtype="float32")
        period = np.empty(shape=0, dtype="int")
        iteration = np.empty(shape=0, dtype="int")

        for iter in range(len(ensemble_forecasts)):
            observation_forecasts = ensemble_forecasts[iter].iloc[rw]
            id = ensemble_forecasts[iter].index[rw]
            survival_probability = survival_probability.append(observation_forecasts)
            keys = observation_forecasts.keys()
            period = np.append(period, observation_forecasts.keys().to_numpy(dtype="int"))
            iteration = np.append(iteration, np.repeat(iter, len(keys)))

        id = np.repeat(ensemble_forecasts[0].index[rw], len(iteration))
        forecast_df = pd.DataFrame({"ID": id, "iter": iteration, "period": period,
                                    "survival_prob": survival_probability})

        return forecast_df

    def get_forecast_prediction_intervals(alpha=0.025) -> pd.DataFrame:
        """Get prediction intervals for all observations for all time points"""

        prediction_intervals_df = pd.DataFrame({
            "ID": np.empty(shape=0, dtype=type(ensemble_forecasts[0].index[0])),
            "Period": np.empty(shape=0, dtype="str"),
            "Lower" + conf + "PercentBound": np.empty(shape=0, dtype="float"),
            "Median": np.empty(shape=0, dtype="float"),
            "Mean": np.empty(shape=0, dtype="float"),
            "Upper" + conf + "PercentBound": np.empty(shape=0, dtype="float")
        })

        for rw in range(len(mean_forecasts.index)):
            forecast_df = aggregate_observation_ensemble_forecasts(ensemble_forecasts, rw)
            for p in range(len(forecast_df['period'].unique())):
                period = p + 1
                mean_forecast = mean_forecasts.iloc[rw, p]
                period_forecasts = forecast_df[forecast_df['period'] == period]['survival_prob']
                forecast_quantiles = period_forecasts.quantile([alpha, 0.5, 1 - alpha])
                df = pd.DataFrame(
                    {"ID": forecast_df["ID"][0],
                     "Period": [period],
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
            "ID": np.empty(shape=0, dtype=type(ensemble_forecasts[0].index[0])),
            "Period": np.empty(shape=0, dtype="str"),
            "Lower" + conf + "PercentBound": np.empty(shape=0, dtype="float"),
            "Median": np.empty(shape=0, dtype="float"),
            "Mean": np.empty(shape=0, dtype="float"),
            "Upper" + conf + "PercentBound": np.empty(shape=0, dtype="float")
        })

        for p in range(len(ensemble_forecasts[0].columns.unique())):
            period = p + 1
            dropout_expected_counts = []
            for d in ensemble_forecasts:
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

    prediction_intervals_df

    return prediction_intervals_df



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
            warn("Invalid ID.", UserWarning)
            return None
    forecast_df = forecast_df.assign(Period=forecast_df['Period'].tolist())
    forecast_df = forecast_df.rename({forecast_df.columns[2]: "Lower", forecast_df.columns[5]: "Upper"}, axis = "columns")
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



df = prediction_intervals_df

plot_forecast_prediction_intervals(self, prediction_intervals_df)
print(*df['ID'])

ID = 925


plot_forecast_prediction_intervals(self, prediction_intervals_df, ID = ID)














