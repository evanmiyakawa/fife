
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


# SEED = 1235
# N_PERSONS = 2000
# N_PERIODS = 50
# exit_prob = 0.4
# ensemble_size = 20


def conduct_one_simulation(N_PERSONS: int = 1000, N_PERIODS: int = 10, exit_prob: float = 0.3, ensemble_size: int = 20):
    """This function assesses out of sample PICP for simulated data with known underlying survival probabilities.
    This function is not finished and does not return anything currently.

    """


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


    data = read_csv("Evan/REIGN_2020_7.csv")

    data["country-leader"] = data["country"] + ": " + data["leader"]
    data["year-month"] = data["year"].astype(int).astype(str) + data["month"].astype(int).astype(str).str.zfill(2)
    data["year-month"] = to_datetime(data["year-month"], format="%Y%m")
    data = concat([data[["country-leader", "year-month"]],
                   data.drop(["ccode", "country-leader", "leader", "year-month"],
                             axis=1)],
                  axis=1)
    total_obs = len(data)
    data = data.drop_duplicates(["country-leader", "year-month"], keep="first")


    unique_months = data['year-month'].unique()
    len(unique_months)
    cutoff_time = unique_months[int(np.round(len(unique_months) * 0.8))]


    train_data = data[data['year-month'] < cutoff_time]
    test_data = data[data['year-month'] >= cutoff_time]

    ### TF Modeler ###

    data_processor = PanelDataProcessor(data=train_data)
    data_processor.build_processed_data()

    data_processor.data

    ### Create TF modeler ###

    modeler_TF = TFSurvivalModeler(data=data_processor.data)

    modeler_TF.n_intervals = np.min([10, modeler_TF.set_n_intervals()])


    params_tf_sigmoid = modeler_TF.hyperoptimize()

    # used with experiment n = 1000

    # experiment n = 2000, exit_prob = 0.4, N_Periods = 50
    # params_tf_sigmoid = {'BATCH_SIZE': 43, 'DENSE_LAYERS': 1, 'DROPOUT_SHARE': 0.0919745976616121, 'EMBED_EXPONENT': 0.1776661700014221, 'EMBED_L2_REG': 0.11395741867225184, 'POST_FREEZE_EPOCHS': 26, 'PRE_FREEZE_EPOCHS': 76, 'NODES_PER_DENSE_LAYER': 725}


    ### compute uncertainty ###

    modeler_TF.build_model(params = params_tf_sigmoid)

    modeler_TF.forecast()

    tf_uncertainty = modeler_TF.compute_model_uncertainty(
        n_iterations=ensemble_size,
        params=params_tf_sigmoid,
        percent_confidence=0.95
    )

    tf_uncertainty

    original_tf_uncertainty = tf_uncertainty.copy()

    model_variance = 0.2 ** 2
    percent_confidence = 0.95
    def add_model_variance(tf_uncertainty, model_variance, percent_confidence = 0.95):
        alpha = (1 - percent_confidence) / 2
        model_sd = np.sqrt(model_variance)
        num_periods = np.max(tf_uncertainty['Period'])
        num_observations = len(tf_uncertainty[tf_uncertainty['Period'] == 1]) - 1
        add_to_interval = -norm.ppf(alpha) * model_sd
        lb_below_median = tf_uncertainty['LB Below Median'] - add_to_interval
        ub_above_median = tf_uncertainty['UB Above Median'] + add_to_interval
        lb_below_median[0:num_periods] = tf_uncertainty['LB Below Median'][0:num_periods] - -norm.ppf(alpha) * (np.sqrt(model_variance * num_observations))
        ub_above_median[0:num_periods] = tf_uncertainty['UB Above Median'][0:num_periods] + -norm.ppf(alpha) * (np.sqrt(model_variance * num_observations))
        new_lb = tf_uncertainty['Median'] + lb_below_median
        new_ub = tf_uncertainty['Median'] + ub_above_median

        for i in range(len(new_lb)):
            if new_lb[i] < 0:
                new_lb[i] = 0
        for i in range(num_periods, len(new_ub)):
            if new_ub[i] > 1:
                new_ub[i] = 1
        for i in range(num_periods):
            if new_ub[i] > num_observations:
                new_ub[i] = num_observations


        tf_uncertainty.iloc[:,4] = new_lb
        tf_uncertainty.iloc[:,7] = new_ub

        return tf_uncertainty


    tf_uncertainty = add_model_variance(
        original_tf_uncertainty,
        model_variance = 0.08 ** 2,
        percent_confidence = 0.95
    )

    # modeler_TF.plot_forecast_prediction_intervals(tf_uncertainty)
    # modeler_TF.plot_forecast_prediction_intervals(tf_uncertainty, 17)
    # modeler_TF.plot_forecast_prediction_intervals(tf_uncertainty, 1170)

    # ID = 999

    ids_in_forecasts = tf_uncertainty["ID"].unique()
    ids_in_forecasts = ids_in_forecasts[ids_in_forecasts != "Sum"]

    ids_in_forecasts

    test_data_keep = []
    i = 0
    for i in range(len(test_data)):
        if test_data["ID"].iloc[i] in ids_in_forecasts:
            test_data_keep.append(i)

    test_data1 = test_data.iloc[test_data_keep]


    def get_one_id_in_interval(ID):
        id_uncertainty_df = tf_uncertainty[tf_uncertainty["ID"] == ID]
        id_test_df = test_data1[test_data1["ID"] == ID]
        num_periods = np.min([len(id_uncertainty_df), len(id_test_df)])
        id_uncertainty_df = id_uncertainty_df[0:num_periods]

        survival_prob = 1 - id_test_df['exit_prob'].to_numpy()[0:num_periods]

        for i in range(1, len(survival_prob)):
            survival_prob[i] = survival_prob[i] * survival_prob[i-1]

        id_uncertainty_df['SurvivalProb'] = survival_prob

        above_lb = (id_uncertainty_df['Lower95PercentBound'] < id_uncertainty_df['SurvivalProb']).to_numpy()
        below_ub = (id_uncertainty_df['Upper95PercentBound'] > id_uncertainty_df['SurvivalProb']).to_numpy()

        in_interval = np.empty(shape=num_periods, dtype="bool")
        for p in range(len(in_interval)):
            if above_lb[p] and below_ub[p]:
                in_interval[p] = True
            else:
                in_interval[p] = False

        id_uncertainty_df["InInterval"] = in_interval

        return id_uncertainty_df

    all_uncertainty_df = get_one_id_in_interval(ID = ids_in_forecasts[0])
    for i in range(1, len(ids_in_forecasts)):
        df = get_one_id_in_interval(ID = ids_in_forecasts[i])
        if (len(df) > 0):
            all_uncertainty_df = pd.concat([all_uncertainty_df, df])

    all_uncertainty_df

    all_uncertainty_df.iloc[:,[1,4,7,8,9]]

    # Out-of-sample PI coverage probability across all time horizons
    picp = np.sum(all_uncertainty_df["InInterval"]) / len(all_uncertainty_df)

    picp

    # PICP for specific time horizons

    def tf_horizon_picp(horizon):
        horizon_uncertainty_df = all_uncertainty_df[all_uncertainty_df["Period"] == horizon]
        picp_tf_horizon = np.sum(horizon_uncertainty_df["InInterval"]) / len(horizon_uncertainty_df)
        return picp_tf_horizon

    tf_horizon_picp(1)
    tf_horizon_picp(2)
    tf_horizon_picp(3)
    tf_horizon_picp(5)
    tf_horizon_picp(9)
    tf_horizon_picp(10)



    ### chernoff bounds

    forecasts = modeler_TF.forecast()

    chernoff_bounds = compute_aggregation_uncertainty(forecasts)
    chernoff_bounds.iloc[:,[1,2,3]]




    ### LGB Modeler

    modeler_LGB = LGBSurvivalModeler(data=data_processor.data)

    modeler_LGB.n_intervals = np.min([10, modeler_LGB.set_n_intervals()])

    params_lgb = modeler_LGB.hyperoptimize()

    # used in experiment with n = 2000, exit_prob = 0.4, periods = 50
    # params_lgb = {0: {'num_iterations': 40, 'learning_rate': 0.054847307516124974, 'num_leaves': 195, 'max_depth': 17, 'min_data_in_leaf': 230, 'min_sum_hessian_in_leaf': 0.10629947853891941, 'bagging_freq': 1, 'bagging_fraction': 0.5126590495875429, 'feature_fraction': 0.8064549591178001, 'lambda_l1': 0.00012897198080030192, 'lambda_l2': 0.04340068844005275, 'min_gain_to_split': 0.23013066404695204, 'min_data_per_group': 154, 'max_cat_threshold': 81, 'cat_l2': 35.76720695701277, 'cat_smooth': 428.4782823888627, 'max_cat_to_onehot': 34, 'max_bin': 559, 'min_data_in_bin': 48, 'objective': 'binary'}, 1: {'num_iterations': 40, 'learning_rate': 0.054847307516124974, 'num_leaves': 195, 'max_depth': 17, 'min_data_in_leaf': 230, 'min_sum_hessian_in_leaf': 0.10629947853891941, 'bagging_freq': 1, 'bagging_fraction': 0.5126590495875429, 'feature_fraction': 0.8064549591178001, 'lambda_l1': 0.00012897198080030192, 'lambda_l2': 0.04340068844005275, 'min_gain_to_split': 0.23013066404695204, 'min_data_per_group': 154, 'max_cat_threshold': 81, 'cat_l2': 35.76720695701277, 'cat_smooth': 428.4782823888627, 'max_cat_to_onehot': 34, 'max_bin': 559, 'min_data_in_bin': 48, 'objective': 'binary'}, 2: {'num_iterations': 40, 'learning_rate': 0.054847307516124974, 'num_leaves': 195, 'max_depth': 17, 'min_data_in_leaf': 230, 'min_sum_hessian_in_leaf': 0.10629947853891941, 'bagging_freq': 1, 'bagging_fraction': 0.5126590495875429, 'feature_fraction': 0.8064549591178001, 'lambda_l1': 0.00012897198080030192, 'lambda_l2': 0.04340068844005275, 'min_gain_to_split': 0.23013066404695204, 'min_data_per_group': 154, 'max_cat_threshold': 81, 'cat_l2': 35.76720695701277, 'cat_smooth': 428.4782823888627, 'max_cat_to_onehot': 34, 'max_bin': 559, 'min_data_in_bin': 48, 'objective': 'binary'}, 3: {'num_iterations': 40, 'learning_rate': 0.054847307516124974, 'num_leaves': 195, 'max_depth': 17, 'min_data_in_leaf': 230, 'min_sum_hessian_in_leaf': 0.10629947853891941, 'bagging_freq': 1, 'bagging_fraction': 0.5126590495875429, 'feature_fraction': 0.8064549591178001, 'lambda_l1': 0.00012897198080030192, 'lambda_l2': 0.04340068844005275, 'min_gain_to_split': 0.23013066404695204, 'min_data_per_group': 154, 'max_cat_threshold': 81, 'cat_l2': 35.76720695701277, 'cat_smooth': 428.4782823888627, 'max_cat_to_onehot': 34, 'max_bin': 559, 'min_data_in_bin': 48, 'objective': 'binary'}, 4: {'num_iterations': 9, 'learning_rate': 0.4371979591383067, 'num_leaves': 182, 'max_depth': 14, 'min_data_in_leaf': 400, 'min_sum_hessian_in_leaf': 0.19810064824490978, 'bagging_freq': 1, 'bagging_fraction': 0.9447375702617523, 'feature_fraction': 0.8060536105220231, 'lambda_l1': 0.00921155084912419, 'lambda_l2': 0.00038661973970888236, 'min_gain_to_split': 0.1861754944824994, 'min_data_per_group': 503, 'max_cat_threshold': 342, 'cat_l2': 41.70345545189665, 'cat_smooth': 504.51350379224084, 'max_cat_to_onehot': 18, 'max_bin': 581, 'min_data_in_bin': 8, 'objective': 'binary'}, 5: {'num_iterations': 24, 'learning_rate': 0.3399769785700486, 'num_leaves': 57, 'max_depth': 26, 'min_data_in_leaf': 318, 'min_sum_hessian_in_leaf': 0.09736884236345539, 'bagging_freq': 1, 'bagging_fraction': 0.643301085090667, 'feature_fraction': 0.5302622339961853, 'lambda_l1': 0.09785722971899571, 'lambda_l2': 0.09386691904045663, 'min_gain_to_split': 0.24883194895518074, 'min_data_per_group': 115, 'max_cat_threshold': 130, 'cat_l2': 23.714642462310962, 'cat_smooth': 769.015126671467, 'max_cat_to_onehot': 27, 'max_bin': 46, 'min_data_in_bin': 56, 'objective': 'binary'}, 6: {'num_iterations': 40, 'learning_rate': 0.054847307516124974, 'num_leaves': 195, 'max_depth': 17, 'min_data_in_leaf': 230, 'min_sum_hessian_in_leaf': 0.10629947853891941, 'bagging_freq': 1, 'bagging_fraction': 0.5126590495875429, 'feature_fraction': 0.8064549591178001, 'lambda_l1': 0.00012897198080030192, 'lambda_l2': 0.04340068844005275, 'min_gain_to_split': 0.23013066404695204, 'min_data_per_group': 154, 'max_cat_threshold': 81, 'cat_l2': 35.76720695701277, 'cat_smooth': 428.4782823888627, 'max_cat_to_onehot': 34, 'max_bin': 559, 'min_data_in_bin': 48, 'objective': 'binary'}, 7: {'num_iterations': 43, 'learning_rate': 0.31786795242749843, 'num_leaves': 56, 'max_depth': 17, 'min_data_in_leaf': 208, 'min_sum_hessian_in_leaf': 0.04999680708985953, 'bagging_freq': 1, 'bagging_fraction': 0.5923845257989745, 'feature_fraction': 0.5788351806135542, 'lambda_l1': 0.015183923109000456, 'lambda_l2': 0.06309733470499498, 'min_gain_to_split': 0.1828777629747, 'min_data_per_group': 67, 'max_cat_threshold': 271, 'cat_l2': 61.95564202609179, 'cat_smooth': 683.4765827463168, 'max_cat_to_onehot': 51, 'max_bin': 617, 'min_data_in_bin': 49, 'objective': 'binary'}, 8: {'num_iterations': 45, 'learning_rate': 0.4277116740512498, 'num_leaves': 173, 'max_depth': 20, 'min_data_in_leaf': 238, 'min_sum_hessian_in_leaf': 0.14996334014924353, 'bagging_freq': 1, 'bagging_fraction': 0.7114454822068397, 'feature_fraction': 0.6614377616849302, 'lambda_l1': 0.0012847139905505108, 'lambda_l2': 0.04736660435737013, 'min_gain_to_split': 0.022010674712707505, 'min_data_per_group': 290, 'max_cat_threshold': 399, 'cat_l2': 30.513063797369636, 'cat_smooth': 1794.6346620219133, 'max_cat_to_onehot': 18, 'max_bin': 537, 'min_data_in_bin': 16, 'objective': 'binary'}, 9: {'num_iterations': 95, 'learning_rate': 0.41830688405555094, 'num_leaves': 199, 'max_depth': 31, 'min_data_in_leaf': 319, 'min_sum_hessian_in_leaf': 0.15961774200093565, 'bagging_freq': 0, 'bagging_fraction': 0.6749524210724969, 'feature_fraction': 0.7210691609252785, 'lambda_l1': 0.004380497110815287, 'lambda_l2': 0.005989020875602936, 'min_gain_to_split': 0.22080548385464802, 'min_data_per_group': 395, 'max_cat_threshold': 428, 'cat_l2': 19.53264878762593, 'cat_smooth': 165.06016643531012, 'max_cat_to_onehot': 1, 'max_bin': 552, 'min_data_in_bin': 16, 'objective': 'binary'}}

    # # used in countries data
    # params_lgb = {0: {'num_iterations': 122, 'learning_rate': 0.2915252816037152, 'num_leaves': 99, 'max_depth': 4, 'min_data_in_leaf': 96, 'min_sum_hessian_in_leaf': 0.20172852928181512, 'bagging_freq': 0, 'bagging_fraction': 0.6046631553047732, 'feature_fraction': 0.5153532774904914, 'lambda_l1': 0.04987458804854229, 'lambda_l2': 0.0005002703622912462, 'min_gain_to_split': 0.19619270163907934, 'min_data_per_group': 3, 'max_cat_threshold': 227, 'cat_l2': 62.4605416113113, 'cat_smooth': 760.7448768379404, 'max_cat_to_onehot': 2, 'max_bin': 909, 'min_data_in_bin': 63, 'objective': 'binary'}, 1: {'num_iterations': 22, 'learning_rate': 0.20613524371903416, 'num_leaves': 139, 'max_depth': 11, 'min_data_in_leaf': 419, 'min_sum_hessian_in_leaf': 0.09926481071964113, 'bagging_freq': 0, 'bagging_fraction': 0.5897226325089266, 'feature_fraction': 0.90061892598936, 'lambda_l1': 0.0008093105967174184, 'lambda_l2': 0.00010589415852452283, 'min_gain_to_split': 0.03322817480469905, 'min_data_per_group': 382, 'max_cat_threshold': 356, 'cat_l2': 40.690212370538816, 'cat_smooth': 420.9137597902731, 'max_cat_to_onehot': 54, 'max_bin': 767, 'min_data_in_bin': 22, 'objective': 'binary'}, 2: {'num_iterations': 11, 'learning_rate': 0.27585948842478264, 'num_leaves': 31, 'max_depth': 27, 'min_data_in_leaf': 510, 'min_sum_hessian_in_leaf': 0.09226137389884365, 'bagging_freq': 0, 'bagging_fraction': 0.6292416459542836, 'feature_fraction': 0.8865074249682052, 'lambda_l1': 0.022347649810802585, 'lambda_l2': 0.050653486800074886, 'min_gain_to_split': 0.11586425412459911, 'min_data_per_group': 16, 'max_cat_threshold': 384, 'cat_l2': 36.73886285728289, 'cat_smooth': 423.2095164536322, 'max_cat_to_onehot': 37, 'max_bin': 278, 'min_data_in_bin': 54, 'objective': 'binary'}, 3: {'num_iterations': 36, 'learning_rate': 0.1574828399076171, 'num_leaves': 158, 'max_depth': 14, 'min_data_in_leaf': 111, 'min_sum_hessian_in_leaf': 0.13135330303701362, 'bagging_freq': 0, 'bagging_fraction': 0.5263504328314093, 'feature_fraction': 0.8112284347486276, 'lambda_l1': 0.00984704961001501, 'lambda_l2': 0.000465783854354588, 'min_gain_to_split': 0.1587848310683472, 'min_data_per_group': 149, 'max_cat_threshold': 1, 'cat_l2': 0.6435994344210236, 'cat_smooth': 361.48137521457016, 'max_cat_to_onehot': 37, 'max_bin': 594, 'min_data_in_bin': 29, 'objective': 'binary'}, 4: {'num_iterations': 40, 'learning_rate': 0.054847307516124974, 'num_leaves': 195, 'max_depth': 17, 'min_data_in_leaf': 230, 'min_sum_hessian_in_leaf': 0.10629947853891941, 'bagging_freq': 1, 'bagging_fraction': 0.5126590495875429, 'feature_fraction': 0.8064549591178001, 'lambda_l1': 0.00012897198080030192, 'lambda_l2': 0.04340068844005275, 'min_gain_to_split': 0.23013066404695204, 'min_data_per_group': 154, 'max_cat_threshold': 81, 'cat_l2': 35.76720695701277, 'cat_smooth': 428.4782823888627, 'max_cat_to_onehot': 34, 'max_bin': 559, 'min_data_in_bin': 48, 'objective': 'binary'}, 5: {'num_iterations': 15, 'learning_rate': 0.2226225789330489, 'num_leaves': 10, 'max_depth': 12, 'min_data_in_leaf': 357, 'min_sum_hessian_in_leaf': 0.10712361545268292, 'bagging_freq': 1, 'bagging_fraction': 0.7795148808216431, 'feature_fraction': 0.7690686395807743, 'lambda_l1': 0.00022967656980881187, 'lambda_l2': 0.0003487182762106815, 'min_gain_to_split': 0.07005485734597561, 'min_data_per_group': 177, 'max_cat_threshold': 164, 'cat_l2': 35.21493845282157, 'cat_smooth': 457.6026532280948, 'max_cat_to_onehot': 47, 'max_bin': 360, 'min_data_in_bin': 31, 'objective': 'binary'}, 6: {'num_iterations': 40, 'learning_rate': 0.054847307516124974, 'num_leaves': 195, 'max_depth': 17, 'min_data_in_leaf': 230, 'min_sum_hessian_in_leaf': 0.10629947853891941, 'bagging_freq': 1, 'bagging_fraction': 0.5126590495875429, 'feature_fraction': 0.8064549591178001, 'lambda_l1': 0.00012897198080030192, 'lambda_l2': 0.04340068844005275, 'min_gain_to_split': 0.23013066404695204, 'min_data_per_group': 154, 'max_cat_threshold': 81, 'cat_l2': 35.76720695701277, 'cat_smooth': 428.4782823888627, 'max_cat_to_onehot': 34, 'max_bin': 559, 'min_data_in_bin': 48, 'objective': 'binary'}, 7: {'num_iterations': 11, 'learning_rate': 0.1702961290473979, 'num_leaves': 179, 'max_depth': 22, 'min_data_in_leaf': 269, 'min_sum_hessian_in_leaf': 0.15752561167843093, 'bagging_freq': 0, 'bagging_fraction': 0.9578774744577249, 'feature_fraction': 0.8570277889711919, 'lambda_l1': 0.0001324710685646852, 'lambda_l2': 0.005333862202109959, 'min_gain_to_split': 0.24649235922179286, 'min_data_per_group': 379, 'max_cat_threshold': 8, 'cat_l2': 21.98774734225622, 'cat_smooth': 345.76118232635145, 'max_cat_to_onehot': 34, 'max_bin': 106, 'min_data_in_bin': 35, 'objective': 'binary'}, 8: {'num_iterations': 11, 'learning_rate': 0.15752820694008907, 'num_leaves': 239, 'max_depth': 26, 'min_data_in_leaf': 213, 'min_sum_hessian_in_leaf': 0.04434104149628673, 'bagging_freq': 0, 'bagging_fraction': 0.5018078874549995, 'feature_fraction': 0.6581524470916423, 'lambda_l1': 0.0017400095999003716, 'lambda_l2': 0.06945084966760164, 'min_gain_to_split': 0.22851244788797354, 'min_data_per_group': 115, 'max_cat_threshold': 102, 'cat_l2': 3.9800920839843785, 'cat_smooth': 629.4586461963283, 'max_cat_to_onehot': 20, 'max_bin': 76, 'min_data_in_bin': 60, 'objective': 'binary'}, 9: {'num_iterations': 40, 'learning_rate': 0.054847307516124974, 'num_leaves': 195, 'max_depth': 17, 'min_data_in_leaf': 230, 'min_sum_hessian_in_leaf': 0.10629947853891941, 'bagging_freq': 1, 'bagging_fraction': 0.5126590495875429, 'feature_fraction': 0.8064549591178001, 'lambda_l1': 0.00012897198080030192, 'lambda_l2': 0.04340068844005275, 'min_gain_to_split': 0.23013066404695204, 'min_data_per_group': 154, 'max_cat_threshold': 81, 'cat_l2': 35.76720695701277, 'cat_smooth': 428.4782823888627, 'max_cat_to_onehot': 34, 'max_bin': 559, 'min_data_in_bin': 48, 'objective': 'binary'}}

    modeler_LGB.build_model(params = params_lgb, parallelize=False,
                            n_intervals = modeler_LGB.n_intervals)


    lgb_uncertainty = modeler_LGB.compute_model_uncertainty(
        n_iterations=ensemble_size,
        params=params_lgb,
        percent_confidence=0.95,
        langevin_variance = 0
    )


    lgb_uncertainty

    lgb_uncertainty.iloc[:,3:5]

    # modeler_LGB.plot_forecast_prediction_intervals(lgb_uncertainty)
    # modeler_LGB.plot_forecast_prediction_intervals(lgb_uncertainty, 1170)

    individual_id = modeler_LGB.config["INDIVIDUAL_IDENTIFIER"]


    ids_in_forecasts = lgb_uncertainty["ID"].unique()
    ids_in_forecasts = ids_in_forecasts[ids_in_forecasts != "Sum"]

    ids_in_forecasts

    test_data_keep = []
    i = 0
    for i in range(len(test_data)):
        if test_data[individual_id].iloc[i] in ids_in_forecasts:
            test_data_keep.append(i)

    test_data1 = test_data.iloc[test_data_keep]



    def get_one_id_in_interval(ID):
        id_uncertainty_df = lgb_uncertainty[lgb_uncertainty["ID"] == ID]
        id_test_df = test_data1[test_data1[individual_id] == ID]
        num_periods = np.min([len(id_uncertainty_df), len(id_test_df)])
        id_uncertainty_df = id_uncertainty_df[0:num_periods]

        survival_prob = 1 - id_test_df['exit_prob'].to_numpy()[0:num_periods]

        for i in range(1, len(survival_prob)):
            survival_prob[i] = survival_prob[i] * survival_prob[i-1]

        id_uncertainty_df['SurvivalProb'] = survival_prob

        above_lb = (id_uncertainty_df['Lower95PercentBound'] < id_uncertainty_df['SurvivalProb']).to_numpy()
        below_ub = (id_uncertainty_df['Upper95PercentBound'] > id_uncertainty_df['SurvivalProb']).to_numpy()

        in_interval = np.empty(shape=num_periods, dtype="bool")
        for p in range(len(in_interval)):
            if above_lb[p] and below_ub[p]:
                in_interval[p] = True
            else:
                in_interval[p] = False

        id_uncertainty_df["InInterval"] = in_interval

        return id_uncertainty_df

    all_uncertainty_df = get_one_id_in_interval(ID = ids_in_forecasts[0])
    for i in range(1, len(ids_in_forecasts)):
        df = get_one_id_in_interval(ID = ids_in_forecasts[i])
        if (len(df) > 0):
            all_uncertainty_df = pd.concat([all_uncertainty_df, df])

    all_uncertainty_df

    all_uncertainty_df.iloc[:,[1,2,5,6,7]]

    # Out-of-sample PI coverage probability across all time horizons
    picp_lgb = np.sum(all_uncertainty_df["InInterval"]) / len(all_uncertainty_df)
    picp_lgb

    # PICP for each time horizon
    def lgb_horizon_picp(horizon):
        horizon_uncertainty_df = all_uncertainty_df[all_uncertainty_df["Period"] == horizon]
        picp_lgb_horizon = np.sum(horizon_uncertainty_df["InInterval"]) / len(horizon_uncertainty_df)
        return picp_lgb_horizon

    lgb_horizon_picp(1)
    lgb_horizon_picp(2)
    lgb_horizon_picp(3)
    lgb_horizon_picp(5)
    lgb_horizon_picp(9)
    lgb_horizon_picp(10)


}