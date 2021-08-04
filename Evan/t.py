data = fabricate_data(
    N_PERSONS=N_PERSONS,
    N_PERIODS=N_PERIODS,
    exit_prob=exit_prob,
    SEED=SEED,
    dgp=1,
    covariates_affect_outcome=True
)

data = data.drop("exit_type", axis=1)
train_data = data[data['period'] < np.round(N_PERIODS / 2)]
train_data = train_data.drop("exit_prob", axis=1)
test_data = data[data['period'] >= np.round(N_PERIODS / 2)]

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

### TF Modeler ###

data_processor = PanelDataProcessor(data=train_data)
data_processor.build_processed_data()

data_processor.data

modeler_LGB = LGBSurvivalModeler(data=data_processor.data)

# modeler_LGB.n_intervals = modeler_LGB.set_n_intervals()
modeler_LGB.n_intervals = 10

params_lgb = modeler_LGB.hyperoptimize()

lgb_uncertainty = modeler_LGB.compute_model_uncertainty(
    n_iterations=ensemble_size,
    params=params_lgb,
    percent_confidence=0.95
    # langevin_variance = 0.4
)

