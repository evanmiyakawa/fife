"""Import"""

from fife.lgb_modelers import LGBSurvivalModeler
from fife.tf_modelers import TFSurvivalModeler
from fife.processors import PanelDataProcessor
from fife.utils import make_results_reproducible
from pandas import concat, date_range, read_csv, to_datetime
import pandas as pd
import numpy as np
from ppprint import ppprint


### Set up basic data, subset of original ###

dat = read_csv("Evan/REIGN_2020_7.csv")
# data = read_csv("Evan/REIGN_2020_7.csv")

countries_unique = np.unique(dat["country"])

countries_sub = countries_unique[0:10]

# multiple ways to do filtering
# data.query("country in countries_sub")
data = dat[dat["country"].isin(countries_sub)]

#data = dat[1:100]


data["country-leader"] = data["country"] + ": " + data["leader"]
data["year-month"] = data["year"].astype(int).astype(str) + data["month"].astype(int).astype(str).str.zfill(2)
data["year-month"] = to_datetime(data["year-month"], format="%Y%m")
data = concat([data[["country-leader", "year-month"]],
               data.drop(["ccode", "country-leader", "leader", "year-month"],
                         axis=1)],
               axis=1)



total_obs = len(data)
data = data.drop_duplicates(["country-leader", "year-month"], keep="first")
n_duplicates = total_obs - len(data)

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

from fife.tf_modelers import TFSurvivalModeler
from fife.tf_modelers import FeedforwardNeuralNetworkModeler


modeler_TF = TFSurvivalModeler(data = data_processor.data)
modeler_TF.build_model()
# params = modeler_TF.hyperoptimize(n_trials = 1)
# modeler_TF.build_model(params = params)
forecasts_TF = modeler_TF.forecast()
# forecasts_TF["15-period Survival Probability"]

forecasts_TF.columns = list(map(str, np.arange(1,len(forecasts_TF.columns) + 1,1)))


# modeler_FF = FeedforwardNeuralNetworkModeler(data = data_processor.data)
# modeler_FF.build_model()
# forecasts_TF = modeler_TF.forecast()
# forecasts_TF["50-period Survival Probability"]




# data.columns




### Gal MC dropout existing implementation ###

from fife import tf_modelers
from fife.base_modelers import default_subset_to_all
from fife.tf_modelers import split_categorical_features
from typing import List, Union
import tensorflow.keras.backend as K




### Gal function ####

self = modeler_TF
subset = None
n_iterations = 200

def compute_model_uncertainty(
        self, subset: Union[None, pd.core.series.Series] = None, n_iterations: int = 200
) -> np.ndarray:
    """Predict with dropout as proposed by Gal and Ghahramani (2015).

    See https://arxiv.org/abs/1506.02142.

    Args:
        subset: A Boolean Series that is True for observations for which
            predictions will be produced. If None, default to all
            observations.
        n_iterations: Number of random dropout specifications to obtain
            predictions from.

    Returns:
        A numpy array of predictions by observation, lead length, and
        iteration.
    """
    subset = default_subset_to_all(subset, self.data)
    model_inputs = split_categorical_features(
        self.data[subset], self.categorical_features, self.numeric_features
    )
    predict_with_dropout = K.function(
        # self.model.inputs + [K.learning_phase()], self.model.outputs
        self.model.inputs, self.model.outputs
    )
    predictions = np.dstack(
        [predict_with_dropout(model_inputs)[0] for i in range(n_iterations)]
    )
    return predictions


typemodel_inputs

modeler_TF.compute_model_uncertainty()

type(model_inputs)
len(model_inputs)

self2 = predict_with_dropout
inputs = model_inputs

def temp(self2, inputs):
    if not isinstance(inputs, (list, tuple)):
      raise TypeError('`inputs` should be a list or tuple.')
    feed_dict = {}
    for tensor, value in zip(self2.inputs, inputs):
      if is_sparse(tensor):
        sparse_coo = value.tocoo()
        indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
                                  np.expand_dims(sparse_coo.col, 1)), 1)
        value = (indices, sparse_coo.data, sparse_coo.shape)
      feed_dict[tensor] = value
    session = get_session()
    updated = session.run(
        self2.outputs + [self2.updates_op],
        feed_dict=feed_dict,
        **self2.session_kwargs)
    return updated[:len(self2.outputs)]



type(model_inputs[32])

type(model_inputs[32]['year-month'])


# predict_with_dropout(self.model.inputs)[0]

type(self.model.inputs)

len(self.model.inputs)
len(self.model.outputs)

type(model_inputs[0][0])

predict_with_dropout()