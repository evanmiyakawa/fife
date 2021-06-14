"""Import"""

from fife.lgb_modelers import LGBSurvivalModeler
from fife.processors import PanelDataProcessor
from fife.utils import make_results_reproducible
from pandas import concat, date_range, read_csv, to_datetime
import pandas as pd
import numpy as np


### Set up basic data, subset of original ###

dat = read_csv("Evan/REIGN_2020_7.csv")
# data = read_csv("Evan/REIGN_2020_7.csv")

countries_unique = np.unique(dat["country"])

countries_sub = countries_unique[0:100]

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

modeler = LGBSurvivalModeler(data=data_processor.data);
modeler.build_model(parallelize=False);

forecasts = modeler.forecast()

# data.columns

print(forecasts)
print(forecasts["12-period Survival Probability"])



### Chernoff Bounds existing implementation ###

forecasts.columns = list(map(str, np.arange(1,len(forecasts.columns) + 1,1)))

from fife.utils import compute_aggregation_uncertainty

chernoff_df = compute_aggregation_uncertainty(forecasts)

chernoff_df


# test function directly

percent_confidence = 0.95
individual_probabilities = forecasts

def compute_aggregation_uncertainty(
    individual_probabilities: pd.core.frame.DataFrame, percent_confidence: float = 0.95
) -> pd.core.frame.DataFrame:
    """Statistically bound number of events given each of their probabilities, using Chernoff bounds.

    Args:
        individual_probabilities: A DataFrame of probabilities where each row
            represents an individual and each column represents an event.
        percent_confidence: The percent confidence of the two-sided intervals
            defined by the computed bounds.

    Raises:
        ValueError: If percent_confidence is outside of the interval (0, 1).

    Returns:
        A DataFrame containing, for each column in individual_probabilities,
        the expected number of events and interval bounds on the number of
        events.
    """
    if (percent_confidence <= 0) | (percent_confidence >= 1):
        raise ValueError("Percent confidence outside (0, 1)")
    means = individual_probabilities.transpose().to_numpy().sum(axis=1)
    cutoff = np.log((1 - percent_confidence) / 2) / means
    upper_bounds = (
        1 + ((-cutoff) + np.sqrt(np.square(cutoff) - (8 * cutoff))) / 2
    ) * means
    upper_bounds = np.minimum(upper_bounds, individual_probabilities.shape[0])
    lower_bounds = np.maximum((1 - np.sqrt(-2 * cutoff)) * means, 0)
    lower_bounds = np.maximum(lower_bounds, 0)
    conf = str(int(percent_confidence * 100))
    times = list(individual_probabilities)
    output = pd.DataFrame(
        {
            "Event": times,
            "ExpectedEventCount": means,
            "Lower" + conf + "PercentBound": lower_bounds,
            "Upper" + conf + "PercentBound": upper_bounds,
        }
    )
    return output


### New function using optimization

from scipy.optimize import minimize, minimize_scalar
from scipy.optimize import Bounds

percent_confidence = 0.95
individual_probabilities = forecasts


# https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

def compute_aggregation_uncertainty(
    individual_probabilities: pd.core.frame.DataFrame, percent_confidence: float = 0.95
) -> pd.core.frame.DataFrame:
    """Statistically bound number of predicted events given each of their probabilities, using Chernoff bounds.

    Args:
        individual_probabilities: A DataFrame of probabilities where each row
            represents an individual and each column represents an event.
        percent_confidence: The percent confidence of the two-sided intervals
            defined by the computed bounds.

    Raises:
        ValueError: If percent_confidence is outside of the interval (0, 1).

    Returns:
        A DataFrame containing, for each column in individual_probabilities,
        the expected number of events and interval bounds on the number of
        events.
    """
    if (percent_confidence <= 0) | (percent_confidence >= 1):
        raise ValueError("Percent confidence outside (0, 1)")
    means = individual_probabilities.transpose().to_numpy().sum(axis=1)
    alpha = (1 - percent_confidence) / 2


    def chernoff_upper_bounds (means: np.ndarray, alpha: float) -> np.ndarray:
        """Use optimizer to find the Chernoff upper bounds for given significance level"""

        def minimize_delta_function_ub(delta: float, mu: float, alpha: float):
            quantity = (((np.exp(delta)) / ((1 + delta) ** (1 + delta))) ** mu - alpha) ** 2


            if (delta <= 0):
                quantity = (quantity * 1e4) ** 2

            return quantity


        deltas_solved = np.array([])

        for mu in means:
            one_delta = minimize_scalar(minimize_delta_function_ub, args=(mu, alpha),
                 method="Bounded", bounds=(0, 3)).x
            deltas_solved = np.append(deltas_solved, one_delta)

        upper_bounds = (1 + deltas_solved) * means

        return upper_bounds


    def chernoff_lower_bounds (means: np.ndarray, alpha: float) -> np.ndarray:
        """Use optimizer to find the Chernoff lower bounds for given significance level"""


        def minimize_delta_function_lb(delta: float, mu: float, alpha: float):
            quantity = ((1 / (np.exp(delta) * ((1 - delta) ** (1 - delta)))) ** mu - alpha) ** 2
            if (delta <= 0):
                quantity = (quantity * 1e4) ** 2

            return quantity


        deltas_solved = np.array([])

        for mu in means:
            one_delta = minimize_scalar(minimize_delta_function_lb, args=(mu, alpha),
                 method="Bounded", bounds=(0, 1)).x
            deltas_solved = np.append(deltas_solved, one_delta)

        lower_bounds = (1 - deltas_solved) * means

        return lower_bounds



    upper_bounds = chernoff_upper_bounds(means, alpha)
    lower_bounds = chernoff_lower_bounds(means, alpha)

    upper_bounds = np.minimum(upper_bounds, individual_probabilities.shape[0])
    lower_bounds = np.maximum(lower_bounds, 0)
    conf = str(int(percent_confidence * 100))
    times = list(individual_probabilities)
    output = pd.DataFrame(
        {
            "Event": times,
            "ExpectedEventCount": means,
            "Lower" + conf + "PercentBound": lower_bounds,
            "Upper" + conf + "PercentBound": upper_bounds,
        }
    )
    return output


compute_aggregation_uncertainty(forecasts)