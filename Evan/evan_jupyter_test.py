## Run default jupyter tutorial ##

from fife.lgb_modelers import LGBSurvivalModeler
from fife.processors import PanelDataProcessor
from fife.utils import make_results_reproducible
from pandas import concat, date_range, read_csv, to_datetime

import warnings
warnings.filterwarnings('ignore')

data = read_csv("https://www.dl.dropboxusercontent.com/s/3tdswu2jfgwp4xw/REIGN_2020_7.csv?dl=0")
print(data.head())


data["country-leader"] = data["country"] + ": " + data["leader"]
data["year-month"] = data["year"].astype(int).astype(str) + data["month"].astype(int).astype(str).str.zfill(2)
data["year-month"] = to_datetime(data["year-month"], format="%Y%m")
data = concat([data[["country-leader", "year-month"]],
               data.drop(["ccode", "country-leader", "leader", "year-month"],
                         axis=1)],
               axis=1)
data.head()

total_obs = len(data)
data = data.drop_duplicates(["country-leader", "year-month"], keep="first")
n_duplicates = total_obs - len(data)
print(f"{n_duplicates} observations with a duplicated identifier pair deleted.")

data_processor = PanelDataProcessor(data=data)
data_processor.build_processed_data()
data_processor.data.head()

modeler = LGBSurvivalModeler(data=data_processor.data);
modeler.build_model(parallelize=False);

forecasts = modeler.forecast()
custom_forecast_headers = date_range(data["year-month"].max(),
                                     periods=len(forecasts.columns) + 1,
                                     freq="M").strftime("%b %Y")[1:]
forecasts.columns = custom_forecast_headers
forecasts