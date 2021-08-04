
from tests_performance import *
from tests_performance.Data_Fabrication import *

df1 = fabricate_data(N_PERSONS=1000, N_PERIODS=20, SEED=1234, exit_prob=0.3, dgp=1)

# ID is unique identifier for individual
# Period is the time period for the observation
# X1 is categorical variable that affects the value of exit_type based on selected dgp
# X2 and X3 are continuous variables that don't affect anything
# exit_type is the response, categorical describing the category of exit
# probability of exit

df1[df1['exit_type'] == "X"]

df1[31:60]


