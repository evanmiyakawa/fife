import pandas as pd

x = np.array(np.random.uniform())
x = np.empty(shape = 10, dtype = "float")

for i in range(10):
    x[i] = np.random.uniform()

x = pd.Series(np.arange(0,1,0.2))
x

x.quantile(0.05)
x.quantile(0.50)

x[2] = 0.45
x
