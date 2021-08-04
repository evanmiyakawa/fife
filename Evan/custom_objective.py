import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import load_breast_cancer
import seaborn as sns

def l2_loss(y, data):
    t = data.get_label()
    grad = y - t
    hess = np.ones_like(y)
    return grad, hess

def l2_eval(y, data):
    t = data.get_label()
    loss = (y - t) ** 2
    return 'l2', loss.mean(), False

def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _negative_sigmoid(x):
    exp = np.exp(x)
    return exp / (exp + 1)

def sigmoid(x):
    positive = x >= 0
    negative = ~positive
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])
    return result


def _positive_softplus(x):
    return x + np.log1p(np.exp(-x))

def _negative_softplus(x):
    return np.log1p(np.exp(x))

def softplus(x):
    positive = x >= 0
    negative = ~positive
    result = np.empty_like(x)
    result[positive] = _positive_softplus(x[positive])
    result[negative] = _negative_softplus(x[negative])
    return result


def bce_loss(z, data):
    t = data.get_label()
    y = sigmoid(z)
    grad = y - t
    hess = y * (1 - y)
    return grad, hess

def bce_eval(z, data):
    t = data.get_label()
    loss = t * softplus(-z) + (1 - t) * softplus(z)
    return 'bce', loss.mean(), False


def custom_asymmetric_train(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual<0, -2*10.0*residual, -2*residual)
    hess = np.where(residual<0, 2*10.0, 2.0)
    return grad, hess


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
X_train, X_test, y_train, y_test = train_test_split(df, data.target, random_state=0)
lgb_train = lgb.Dataset(X_train, y_train)

lgbm_params = {
    # 'objective': 'binary',
    'objective': 'custom_asymmetric_train',
    'random_seed': 0,
    }
model = lgb.train(lgbm_params,
                  lgb_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred>0.5))
# >> 0.972027972027972

lgbm_params = {
    'random_seed': 0
    }
model = lgb.train(lgbm_params,
                  lgb_train,
                  fobj=bce_loss)#,
                  # feval=bce_eval)
model = lgb.train(lgbm_params,
                  lgb_train)

# Note: When using custom objective, the model outputs logits
y_pred2 = sigmoid(model.predict(X_test))
print(accuracy_score(y_test, y_pred2>0.5))
# >> 0.972027972027972

np.var(y_pred)
np.var(y_pred2)





### for regression

df = sns.load_dataset('tips')
X_train, X_test, y_train, y_test = train_test_split(df.drop(['total_bill'], axis=1), df['total_bill'], random_state=0)
lgb_train = lgb.Dataset(X_train, y_train)

# Using built-in objective
lgbm_params = {
    'objective': 'regression',
    'random_seed': 0
    }
model = lgb.train(lgbm_params,
                  lgb_train)
y_pred = model.predict(X_test)
print(mean_squared_error(y_test, y_pred))
# >> 31.94812324773213

# Using custom objective
lgbm_params = {
    'random_seed': 0
    }

model = lgb.train(lgbm_params,
                  lgb_train,
                  fobj=l2_loss,
                  feval=l2_eval)
y_pred = model.predict(X_test)
print(mean_squared_error(y_test, y_pred))
# >> 31.947398098316526











from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np

x = np.array([-2.2, -1.4, -.8, .2, .4, .8, 1.2, 2.2, 2.9, 4.6])
y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

logr = LogisticRegression(solver='lbfgs')
logr.fit(x.reshape(-1, 1), y)

y_pred = logr.predict_proba(x.reshape(-1, 1))[:, 1].ravel()
loss = log_loss(y, y_pred)

print('x = {}'.format(x))
print('y = {}'.format(y))
print('p(y) = {}'.format(np.round(y_pred, 2)))
print('Log Loss / Cross Entropy = {:.4f}'.format(loss))

np.exp(0.2)/(1 + np.exp(0.2))