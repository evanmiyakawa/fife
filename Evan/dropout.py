import matplotlib.pyplot as plt
import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import keras
import sys, time, os, warnings
import numpy as np
import pandas as pd
from collections import Counter

import tensorflow.python.keras.backend as Kk

Kk.

warnings.filterwarnings("ignore")
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.visible_device_list = "4"
# set_session(tf.Session(config=config))

import numpy as np
w1, b1 = np.array([3,7]).reshape(1,2), np.array([5,1]).reshape(2)
w2, b2 = np.array([-10,10]).reshape(2,1), np.array([11]).reshape(1)
print("w1={},\nb1={},\nw2={},\nb2={}".format(w1,b1,w2,b2))
weights = ((w1,b1), (w2,b2))


def relu(x):
    x[x < 0] = 0
    return (x)


def NN2layer_dropout(x, weights, p_dropout=0.1):
    ## rate: float between 0 and 1. Fraction of the input units to drop.

    (w1, b1), (w2, b2) = weights
    ## dropout
    keep_prob = 1 - p_dropout
    Nh1 = w2.shape[0]
    Ikeep = np.random.binomial(1, keep_prob, size=Nh1)
    I = np.zeros((1, Nh1))
    I[:, Ikeep == 1] = 1 / keep_prob

    ## 2-layer NN with Dropout
    h1 = relu(np.dot(x, w1) + b1)
    y = np.dot(h1 * I, w2) + b2
    return (y[0, 0])

x = np.array([2]).reshape(1,1)

from collections import Counter, OrderedDict


def plot_histogram(ys, ax, title=""):
    counter = Counter(ys)
    counter = OrderedDict(sorted(counter.items()))
    probs = np.array(counter.values()) * 100.0 / np.sum(counter.values())
    keys = np.array(counter.keys())
    print(title)
    for k, p in zip(keys, probs):
        print("{:8.2f}: {:5.2f}%".format(k, p))
    n, bins, rect = ax.hist(ys)
    ax.set_title(title)
    ax.grid(True)


Nsim = 10
p_dropout = 0.7
ys_handmade = []
for _ in range(Nsim):
    ys_handmade.append(NN2layer_dropout(x, weights, p_dropout))

fig = plt.figure(figsize=(6, 2))
ax = fig.add_subplot(1, 1, 1)



import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense,Dropout

Nfeat = 1
tinput = Input(shape=(Nfeat,), name="ts_input")
h = Dense(2, activation='relu',name="dense1")(tinput)
hout = Dropout(p_dropout)(h, training = True)
out = Dense(1, activation="linear",name="dense2")(hout)

model = Model(inputs=[tinput], outputs=out)

model.summary()


for layer in model.layers:
    if layer.name == "dense1":
        layer.set_weights((w1,b1))
    if layer.name == "dense2":
        layer.set_weights((w2,b2))


class KerasDropoutPrediction(object):
    def __init__(self,model):
        self.f = K.function(
                [model.layers[0].input],
                [model.layers[-1].output])
        # self.f = K.function(
        #         [model.layers[0].input,
        #          K.learning_phase()],
        #         [model.layers[-1].output])
    def predict(self,x, n_iter=10):
        result = []
        for _ in range(n_iter):
            result.append(self.f([x , 1]))
        result = np.array(result).reshape(n_iter,len(x)).T
        return result

kdp = KerasDropoutPrediction(model)
result = kdp.predict(x,Nsim)
ys_keras = result.flatten()


t = [model.layers[0].input, K.learning_phase()]

K.function([model.layers[0].input,
                 K.learning_phase()],
                [model.layers[-1].output])
K.function([model.layers[0].input],
                [model.layers[-1].output])
K.learning_phase()


inputs = model.layers[0].input
outputs = model.layers[-1].output
model = keras.Model(inputs, outputs)



inputs2 = keras.Input(shape=(10,))
x2 = keras.layers.Dense(3)(inputs2)
outputs2 = keras.layers.Dropout(0.5)(x2, training=True)

model2 = keras.Model(inputs2, outputs2)

model.summary()
model2.summary()