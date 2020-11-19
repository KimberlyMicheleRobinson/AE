import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score

from DeepADoTS_master.src.evaluation import Evaluator
from agots.multivariate_generators.multivariate_data_generator import MultivariateDataGenerator
from DeepADoTS_master.src.algorithms.lstm_enc_dec_axl import LSTMED
from DeepADoTS_master.src.algorithms.gru_enc_dec_axl import GRUED
from DeepADoTS_master.src.algorithms.rnn_enc_dec_axl import RNNED
from io import StringIO
import csv
from sklearn.metrics import roc_auc_score



STREAM_LENGTH = 10000
N = 3
K = 0

dg = MultivariateDataGenerator(STREAM_LENGTH, N, K)
df = dg.generate_baseline(initial_value_min=-200, initial_value_max=200)

df = dg.add_outliers({'extreme': [{'n': 2, 'timestamps': [(i, ) for i in np.linspace(2100, 9999, 100, endpoint=True, dtype=int)],
                                   'factor': 100
                                   }]})

label = np.zeros(10000)
for i in np.linspace(2100, 9999, 100, endpoint=True, dtype=int) :
    label[i] = 1

df = df.drop(columns=['x0'])
df = df.drop(columns=['x1'])
print(df.shape)
for col in df.columns:
    plt.plot(df[col], label=col)
plt.legend()
plt.show()

df.corr()

# l2 = GRUED()
# l2.fit(df[:2500])
# p2 = l2.predict(df[2500:])
# print(p2.shape)
# plt.plot(p2)
# plt.legend()
# plt.show()
#
#
#
#
# l3 = RNNED()
# l3.fit(df[:2500])
# p3 = l3.predict(df[2500:])
# print(p3.shape)
# plt.plot(p3)
# plt.legend()
# plt.show()

#url = 'https://raw.githubusercontent.com/numenta/NAB/master/results/skyline/realTweets/skyline_Twitter_volume_AMZN.csv'
#data = pd.read_csv(url, sep=",")  # use sep="," for coma separation.
# data_test = pd.read_csv("E:\\ML\\Waveform Data\\COL 1 Time COL 2 Current\\TEK00014.csv", sep=",", header=None)
# print(data.shape)
# #data = np.loadtxt('chfdb_chf01_275.txt')
# data_train = data["value"].to_frame()
#
# label = data["label"].to_frame()
# print(data_train.shape)
# plt.plot(data_train)
# plt.show()
# #data = data.loc[:, [1, 2]]
# count = []
# print(data_train)
# for i in range(len(label)):
#     if label[i] == 1:
#         count.append(i)
# print(count)
# data = data.loc[:, [1]]
# data_test = data_test.loc[:, [1]]

# data.columns = ["time", "value"]
# data_test.columns = ["time", "value"]


# print(label)
# data = data.drop(columns=['Class'])
# print(data)
# time = data["timestamp"]

# print(x.shape)
# plt.plot(df)
# plt.show()
# plt.plot(data_test)
# plt.show()
#
l1 = LSTMED()
l1.fit(df[:2000])

p1, e1, o1 = l1.predict(df[2000:2500])
evaluate = Evaluator()
t = evaluate.get_optimal_threshold(y_test=label[2000:2500], score=p1)
print(t)
# plt.plot(o1)
# plt.show()
# plt.plot(e1)
# plt.show()
# plt.plot(p1)
# plt.show()
#
label_test = label[2500:]
#
p_test, e_test, o_test = l1.predict(df[2500:])
plt.plot(df[2500:])
plt.show()
plt.plot(o_test)
plt.show()
plt.plot(e_test)
plt.show()
plt.plot(p_test)
plt.show()
result_test = np.zeros(7500)
for i in range(7500):
    if p_test[i] >= t:
        result_test[i] = 1
auc_test = roc_auc_score(label_test, result_test)
print(auc_test)
# a = [y for y in data.index[data["label"] == 1].to_list() if y in I]
# print(p2)
# print(I)
# print(a)
# print(len(a)/len(data.index[data["label"] == 1].to_list()))
# print(len(a)/len(I))
#
# plt.plot(p2)
# plt.legend()
# plt.show()
