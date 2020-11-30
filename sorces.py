import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score
import torch
from DeepADoTS_master.src.evaluation import Evaluator
from agots.multivariate_generators.multivariate_data_generator import MultivariateDataGenerator
from DeepADoTS_master.src.algorithms.lstm_enc_dec_axl_2 import LSTMED
from DeepADoTS_master.src.algorithms.gru_enc_dec_axl import GRUED
from DeepADoTS_master.src.algorithms.rnn_enc_dec_axl import RNNED
from io import StringIO
import csv
from sklearn.metrics import roc_auc_score



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



# data = pd.read_table("E:\\ML\\BA\\data\\kddcup.data.corrected.txt", sep=',', header=None)
# data.columns = [
#     'duration',
#     'protocol_type',
#     'service',
#     'flag',
#     'src_bytes',
#     'dst_bytes',
#     'land',
#     'wrong_fragment',
#     'urgent',
#     'hot',
#     'num_failed_logins',
#     'logged_in',
#     'num_compromised',
#     'root_shell',
#     'su_attempted',
#     'num_root',
#     'num_file_creations',
#     'num_shells',
#     'num_access_files',
#     'num_outbound_cmds',
#     'is_host_login',
#     'is_guest_login',
#     'count',
#     'srv_count',
#     'serror_rate',
#     'srv_serror_rate',
#     'rerror_rate',
#     'srv_rerror_rate',
#     'same_srv_rate',
#     'diff_srv_rate',
#     'srv_diff_host_rate',
#     'dst_host_count',
#     'dst_host_srv_count',
#     'dst_host_same_srv_rate',
#     'dst_host_diff_srv_rate',
#     'dst_host_same_src_port_rate',
#     'dst_host_srv_diff_host_rate',
#     'dst_host_serror_rate',
#     'dst_host_srv_serror_rate',
#     'dst_host_rerror_rate',
#     'dst_host_srv_rerror_rate',
#     'outcome'
# ]
# smtp = data.loc[data["service"] == "smtp"]
#
# smtp.to_csv('smtp.csv', index=None)
label = []
data = pd.read_csv("E:\\ML\\smtp.csv", sep=",")
for i in data["outcome"]:
    if i == "normal.":
        label.append(0)
    else:
        label.append(1)
data["outcome"] = label
print(data.loc[data["outcome"] == 1])

# data = pd.read_csv("E:\\ML\\BA\\data\\stmpRowDatatest.csv",
#                     sep=",", header=None)  # use sep="," for coma separation.

print(data.shape)
# label = pd.read_csv("E:\\ML\\label.csv",
#                    header=None)
label = data['outcome']
data_train = data.drop(columns=['outcome'])
print(data_train)
data_train = (data_train - data_train.mean(axis=0))/data_train.std(axis=0)
train = data_train[:67000]


# 自己建立窗口
sequences = []
win = train.values
for i in range(win.shape[0] - 30 + 1):
    sequences.append(train[i:i + 30])

index = []
i = 0
for l in label[:66971]:
    if l == 1:
        for j in range(30):
            index.append(i+j)

    i = i+1
print(index)
index = np.unique(index)

np.set_printoptions(threshold=np.inf)
print(index)
sequences = np.delete(sequences, index)


# label.columns = ['class']
# print(data.head(10))
# print(label.loc[label['class'] == 1])


predict = data_train[67000:77000]
import json

with open("E:\\ML\\DeepADoTS_master\\src\\algorithms\\paramter.json", 'r') as load_f:
    load_dict = json.load(load_f)
    print(load_dict["num_epochs"])
    print(load_dict)

model = LSTMED(num_epochs=load_dict["num_epochs"], batch_size=load_dict["batch_size"],
                 hidden_size=load_dict["hidden_size"], sequence_length=load_dict["sequence_length"])

model.fit(train, sequences)
#self.lstmed.load_state_dict(torch.load('checkpoint.pt'))
#model.fit(train_4)
score, error, output = model.predict(predict)
plt.figure(figsize=(50, 5))  # 图像大小
x = np.arange(10000)
y2 = score
plt.plot(x, y2)
plt.legend(['Anomaly score'])
plt.ylabel('Value')
plt.xlabel('Data')
plt.savefig('AnomalyScore_10001_20001.svg',format='svg')
plt.show()
from DeepADoTS_master.src.evaluation import Evaluator

evaluate = Evaluator()
t = evaluate.get_optimal_threshold(y_test=label[67000:77000], score=score)
print(t)

predict3 = data_train[77000:]
score3, error3, output3 = model.predict(predict3)
plt.figure(figsize=(50, 5))  # 图像大小
x = np.arange(len(score3))
y1 = predict3
y2 = score3
y3 = output3
plt.plot(y2)
plt.show()
plt.plot(y3)
plt.show()
plt.plot(error3)
plt.show()
# plt.legend(['rowData', 'output'])
# plt.ylabel('Value')
# plt.xlabel('Data (transform to original dataTime should add 20003)')
# plt.savefig('rowData_20002_andOutput.svg', format='svg')
# plt.show()

y1 = score3
plt.plot(x, y1)
plt.axhline(y=t, color='r', linestyle='-')
plt.legend(['score'])
plt.ylabel('Value')
plt.xlabel('Data')
plt.savefig('withThreshold.svg', format='svg')
plt.show()

# pd.DataFrame(score3).to_csv('E:\\ML\\BA\\result\\LSTM\\score_newsmtp_Lstm.csv', index=True)
# pd.DataFrame(error3).to_csv('E:\\ML\\BA\\result\\LSTM\\error_newsmtp_Lstm.csv', index=True)
# pd.DataFrame(output3).to_csv('E:\\ML\\BA\\result\\LSTM\\output_newsmtp_Lstm.csv', index=True)

anomolies = []
for i in np.arange(len(score3)):
    if score3[i] > t:
        anomolies.append(i + 77002)
print(len(anomolies))

result_test = np.zeros(len(score3))
for i in range(len(score3)):
    if score3[i] > t:
        result_test[i] = 1

from sklearn.metrics import f1_score
auc = roc_auc_score(label[77000:], result_test)
f1_score = f1_score(label[77000:], result_test)
print("Auc : " + str(auc))
print("F1_Score : " + str(f1_score))