"""
Work in progress
Taken from https://github.com/iuype/TRCA-SSVEP/blob/master/Main.py
"""

# '''
# Author:
#     Yu Pei, 1666424499@qq.com
# Versions:
#     v1.0: 2019-12-12,
#     V1.1: 2019-12-15, fix the bug ： det(Q) = 0
# '''
# import os
# import sys
# import argparse
# from scipy.io import loadmat
# import glob
# import numpy as np
# from sklearn.model_selection import train_test_split
# from scipy import signal
# '''
# (1).高通滤波
# #这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除10hz以下频率成分，即截至频率为10hz，则wn=2*10/1000=0.02
# from scipy import signal
# b, a = signal.butter(8, 0.02, 'highpass')
# filtedData = signal.filtfilt(b, a, data)#data为要过滤的信号
# (2).低通滤波
# #这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除10hz以上频率成分，即截至频率为10hz，则wn=2*10/1000=0.02
# from scipy import signal
# b, a = signal.butter(8, 0.02, 'lowpass') 
# filtedData = signal.filtfilt(b, a, data)       #data为要过滤的信号
 
# (3).带通滤波
# #这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除10hz以下和400hz以上频率成分，即截至频率为10hz和400hz,则wn1=2*10/1000=0.02,wn2=2*400/1000=0.8。Wn=[0.02,0.8]
# from scipy import signal
# b, a = signal.butter(8, [0.02,0.8], 'bandpass')
# filtedData = signal.filtfilt(b, a, data)   #data为要过滤的信号
# ————————————————
# 版权声明：本文为CSDN博主「John-Cao」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/weixin_37996604/article/details/82864680
# '''

# class TRCA():
#     def __init__(self, opt):
#         self.opt = opt
#         self.channels       = ["POz", "PO3", "PO4", "PO5", "PO6", "Oz", "O1", "O2"]
#         self.sample_rate    = 1000      # 1000Hz
#         self.downsample_rate= 250       # 250Hz

#         self.latency        = 0.14      # [0.14s, 0.14s + d]
#         self.data_length    = self.opt.data_length      # d , data length

#         self.traindata      = None
#         self.trainlabel     = None
#         self.testdata       = None
#         self.testlabel      = None

#         self.Nm             = self.opt.Nm
#         self.Nf             = self.opt.Nf
#         self.Nc             = len(self.channels)
#         self.X_hat          = np.zeros((self.Nm, self.Nf, self.Nc, int(self.downsample_rate * self.data_length)))    # (Nm, Nf, Nc, data_length)

#         self.W = np.zeros((self.Nm, self.Nf, len(self.channels), 1))    # (Nm, Nf, Nc, 1)

#         self.w1             = [2*(m+1)*8/self.downsample_rate for m in range(self.Nm)]
#         self.w2             = [2 * 90 / self.downsample_rate for m in range(self.Nm)]


#     def load_data(self, dataroot = None):
#         if dataroot is None:
#             print("dataroot error -->: ",dataroot)

#         datapath = glob.glob(os.path.join(dataroot,"xxrAR","*"))    # ['.\\datasets\\xxrAR\\EEG.mat']

#         oEEG = loadmat(datapath[0])

#         """
#         data 数据格式(trials, filter_bank, channals, timep_oints)  --> (160, Nm, Nf, 0.5s * 采样率)
#         """
#         self.traindata, self.testdata, self.trainlabel, self.testlabel = self.segment(oEEG)

#     def segment(self , oEEG):
#         EEG = oEEG["EEG"]
#         data = EEG["data"][0, 0]
#         event = EEG["event"][0, 0]

#         chanlocs = EEG["chanlocs"][0,0]

#         channels_idx = []

#         for i in range(chanlocs.shape[0]):
#             if chanlocs[i,0][0][0] in self.channels:
#                 channels_idx.append(i)

#         all_data = np.zeros(
#             (
#                 160,   # 样本个数
#                 self.Nc,      # 通道数
#                 int(self.downsample_rate*self.data_length) # 数据长度
#             )
#         )

#         all_label = np.zeros((160, 1))   # [0,1,2,...,Nf] 取值范围

#         for idx in range(event.shape[0]):
#             lb = int(event["type"][idx, 0][0]) - 1
#             lat = event["latency"][idx, 0][0][0]

#             # 原始数据为 1000hz 采样 ， 根据原文，需要降采样到 250hz 。
#             all_data[idx] = data[channels_idx, int(self.latency* self.sample_rate + lat ) : int(self.latency* self.sample_rate + lat + self.data_length * self.sample_rate )][:,::4]
#             all_label[idx,0] = lb

#         all_data = self.filerbank(all_data)

#         # 9 ： 1 分割训练集与测试集
#         train_X, test_X, train_y, test_y = train_test_split(all_data,
#                                                             all_label,
#                                                             test_size = 0.2,
#                                                             random_state = 0)
#         return train_X, test_X, train_y, test_y

#     '''
#     这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除10hz以下和400hz以上频率成分，即截至频率为10hz和400hz,则wn1=2*10/1000=0.02,wn2=2*400/1000=0.8。Wn=[0.02,0.8]
#     b, a = signal.butter(8, [0.02,0.8], 'bandpass')
#     filtedData = signal.filtfilt(b, a, data)   #data为要过滤的信号
#     '''
#     def filerbank(self, data):
#         data_filterbank = np.zeros((data.shape[0], self.Nm, len(self.channels), data.shape[2]))

#         for i in range(self.Nm):
#             # 8 是滤波器的阶数， 不确定用哪个。。
#             # print([self.w1[i], self.w2[i]])
#             b, a = signal.butter(self.opt.filter_order, [self.w1[i], self.w2[i]], 'bandpass')
#             data_filterbank[:, i, :, :] =  signal.filtfilt(b, a, data, axis= -1)

#         return data_filterbank

#     def Cov(self,X,Y):

#         X = X.reshape(-1, 1)
#         Y = Y.reshape(-1, 1)

#         # print(X.shape, Y.shape)

#         X_hat = np.mean(X)
#         Y_hat = np.mean(Y)

#         X = X - X_hat
#         Y = Y - Y_hat

#         ret = np.dot(X.T ,Y)
#         ret /= (X.shape[0])

#         return ret


#     def fit(self):

#         S = np.zeros((self.Nm, self.Nf, self.Nc, self.Nc))
#         Q = np.zeros((self.Nm, self.Nf, self.Nc, self.Nc))

#         # S
#         for m in range(self.Nm):
#             for n in range(self.Nf):
#                 idxs = [] # stimulus n 的索引
#                 for i in range(self.traindata.shape[0]):
#                     if self.trainlabel[i, 0] == n:
#                         idxs.append(i)
#                 for j1 in range(self.Nc):
#                     for j2 in range(self.Nc):
#                         for h1 in idxs:
#                             for h2 in idxs:
#                                 if h1 != h2:
#                                     S[m, n, j1, j2] += self.Cov(self.traindata[h1, m, j1, :], self.traindata[h2, m, j2, :])
#                 # print(S[m,n])  # 检查 S是对称的，没有问题

#         # Q
#         for m in range(self.Nm):
#             for n in range(self.Nf):
#                 idxs = [] # stimulus n 的索引
#                 for i in range(self.traindata.shape[0]):
#                     if self.trainlabel[i, 0] == n:
#                         idxs.append(i)
#                 for h in idxs:
#                     for j1 in range(self.Nc):
#                         for j2 in range(self.Nc):
#                             Q[m, n, j1, j2] += self.Cov(self.traindata[h, m, j1, :], self.traindata[h, m, j2, :])
#                 Q[m, n] /= len(idxs)
#                 # print(Q[m, n]) # 发现bug det(Q) = 0 ... 我日

#         for m in range(self.Nm):
#             for n in range(self.Nf):

#                 e_vals, e_vecs = np.linalg.eig(np.linalg.inv(Q[m, n]).dot(S[m, n]))

#                 max_e_vals_idx = np.argmax(e_vals)

#                 self.W[m, n, :, 0] = e_vecs[:, max_e_vals_idx]

#         # calculate hat
#         for m in range(self.Nm):
#             for n in range(self.Nf):
#                 idxs = [] # stimulus n 的索引
#                 for i in range(self.traindata.shape[0]):
#                     if self.trainlabel[i, 0] == n:
#                         idxs.append(i)

#                 for h in idxs:

#                     self.X_hat[m,n] += self.traindata[h, m]  # （8, 125）

#                 self.X_hat[m, n] /= len(idxs)


#         tot = 0
#         tot_correct = 0

#         for i in range(self.testdata.shape[0]):
#             pre_lb , lb = self.inference(self.testdata[i]),self.testlabel[i,0]
#             if pre_lb == lb:
#                 tot_correct += 1
#             tot += 1

#         print(tot_correct / tot)


#         tot = 0
#         tot_correct = 0

#         for i in range(self.traindata.shape[0]):
#             pre_lb , lb = self.inference(self.traindata[i]),self.trainlabel[i,0]
#             if pre_lb == lb:
#                 tot_correct += 1
#             tot += 1

#         print(tot_correct / tot)


#     def inference(self, X): # (Nm ,Nc, data_length)
#         r = np.zeros((self.Nm, self.Nf))

#         for m in range(self.Nm):
#             for n in range(self.Nf):
#                 r[m, n] = self.pearson_corr_1D(X[m].T.dot(self.W[m, n]), self.X_hat[m, n].T.dot(self.W[m, n]))

#         Pn = np.zeros(self.Nf)
#         for n in range(self.Nf):
#             for m in range(self.Nm):
#                 Pn[n] += ((m+1) ** (-1.25) + 0.25 ) * (r[m, n] ** 2)

#         pre_label = np.argmax(Pn)

#         return pre_label

#     def pearson_corr_1D(self, a, b):
#         a = a.reshape(-1)
#         b = b.reshape(-1)
#         ret = self.Cov(a,b) / (np.std(a) * np.std(b))
#         return ret


#     def pearson_corr_2D(self, a, b):
#         """
#         todo
#         2维皮尔逊相关系数
#         两个变量之间的皮尔逊相关系数定义为两个变量之间的协方差和标准差的商
#         """
#         return 0.5

#     def __del__(self):
#         pass




# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--epochs", type=int, default=100, help="number of epochs")

#     parser.add_argument("--dataroot", type=str, default=os.path.join(".", "datasets"), help="the folder of data")
#     parser.add_argument("--filter_order", type=int, default=8, help="order of filter")
#     parser.add_argument("--Nm", type=int, default = 7, help="number of bank")
#     parser.add_argument("--data_length", type=float, default=0.5, help="task time points")
#     parser.add_argument("--Nf", type=int, default=8, help="number of stimulus")

#     opt = parser.parse_args()
#     print(opt)

#     trca = TRCA(opt)
#     trca.load_data(opt.dataroot)
#     trca.fit()

#     print("done!")