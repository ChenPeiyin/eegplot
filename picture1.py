import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from scipy.io import loadmat
import numpy as np
from scipy import signal
import mne
import matplotlib.pyplot as plt
from mne.viz import plot_montage, plot_topomap
import copy
from scipy.fftpack import fft


# 画四分类的脑图

# 定义超参数
batch_size = 64
learning_rate = 0.001
num_epoches = 600

C = 22            # 脑电通道数
T = 1000         # 采样个数（采样频率*时间）
num_classes = 4   # 分类数
p = 0.5           # Dropout参数

# 读取数据
A01T = loadmat("F:\\BCIIV2a\\train\\A04Tdata.mat")
A01E = loadmat("F:\\BCIIV2a\\test\\A04Edata.mat")

# loadmat() 得到的数据是字典类型，通过 key 将其中的数组提取出来

# 训练集
x_T = A01T['data']  # (288, 22, 1000)
y_T = A01T['label']  # (1, 288)
y_T = y_T.reshape(-1)

# 测试集
x_E = A01E['data']  # (288, 22, 1000)
y_E = A01E['label']  # (1, 288)
y_E = y_E.reshape(-1)

# 分成4类
a = x_T[y_T == 0]
b = x_T[y_T == 1]
c = x_T[y_T == 2]
d = x_T[y_T == 3]  # (72, 22, 1000)


# 提取PSD功率谱密度特征


def PSD_extract(data, fs, num_fft):
    """
    data: 用来做傅里叶变换的数组
    num_fft: 傅里叶变换的长度
    """
    psd_fea = []
    chan = data.shape[0]
    length = data.shape[1]
    for i in range(chan):
        chan_data = data[i, :]

        # FFT
        Y = np.abs(fft(chan_data.reshape(-1), num_fft))
        Y = Y ** 2 / num_fft

        ps = np.mean(Y)
        psd_fea.append(ps)
    fea = np.array(psd_fea)
    return fea


data_a = np.mean(a, axis=0)  # (22, 1000)
data_b = np.mean(b, axis=0)  # (22, 1000)
data_c = np.mean(c, axis=0)  # (22, 1000)
data_d = np.mean(d, axis=0)  # (22, 1000)

a_psd = PSD_extract(data=data_a, fs=250, num_fft=1024)    # (22)
b_psd = PSD_extract(data=data_b, fs=250, num_fft=1024)    # (22)
c_psd = PSD_extract(data=data_c, fs=250, num_fft=1024)    # (22)
d_psd = PSD_extract(data=data_d, fs=250, num_fft=1024)    # (22)

# 创建一个info结构
ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3',
                 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
ch_sfreq = 250
info = mne.create_info(ch_names=ch_names, sfreq=ch_sfreq, ch_types='eeg')


a_evoked = a_psd[:, np.newaxis]   # (22, 1)
b_evoked = b_psd[:, np.newaxis]
c_evoked = c_psd[:, np.newaxis]
d_evoked = d_psd[:, np.newaxis]

# 给evoked起一个名称
a_comment = "Left hand"
b_comment = "Right hand"
c_comment = "Feet"
d_comment = "Tongue"

# 利用mne.EvokedArray创建Evoked对象

a_evoked_array = mne.EvokedArray(a_evoked, info, comment=a_comment, nave=22)
b_evoked_array = mne.EvokedArray(b_evoked, info, comment=b_comment, nave=22)
c_evoked_array = mne.EvokedArray(c_evoked, info, comment=c_comment, nave=22)
d_evoked_array = mne.EvokedArray(d_evoked, info, comment=d_comment, nave=22)

# 为evoked数据设置电极位置信息
montage=mne.channels.make_standard_montage('standard_1020')

a_evoked_array.set_montage(montage, on_missing = 'raise', verbose = None)
b_evoked_array.set_montage(montage, on_missing = 'raise', verbose = None)
c_evoked_array.set_montage(montage, on_missing = 'raise', verbose = None)
d_evoked_array.set_montage(montage, on_missing = 'raise', verbose = None)

# print(a_evoked_array)

# evoked_array.animate_topomap()

# mne.viz.plot_topomap(evoked_array.data[:, 0], evoked_array.info,show=True)

fig, ax = plt.subplots(1, 4, figsize=(8, 3))
mne.viz.plot_topomap(a_evoked_array.data[:, 0], a_evoked_array.info, names=ch_names, show_names=False,
             outlines='head', show=False, cmap='jet', axes=ax[0])
mne.viz.plot_topomap(b_evoked_array.data[:, 0], b_evoked_array.info, names=ch_names, show_names=False,
             outlines='head', show=False, cmap='jet', axes=ax[1])
mne.viz.plot_topomap(c_evoked_array.data[:, 0], c_evoked_array.info, names=ch_names, show_names=False,
             outlines='head', show=False, cmap='jet', axes=ax[2])
mne.viz.plot_topomap(d_evoked_array.data[:, 0], d_evoked_array.info, names=ch_names, show_names=False,
             outlines='head', show=False, cmap='jet', axes=ax[3])

for ax, title in zip(ax[:4], ['Left hand', 'Right hand', 'Feet', 'Tongue']):
    ax.set_title(title)
plt.show()

im, cn = mne.viz.plot_topomap(d_evoked_array.data[:, 0], a_evoked_array.info, names=ch_names, show_names=False,
             outlines='head', show=False, cmap='jet', vmin=None, vmax=None)
plt.colorbar(im)
plt.show()

# a_evoked_array.animate_topomap()

# im, cn = mne.viz.plot_topomap(d_evoked_array.data[:, 0], a_evoked_array.info, names=ch_names, show_names=False,
     #        outlines='head', show=False, cmap='jet', vmin=None, vmax=None)
# plt.colorbar(im, ax=ax[4])

# plt.figure()


#a_evoked_array.plot_sensors(ch_type='eeg',show_names='True')
#plt.show()