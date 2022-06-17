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
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch


# 画四分类的脑图


# 读取数据
A01T = loadmat("F:\\BCIIV2a\\train\\A01Tdata.mat")
A01E = loadmat("F:\\BCIIV2a\\test\\A01Edata.mat")

# loadmat() 得到的数据是字典类型，通过 key 将其中的数组提取出来
# 训练集
x_T = A01T['data']  # (288, 22, 1000)
y_T = A01T['label']  # (1, 288)
y_T = y_T.reshape(-1)
data = x_T
labels = y_T


# 创建一个info结构
ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3',
                 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
ch_sfreq = 250
info = mne.create_info(ch_names=ch_names, sfreq=ch_sfreq, ch_types='eeg')
info.set_montage('standard_1020')  #使用已有的电极国际10-20位置信息


# 创建事件
# 说明：事件信息是三列，第一列是事件对应的采样点，第二列是event id，第三列是对应trial的标签，标签0对应的是A
events = np.column_stack((np.arange(0,288*1000, 1000),
                          np.zeros(288,dtype=int),
                          labels))
event_dict = dict(condition_A=0, condition_B=1, condition_C=2, condition_D=3)
print(events)


# trial序列图
# 显示3个trial，每个通道都是显示完整的通道，通道的scale是20uF
simulated_epochs = mne.EpochsArray(data, info, tmin=-0.2, events=events, event_id=event_dict)
simulated_epochs.plot(picks='eeg', show_scrollbars=True, n_epochs=3, scalings=dict(eeg=20), events=events,
                      event_id=0)
plt.show()


# ERP混合图（erp曲线和所有trial erp的色彩图）
simulated_epochs['condition_A'].plot_image(picks='Cz', combine='mean')
simulated_epochs['condition_B'].plot_image(picks='Cz', combine='mean')
simulated_epochs['condition_C'].plot_image(picks='Cz', combine='mean')
simulated_epochs['condition_D'].plot_image(picks='Cz', combine='mean')


# erp曲线图
evocked=simulated_epochs['condition_A'].average()
evocked.plot(picks='C3')
evocked=simulated_epochs['condition_B'].average()
evocked.plot(picks='C4')
evocked.plot()
plt.show()


#erp对比曲线图
evokeds = dict(A=list(simulated_epochs['condition_A'].iter_evoked()),
               B=list(simulated_epochs['condition_B'].iter_evoked()),
               C=list(simulated_epochs['condition_C'].iter_evoked()),
               D=list(simulated_epochs['condition_D'].iter_evoked()))
mne.viz.plot_compare_evokeds(evokeds, combine='mean', picks='Cz')

#erp对比曲线图
evokeds = dict(A=list(simulated_epochs['condition_A'].iter_evoked()),
               B=list(simulated_epochs['condition_B'].iter_evoked()),
            )
mne.viz.plot_compare_evokeds(evokeds, combine='mean', picks='C3')

#erp对比曲线图
evokeds = dict(A=list(simulated_epochs['condition_A'].iter_evoked()),
               B=list(simulated_epochs['condition_B'].iter_evoked()),
            )
mne.viz.plot_compare_evokeds(evokeds, combine='mean', picks='C4')

# 计算全局场功率
simulated_epochs.plot_image(combine='gfp', sigma=2., cmap="YlGnBu_r")


# 电极位置图
simulated_epochs.plot_sensors(ch_type='eeg',show_names='True')
plt.show()


# psd图
simulated_epochs['condition_A'].plot_psd(fmin=2., fmax=40., average=True, spatial_colors=False)
plt.show()
simulated_epochs['condition_B'].plot_psd(fmin=2., fmax=40., average=True, spatial_colors=False)
plt.show()
simulated_epochs['condition_C'].plot_psd(fmin=2., fmax=40., average=True, spatial_colors=False)
plt.show()
simulated_epochs['condition_D'].plot_psd(fmin=2., fmax=40., average=True, spatial_colors=False)
plt.show()


# psd头皮图（分频段）
simulated_epochs['condition_A'].plot_psd_topomap(ch_type='eeg', normalize=True)
plt.show()
simulated_epochs['condition_B'].plot_psd_topomap(ch_type='eeg', normalize=True)
plt.show()
simulated_epochs['condition_C'].plot_psd_topomap(ch_type='eeg', normalize=True)
plt.show()
simulated_epochs['condition_D'].plot_psd_topomap(ch_type='eeg', normalize=True)
plt.show()


# 能量时频图和头皮图
freqs = np.arange(3,40,1)
power, itc = tfr_morlet(simulated_epochs['condition_A'], freqs=freqs, n_cycles=2,
                        return_itc=True, decim=1)

power.plot([2], baseline=(-0.2, 0), mode='logratio', title=power.ch_names[2])

fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power.plot_topomap(ch_type='eeg', tmin=0.2, tmax=0.5, fmin=4, fmax=8,
                   baseline=(-0.2, 0), mode='logratio', axes=axis[0],
                   title='4-8', show=False)
power.plot_topomap(ch_type='eeg', tmin=0.2, tmax=0.5, fmin=8, fmax=12,
                   baseline=(-0.2, 0), mode='logratio', axes=axis[1],
                   title='8-12', show=False)
mne.viz.tight_layout()
plt.show()


# 叠加平均
# 以 condition_A 为例
evoked1 = simulated_epochs['condition_A'].average()  # 数据叠加平均
evoked1.plot()  # 绘制逐导联的时序信号图
evoked1.plot_joint()  # 绘制联合图
evoked1.plot_image()  # 绘制逐导联热力图

evoked1.plot_topo()  # 绘制拓扑时序信号图

mne.viz.plot_compare_evokeds(evokeds=evoked1, combine='mean')  # 绘制平均所有电极后的ERP

