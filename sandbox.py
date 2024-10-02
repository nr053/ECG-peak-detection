import pickle
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import wfdb
from utils.helper_functions import return_good_ecg_channel_idx_based_on_lead_off
import mne
import numpy as np
import argparse
from utils.VAF_loader import VAF_loading
from utils.db_loader import DB_loading
from utils.db_generator import Train_Generator
import yaml
import pywt
from scipy import signal
import matplotlib.pyplot as plt



### filtering method
def dwt_idwt(array, wavelet='db3', level=9):
    coeffs = pywt.wavedec(array, wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])
    array_filtered = pywt.waverec(coeffs, wavelet)
    return array_filtered

def lowpass_filter(array, fs, cutoff=40, order=5, remove_lag=True):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(order, normal_cutoff, btype='low', output='sos')
    array_filtered = signal.sosfilt(sos, array)
    if remove_lag == True:
        # filter time delay -> to reduce zero lag
        array_filtered = signal.sosfilt(sos, array_filtered[::-1])
        return array_filtered[::-1][:len(array)]
    else:
        return array_filtered[:len(array)]

def filtering(array, fs=360):
    array_filtered = array.copy()
    array_filtered = dwt_idwt(array_filtered)
    array_filtered = lowpass_filter(array_filtered, fs)
    return array_filtered

### resampling
def resample_ecg(array, fs_pre, fs_post=360):
    t_len = len(array)
    new_len = int(fs_post/fs_pre*t_len)
    array_resampled = signal.resample(array, new_len)
    array_resampled = filtering(array_resampled)
    return array_resampled


### transformation
def normalization(array):
    root_squared_mean = np.mean(array**2)**0.5
    array_norm = array/root_squared_mean/16
    return array_norm

def sw_transform(array):
    len_padding = -len(array)%2**4
    padded = np.pad(array, (0, len_padding), 'edge')
    coeff_swt = pywt.swt(padded, 'sym4', level=4, trim_approx=True)
    # [cAn, cDn, ..., cD2, cD1]
    coeff_swt.reverse()
    # [cD1, cD2, ... ,cDn, cAn]

    feature = coeff_swt[3]
    feature = feature[:len(array)]
    feature = normalization(feature)

    diff = np.diff(array, append=array[-1])
    diff = normalization(diff)

    merged_feature = np.stack([feature, diff], axis=1)
    return merged_feature

def transform(array, use_swt=True):
    if use_swt == False:
        diff = np.diff(array, append=array[-1])
        diff = normalization(diff)
        feature = diff.reshape(-1, 1)
        return feature
    else:
        feature = sw_transform(array)
    return feature

def append_dicts(dict1, dict_list:[]):
    for dict in dict_list:    
        for key in dict:
            for item in dict[key]:
                dict1[key].append(item)
    return dict1

#vaf
#with open("/home/rose/Cortrium/R-peak-detector/VAF_subset/test/00a4aad0-5d1b-4456-a6f0-8e5c13d0feda.pkl", 'rb') as file:
#    vaf = pickle.load(file)


#edf
#raw = mne.io.read_raw_edf("/home/rose/Cortrium/ECG-peak-detection/edf1.edf", preload=True)
#data = raw.get_data()


# other files
record = wfdb.rdsamp("/home/rose/Cortrium/Databases/MIT_BIH/100")
ecg = record[0][:, 1]
fs = record[1]['fs']
# load annotation
ann = wfdb.rdann("/home/rose/Cortrium/Databases/MIT_BIH/100", 'atr')


# other files
record_bw = wfdb.rdsamp("/home/rose/Cortrium/Databases/MIT_BIH_NST/bw")
record_ma = wfdb.rdsamp("/home/rose/Cortrium/Databases/MIT_BIH_NST/ma")
record_em = wfdb.rdsamp("/home/rose/Cortrium/Databases/MIT_BIH_NST/em")
ecg_bw = np.swapaxes(record_bw[0],0,1)[0,:2520]
ecg_ma = np.swapaxes(record_ma[0],0,1)[0,:2520]
ecg_em = np.swapaxes(record_em[0],0,1)[0,:2520]

data_train = DB_loading()
dict_incart = data_train.create_set("INCART", train=True)
dict_mit = data_train.create_set("MIT_BIH", train=True)
dict_qt = data_train.create_set("QTDB", train=True)
dict_train = append_dicts(dict_incart,  [dict_mit, dict_qt])
dataloader_train = Train_Generator(dict_train)

f1 = np.swapaxes(dict_train["feature"][500], 0, 1)[0]
f2 = np.swapaxes(dict_train["feature"][500], 0, 1)[1]

noise = np.random.normal(0,0.01,2520)
noise_filtered = filtering(noise)
noise_feature = transform(noise_filtered)
noise_feature = np.swapaxes(noise_feature, 0, 1)
noise_feature = noise_feature*0.1 #scale the noise

x = np.arange(0, 3*np.pi, 0.1)
x = signal.resample(x, 2520)
sin = np.sin(x)

ecg_bw_filtered = filtering(ecg_bw)
ecg_ma_filtered = filtering(ecg_ma)
ecg_em_filtered = filtering(ecg_em)

ecg_bw_feature = transform(ecg_bw)
ecg_ma_feature = transform(ecg_ma)
ecg_em_feature = transform(ecg_em)
#swap axis
ecg_bw_feature = np.swapaxes(ecg_bw_feature, 0, 1)
ecg_ma_feature = np.swapaxes(ecg_ma_feature, 0, 1)
ecg_em_feature = np.swapaxes(ecg_em_feature, 0, 1)

#feature 1
plt.figure()
plt.plot(f1)
plt.savefig("f1.png")
#feature 2
plt.figure()
plt.plot(f2)
plt.savefig("f2.png")
#noise
plt.figure()
plt.plot(noise)
plt.savefig("noise.png")
#noise filtered
plt.figure()
plt.plot(noise_filtered)
plt.savefig("noise_filtered.png")
#noise features
plt.figure()
plt.plot(noise_feature[0])
plt.savefig("noise_feature1.png")
plt.figure()
plt.plot(noise_feature[1])
plt.savefig("noise_feature2.png")
#feature 1 + noise feature 1
plt.figure()
plt.plot(f1 + noise_feature[0])
plt.savefig("f1_noisy.png")
#feature 2 + noise feature 2
plt.figure()
plt.plot(f2 + noise_feature[1])
plt.savefig("f2_noisy.png")
#feature 1 augmented
plt.figure()
plt.plot((f1 + noise_feature[0])*sin)
plt.savefig("f1_augmented.png")
#feature 2 augmented
plt.figure()
plt.plot((f2 + noise_feature[1])*sin)
plt.savefig("f2_augmented.png")

#bassline wander
plt.figure()
plt.plot(ecg_bw)
plt.savefig("bw.png")
#bassline wander filtered
plt.figure()
plt.plot(ecg_bw_filtered)
plt.savefig("bw_filtered.png")
#bassline wander features
plt.figure()
plt.plot(ecg_bw_feature[0])
plt.savefig("bw_feature1.png")
plt.figure()
plt.plot(ecg_bw_feature[1])
plt.savefig("bw_feature2.png")

#muscle artifact
plt.figure()
plt.plot(ecg_ma)
plt.savefig("ma.png")
#muscle artifact filtered
plt.figure()
plt.plot(ecg_ma_filtered)
plt.savefig("ma_filtered.png")
#muscle artifact features
plt.figure()
plt.plot(ecg_ma_feature[0])
plt.savefig("ma_feature1.png")
plt.figure()
plt.plot(ecg_ma_feature[1])
plt.savefig("ma_feature2.png")

#electrode motion
plt.figure()
plt.plot(ecg_em)
plt.savefig("em.png")
#electrode motion filtered
plt.figure()
plt.plot(ecg_em_filtered)
plt.savefig("em_filtered.png")
#electrode motion features
plt.figure()
plt.plot(ecg_em_feature[0])
plt.savefig("em_feature1.png")
plt.figure()
plt.plot(ecg_em_feature[1])
plt.savefig("em_feature2.png")
plt.close()


#scale the artificial noise again
ecg_bw_feature = ecg_bw_feature*abs(np.random.normal(0,0.05, 1))
ecg_ma_feature = ecg_ma_feature*abs(np.random.normal(0,0.05, 1))
ecg_em_feature = ecg_em_feature*abs(np.random.normal(0,0.05, 1))


#plot the augmented waveforms
#bassline wander
plt.figure()
plt.plot((f1 + noise_feature[0])*sin + ecg_bw_feature[0])
plt.savefig("f1_aug_w_bw.png")
plt.figure()
plt.plot((f2 + noise_feature[1])*sin + ecg_bw_feature[1])
plt.savefig("f2_aug_w_bw.png")
#muscle artifact
plt.figure()
plt.plot((f1 + noise_feature[0])*sin + ecg_ma_feature[0])
plt.savefig("f1_aug_w_ma.png")
plt.figure()
plt.plot((f2 + noise_feature[1])*sin + ecg_ma_feature[1])
plt.savefig("f2_aug_w_ma.png")
#electrode motion
plt.figure()
plt.plot((f1 + noise_feature[0])*sin + ecg_em_feature[0])
plt.savefig("f1_aug_w_em.png")
plt.figure()
plt.plot((f2 + noise_feature[1])*sin + ecg_em_feature[1])
plt.savefig("f2_aug_w_em.png")
#all three
plt.figure()
plt.plot((f1 + noise_feature[0])*sin + ecg_bw_feature[0] + ecg_ma_feature[0] + ecg_em_feature[0])
plt.savefig("f1_aug_w_everything.png")
plt.figure()
plt.plot((f2 + noise_feature[1])*sin + ecg_bw_feature[1] + ecg_ma_feature[1] + ecg_em_feature[1])
plt.savefig("f2_aug_w_everything.png")

# fig = go.Figure(
#     data = [
#         go.Scatter(y=np.swapaxes(ecg_bw, 0, 1)[1]),
#         go.Scatter(y=np.swapaxes(ecg_bw, 0, 1)[0])
#     ]
# )
# fig.write_image("foo.png")
# fig.write_html("foo.html")


# fig = px.line(x=ecg_bw)
# fig.write_image("foo1.png")
# fig.write_html("foo1.html")

#usable_channels, _ = return_good_ecg_channel_idx_based_on_lead_off(vaf["strip_ecg"][0], vaf["lead_off"][0], 3)


# fig = make_subplots(rows=5, cols=1)
# #x_val = np.linspace(0,)
# fig.add_trace(
#     go.Scatter(y = vaf["strip_ecg"][0][0], name="ECG 1"),
#     row=1, col=1,
# )
# fig.add_trace(
#     go.Scatter(y = vaf["strip_ecg"][0][1], name="ECG 2"),
#     row=2, col=1
# )
# fig.add_trace(
#     go.Scatter(y = vaf["strip_ecg"][0][2], name="ECG 3"),
#     row=3, col=1
# )
# fig.add_trace(
#     go.Scatter(y = vaf["strip_ecg"][0][1], name="ECG 2"),
#     row=4, col=1
# )
# fig.add_trace(
#     go.Scatter(y = vaf["strip_ecg"][0][2], name="ECG 3"),
#     row=5, col=1
# )
# fig.update_layout(height=600, title_text="YES THEN")
# fig.write_image("foo.png")
# fig.write_html("foo.html")




# fig, axs = plt.subplots(3)
# axs[0].plot(ecg[idx])
# axs[1].plot(target[idx])
# axs[1].vline()
# axs[2].plot(pred[idx][:target[idx].shape[0]])
# plt.savefig("foo.png")
