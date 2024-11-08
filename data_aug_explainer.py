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
import random
from sklearn.preprocessing import MinMaxScaler

#config
with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

feature_shape = cfg['feature_shape']
target_lv = cfg['target_level']
path_to_repo = cfg['path_to_repo']
path_to_data = cfg['path_to_data']

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
    #target_lv = 1 #for debug purposes
    len_padding = -len(array)%2**target_lv
    padded = np.pad(array, (0, len_padding), 'edge')
    coeff_swt = pywt.swt(padded, 'sym4', level=target_lv, trim_approx=True)
    # [cAn, cDn, ..., cD2, cD1]
    coeff_swt.reverse()
    # [cD1, cD2, ... ,cDn, cAn]

    feature = coeff_swt[3]
    #feature = coeff_swt[1] #for debug purposes
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
with open(path_to_data + "/10K_VAF_subset/VAF_subset/train/00a4aad0-5d1b-4456-a6f0-8e5c13d0feda.pkl", 'rb') as file:
    vaf = pickle.load(file)


#edf
#raw = mne.io.read_raw_edf(path_to_data + "/EDF/edf1.edf", preload=True)
#data = raw.get_data()


# other files
record = wfdb.rdsamp(path_to_data + "/Databases/MIT_BIH/100")
ecg = record[0][:, 1]
fs = record[1]['fs']
# load annotation
ann = wfdb.rdann(path_to_data + "/Databases/MIT_BIH/100", 'atr')


# artifical noise
record_bw = wfdb.rdsamp(path_to_data + "/Databases/MIT_BIH_NST/bw")
record_ma = wfdb.rdsamp(path_to_data + "/Databases/MIT_BIH_NST/ma")
record_em = wfdb.rdsamp(path_to_data + "/Databases/MIT_BIH_NST/em")


bassline_wander = record_bw[0][:, 1] #use second channel only
muscle_artifact = record_ma[0][:, 1]
electrode_motion = record_em[0][:, 1]

#preprocess (filter + transform)
bassline_wander_filtered = filtering(bassline_wander)
muscle_artifact_filtered = filtering(muscle_artifact)
electrode_motion_filtered = filtering(electrode_motion)

bassline_wander_transformed = transform(bassline_wander_filtered)
muscle_artifact_transformed = transform(muscle_artifact_filtered)
electrode_motion_transformed = transform(electrode_motion_filtered)






data_train = DB_loading()
dict_incart = data_train.create_set("INCART", train=True)
dict_mit = data_train.create_set("MIT_BIH", train=True)
dict_qt = data_train.create_set("QTDB", train=True)
dict_train = append_dicts(dict_incart,  [dict_mit, dict_qt])
dataloader_train = Train_Generator(dict_train)

#take a random sample from the input space
input = dict_train["feature"][500][:feature_shape,:]

#plot the input before augmentation
fig, axs = plt.subplots(2)
fig.tight_layout()
axs[0].plot(np.swapaxes(input, 0, 1)[0])
axs[0].set_title("feature 1")
axs[1].plot(np.swapaxes(input, 0, 1)[1])
axs[1].set_title("feature 2")
fig.suptitle("input before augmentation")
plt.savefig("input_before_augmentation.png")
plt.close()


#generate white noise
noise = np.random.normal(0,1,2048)
noise_filtered = filtering(noise)
noise_feature = transform(noise_filtered, use_swt=True)
noise_feature_scaled = noise_feature*0.1 #scale the noise

#plot white noise
fig, axs = plt.subplots(4)
fig.tight_layout()
axs[0].plot(noise)
axs[0].set_title("noise")
axs[1].plot(noise_filtered)
axs[1].set_title("filtered")
axs[2].plot(noise_feature)
axs[2].set_title("transformed")
axs[3].plot(noise_feature_scaled)
axs[3].set_title("scaled")
fig.suptitle("White noise")
plt.savefig("white_noise.png")
plt.close()


#generate sin wave
x = np.arange(0, 3*np.pi, 0.1)
sin = np.sin(x)
sin = signal.resample(sin, 2048)
sin_scaled = sin*random.uniform(0.5,1.5)

#plot sin wave
fig, axs = plt.subplots(2)
fig.tight_layout()
axs[0].plot(sin)
axs[0].set_title("sin")
axs[1].plot(sin_scaled)
axs[1].set_title("sin scaled")
fig.suptitle("sin wave")
plt.savefig("sin.png")
plt.close()

#generate artificial noise
#choose random snippet from each type
start_idx_bw = random.randint(0,bassline_wander_transformed.shape[0] - feature_shape)
start_idx_ma = random.randint(0,muscle_artifact_transformed.shape[0] - feature_shape)
start_idx_em = random.randint(0,electrode_motion_transformed.shape[0] - feature_shape)
bassline_wander_sample = bassline_wander_transformed[start_idx_bw:start_idx_bw+feature_shape]
muscle_artifact_sample = muscle_artifact_transformed[start_idx_ma:start_idx_ma+feature_shape]
electrode_motion_sample = electrode_motion_transformed[start_idx_em:start_idx_em+feature_shape]

# #scale 
# percentile_99 = np.quantile(input, 0.999)
# percentile_01 = np.quantile(input, 0.001)

# bassline_wander = bassline_wander.reshape(-1,1)
# muscle_artifact = muscle_artifact.reshape(-1,1)
# electrode_motion = electrode_motion.reshape(-1,1)

# scaler = MinMaxScaler(feature_range=(percentile_01, percentile_99))
# scaler.fit(bassline_wander)
# bassline_wander_scaled = scaler.transform(bassline_wander)
# scaler.fit(muscle_artifact)
# muscle_artifact_scaled = scaler.transform(muscle_artifact)
# scaler.fit(electrode_motion)
# electrode_motion_scaled = scaler.transform(electrode_motion)




#plot artificial noise
fig, axs = plt.subplots(4,3)
fig.tight_layout()
axs[0,0].plot(bassline_wander[start_idx_bw:start_idx_bw+feature_shape])
axs[0,0].set_title("bassline_wander")
#axs[1,0].plot(bassline_wander_scaled)
#axs[1,0].set_title("bassline_wander scaled")
axs[1,0].plot(bassline_wander_filtered[start_idx_bw:start_idx_bw+feature_shape])
axs[1,0].set_title("bassline_wander filtered")
axs[2,0].plot(bassline_wander_transformed[start_idx_bw:start_idx_bw+feature_shape,0])
axs[2,0].set_title("bassline_wander feature 1")
axs[3,0].plot(bassline_wander_transformed[start_idx_bw:start_idx_bw+feature_shape,1])
axs[3,0].set_title("bassline_wander feature 2")

axs[0,1].plot(muscle_artifact[start_idx_ma:start_idx_ma+feature_shape])
axs[0,1].set_title("muscle_artifact")
#axs[1,1].plot(muscle_artifact_scaled)
#axs[1,1].set_title("muscle_artifact scaled")
axs[1,1].plot(muscle_artifact_filtered[start_idx_ma:start_idx_ma+feature_shape])
axs[1,1].set_title("muscle_artifact filtered")
axs[2,1].plot(muscle_artifact_transformed[start_idx_ma:start_idx_ma+feature_shape,0])
axs[2,1].set_title("muscle_artifact feature 1")
axs[3,1].plot(muscle_artifact_transformed[start_idx_ma:start_idx_ma+feature_shape,1])
axs[3,1].set_title("muscle_artifact feature 2")

axs[0,2].plot(electrode_motion[start_idx_em:start_idx_em+feature_shape])
axs[0,2].set_title("electrode_motion")
#axs[1,2].plot(electrode_motion_scaled)
#axs[1,2].set_title("electrode_motion scaled")
axs[1,2].plot(electrode_motion_filtered[start_idx_em:start_idx_em+feature_shape])
axs[1,2].set_title("electrode_motion filtered")
axs[2,2].plot(electrode_motion_transformed[start_idx_em:start_idx_em+feature_shape,0])
axs[2,2].set_title("electrode_motion feature 1")
axs[3,2].plot(electrode_motion_transformed[start_idx_em:start_idx_em+feature_shape,1])
axs[3,2].set_title("electrode_motion feature 2")


fig.suptitle("artificial noise")
plt.savefig("artificial_noise.png")
plt.close()


#add artificial noise with 50% probability
artificial_noise = np.zeros(bassline_wander_sample.shape)
#if random.choice([True,False]):
artificial_noise_1 = artificial_noise + bassline_wander_sample
#if random.choice([True,False]):
artificial_noise_2 = artificial_noise_1 + muscle_artifact_sample
#if random.choice([True,False]):
artificial_noise_3 = artificial_noise_2 + electrode_motion_sample

#scale the mixed noise 
scale_factor = abs(np.random.normal(0,0.05))
if scale_factor < 2:
    artificial_noise_scaled = artificial_noise_3 * abs(np.random.normal(0,0.05))

fig, axs = plt.subplots(5,2)
fig.tight_layout()
axs[0,0].plot(artificial_noise)
axs[0,0].set_title("artificial_noise")

axs[1,0].plot(artificial_noise_1[:,0])
axs[1,0].set_title("artificial_noise_1 feature 1")
axs[1,1].plot(artificial_noise_1[:,1])
axs[1,1].set_title("artificial_noise_1 feature 2")

axs[2,0].plot(artificial_noise_2[:,0])
axs[2,0].set_title("artificial_noise_2 feature 1")
axs[2,1].plot(artificial_noise_2[:,1])
axs[2,1].set_title("artificial_noise_2 feature 2")

axs[3,0].plot(artificial_noise_3[:,0])
axs[3,0].set_title("artificial_noise_3 feature 1")
axs[3,1].plot(artificial_noise_3[:,1])
axs[3,1].set_title("artificial_noise_3 feature 2")

axs[4,0].plot(artificial_noise_scaled[:,0])
axs[4,0].set_title("artificial_noise_scaled feature 1")
axs[4,1].plot(artificial_noise_scaled[:,1])
axs[4,1].set_title("artificial_noise_scaled feature 2")

fig.suptitle("mixed artificial noise")
plt.savefig("artificial_noise_mixed.png")
plt.close()


#step 1 (white noise and amplitude adjustment)
input_1 = input + noise_feature
input_1[:,0] = input[:,0] * sin 
input_1[:,1] = input[:,1] * sin 
#step 2 (artificial noise)
input_2 = input_1 + artificial_noise
#step 3 (flip or reverse with 50% probability)
#if random.choice([True,False]):
input_flipped = input_2*-1
#if random.choice([True,False]):
input_reversed = input_flipped[::-1]


fig, axs = plt.subplots(5,2)
fig.tight_layout()

axs[0,0].plot(input[:,0])
axs[0,0].set_title("input f1")
axs[0,1].plot(input[:,1])
axs[0,1].set_title("input f2")

axs[1,0].plot(input_1[:,0])
axs[1,0].set_title("input with white noise and scaled f1")
axs[1,1].plot(input_1[:,1])
axs[1,1].set_title("input with white noise and scaled f2")

axs[2,0].plot(input_2[:,0])
axs[2,0].set_title("with artificial noise f1")
axs[2,1].plot(input_2[:,1])
axs[2,1].set_title("with artificial noise f2")

axs[3,0].plot(input_flipped[:,0])
axs[3,0].set_title("input_flipped f1")
axs[3,1].plot(input_flipped[:,1])
axs[3,1].set_title("input_flipped f2")

axs[4,0].plot(input_reversed[:,0])
axs[4,0].set_title("input_reversed f1")
axs[4,1].plot(input_reversed[:,1])
axs[4,1].set_title("input_reversed f2")

fig.suptitle("input features with augmentation")
plt.savefig("input_with_augmentation.png")
plt.close()