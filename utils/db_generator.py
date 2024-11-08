import numpy as np
import torch.utils.data as data
import pandas as pd
import yaml
import wfdb
from sklearn.preprocessing import MinMaxScaler
import pywt
from scipy import signal
import random

with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

feature_shape = cfg["feature_shape"]
batch_size = cfg["batch_size"]
data_path = cfg["path_to_data"]
fs_resampling = cfg["fs_resampling"]
target_lv = cfg["target_level"]

class Test_Loader(data.Dataset):
    def __init__(self, set_dict):
        self.list_feature = set_dict['feature']
        self.list_target = set_dict['target']
        self.iteration = len(self.list_feature)
        # print('Test ... No. of Records :', self.iteration)

    def __len__(self):
        return self.iteration

    def __getitem__(self, index):
        feature = self.list_feature[index]
        target = self.list_target[index]

        padding = -feature.shape[0]%feature_shape # cropping
        if padding != 0:
            feature = np.pad(feature, ((0, padding), (0, 0)))
            target = np.pad(target, (0, padding))

        #crop input and target for training purposes
        #feature = feature[:2048]
        #target = target[:2048]

        X = np.swapaxes(feature, 0, 1)
        y = target[np.newaxis, :]

        return X, y

class Train_Loader(data.Dataset):
    def __init__(self, set_dict, data_aug):
        self.list_feature = set_dict['feature']
        self.list_target = set_dict['target']
        self.iteration = len(self.list_feature)
        # print('Test ... No. of Records :', self.iteration)

        #load artificial noise files
        self.data_aug = data_aug
        record_bw = wfdb.rdsamp(data_path + "/MIT_BIH_NST/bw")
        record_ma = wfdb.rdsamp(data_path + "/MIT_BIH_NST/ma")
        record_em = wfdb.rdsamp(data_path + "/MIT_BIH_NST/em")

        bassline_wander = record_bw[0][:, 1] #use second channel only
        muscle_artifact = record_ma[0][:, 1]
        electrode_motion = record_em[0][:, 1]

        #preprocess (filter + transform)
        bassline_wander_filtered = self.filtering(bassline_wander)
        muscle_artifact_filtered = self.filtering(muscle_artifact)
        electrode_motion_filtered = self.filtering(electrode_motion)

        self.bassline_wander_transformed = self.transform(bassline_wander_filtered)
        self.muscle_artifact_transformed = self.transform(muscle_artifact_filtered)
        self.electrode_motion_transformed = self.transform(electrode_motion_filtered)



        ### filtering method
    def dwt_idwt(self, array, wavelet='db3', level=9):
        coeffs = pywt.wavedec(array, wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0])
        array_filtered = pywt.waverec(coeffs, wavelet)
        return array_filtered

    def lowpass_filter(self, array, fs, cutoff=40, order=5, remove_lag=True):
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

    def filtering(self, array, fs=fs_resampling):
        array_filtered = array.copy()
        array_filtered = self.dwt_idwt(array_filtered)
        array_filtered = self.lowpass_filter(array_filtered, fs)
        return array_filtered

    ### resampling
    def resample_ecg(self, array, fs_pre, fs_post=fs_resampling):
        t_len = len(array)
        new_len = int(fs_post/fs_pre*t_len)
        array_resampled = signal.resample(array, new_len)
        array_resampled = self.filtering(array_resampled)
        return array_resampled

    def resample_label(self, label, fs_pre, fs_post=fs_resampling):
        label_resampled = (label*fs_post/fs_pre).astype('int')
        return label_resampled

    ### transformation
    def normalization(self, array):
        root_squared_mean = np.mean(array**2)**0.5
        array_norm = array/root_squared_mean/16
        return array_norm

    def sw_transform(self, array):
        #target_lv=1 #for debug purposes
        len_padding = -len(array)%2**target_lv
        padded = np.pad(array, (0, len_padding), 'edge')
        coeff_swt = pywt.swt(padded, 'sym4', level=target_lv, trim_approx=True)
        # [cAn, cDn, ..., cD2, cD1]
        coeff_swt.reverse()
        # [cD1, cD2, ... ,cDn, cAn]

        feature = coeff_swt[3]
        #feature = coeff_swt[1] #for debug purposes
        feature = feature[:len(array)]
        feature = self.normalization(feature)

        diff = np.diff(array, append=array[-1])
        diff = self.normalization(diff)

        merged_feature = np.stack([feature, diff], axis=1)
        return merged_feature

    def transform(self, array, use_swt=True):
        if use_swt == False:
            diff = np.diff(array, append=array[-1])
            diff = self.normalization(diff)
            feature = diff.reshape(-1, 1)
            return feature
        else:
            feature = self.sw_transform(array)
        return feature

    def __len__(self):
        return self.iteration

    def __getitem__(self, index):
        feature = self.list_feature[index]
        target = self.list_target[index]

        padding = -feature.shape[0]%feature_shape # cropping
        if padding != 0:
            feature = np.pad(feature, ((0, padding), (0, 0)))
            target = np.pad(target, (0, padding))

        #crop input and target for training purposes
        feature = feature[:feature_shape]
        target = target[:feature_shape]

        #data augmentation 
        if self.data_aug:
            #generate white noise
            noise = np.random.normal(0,1,feature_shape)
            noise = self.filtering(noise)
            noise_feature = self.transform(noise, use_swt=True)
            #noise_feature = np.swapaxes(noise, 0,1)
            noise_feature = noise_feature*0.1 #scale the noise

            #generate sin wave
            x = np.arange(0, 3*np.pi, 0.1)
            sin = np.sin(x)
            sin = signal.resample(sin, feature_shape)
            sin = sin*random.uniform(0.5,1.5) #scale sin wave

            #generate artificial noise
            #choose random snippet from each type
            start_idx_bw = random.randint(0,self.bassline_wander_transformed.shape[0] - feature_shape)
            start_idx_ma = random.randint(0,self.muscle_artifact_transformed.shape[0] - feature_shape)
            start_idx_em = random.randint(0,self.electrode_motion_transformed.shape[0] - feature_shape)
            bassline_wander = self.bassline_wander_transformed[start_idx_bw:start_idx_bw+feature_shape] #use second channel only
            muscle_artifact = self.muscle_artifact_transformed[start_idx_ma:start_idx_ma+feature_shape]
            electrode_motion = self.electrode_motion_transformed[start_idx_em:start_idx_em+feature_shape]
            #scale 
            #percentile_99 = np.quantile(feature, 0.999)
            #percentile_01 = np.quantile(feature, 0.001)

            #bassline_wander = bassline_wander.reshape(-1,1)
            #muscle_artifact = muscle_artifact.reshape(-1,1)
            #electrode_motion = electrode_motion.reshape(-1,1)

            #scaler = MinMaxScaler(feature_range=(percentile_01, percentile_99))
            #scaler.fit(bassline_wander)
            #bassline_wander_scaled = scaler.transform(bassline_wander)
            #scaler.fit(muscle_artifact)
            #muscle_artifact_scaled = scaler.transform(muscle_artifact)
            #scaler.fit(electrode_motion)
            #electrode_motion_scaled = scaler.transform(electrode_motion)
            

            #add artificial noise with 50% probability
            artificial_noise = np.zeros(bassline_wander.shape)
            if random.choice([True,False]):
                artificial_noise = artificial_noise + bassline_wander
            if random.choice([True,False]):
                artificial_noise = artificial_noise + muscle_artifact
            if random.choice([True,False]):
                artificial_noise = artificial_noise + electrode_motion

            #scale the mixed noise 
            scale_factor = abs(np.random.normal(0,0.05))
            if scale_factor < 2:
                artificial_noise = artificial_noise * abs(np.random.normal(0,0.05))

            #step 1 (white noise and amplitude adjustment)
            feature = feature + noise_feature
            feature[:,0] = feature[:,0]*sin
            feature[:,1] = feature[:,1]*sin
            #step 2 (artificial noise)
            feature = feature + artificial_noise
            #step 3 (flip or reverse with 50% probability)
            if random.choice([True,False]):
                feature = feature*-1
            if random.choice([True,False]):
                feature = feature[::-1]

            
            

        X = np.swapaxes(feature.copy(), 0, 1)
        y = target[np.newaxis, :]

        return X, y


def Test_Generator(set_dict):
    data_loader = Test_Loader(set_dict)
    data_generator = data.DataLoader(data_loader, batch_size=1, shuffle=False)
    return data_generator

def Validation_Generator(set_dict):
    data_loader = Test_Loader(set_dict)
    data_generator = data.DataLoader(data_loader, batch_size=128, shuffle=True)
    return data_generator

def Train_Generator(set_dict, data_aug=False):
    data_loader_train = Train_Loader(set_dict, data_aug)
    data_generator_train = data.DataLoader(data_loader_train, batch_size=batch_size, shuffle=True)
    
    return data_generator_train
