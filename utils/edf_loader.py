import numpy as np
import pandas as pd
import sys, os
import pywt
import glob
import wfdb
from scipy import signal
from tqdm import tqdm
import mne
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math

target_lv = 4
fs_resampling = 360
duration = 0.15 # 150ms

class EDF_loading:
    def __init__(self):
        # path definition
        path_base = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        self.path_database = '/home/rose/Cortrium/ECG-peak-detection/'
 

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
        len_padding = -len(array)%2**target_lv
        padded = np.pad(array, (0, len_padding), 'edge')
        coeff_swt = pywt.swt(padded, 'sym4', level=target_lv, trim_approx=True)
        # [cAn, cDn, ..., cD2, cD1]
        coeff_swt.reverse()
        # [cD1, cD2, ... ,cDn, cAn]

        feature = coeff_swt[3]
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

    def make_target(self, array, label, w_size=None):
        if w_size is None:
            w_size = int(fs_resampling*duration)
        target = np.zeros(array.shape[0])
        n_label = np.array([x for x in label if x < array.shape[0]]).astype('int')
        target[n_label] = 1
        target = np.convolve(target, np.ones(w_size), mode='same')
        target = np.where(target > 1, 1, target) # ?
        return target

    ### loading
    def return_idx(self, db_name):
        list_idx = self.report_table[(self.report_table['Database'] == db_name) & (self.report_table['Select'] == 1)].index
        return list_idx

    def nan_helper(self, array):
        nans, x = np.isnan(array), lambda z: z.nonzero()[0]
        array[nans] = np.interp(x(nans), x(~nans), array[~nans])
        return array

    def load_annotation(self, ann):
        # https://archive.physionet.org/physiobank/annotations.shtml
        beat_labels = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
        in_beat_labels = np.in1d(ann.symbol, beat_labels)
        sorted_anno = ann.sample[in_beat_labels]
        sorted_anno = np.unique(sorted_anno)
        sorted_anno = sorted_anno[sorted_anno >= 0]
        return sorted_anno

    def load_data(self, file, verbose=False):
        # load data

        record = mne.io.read_raw_edf(file)
        ecg = record.get_data()
        fs = record.info["sfreq"]

        # load annotation
        label = np.array([])
        mask = np.array([])

        return ecg, label, fs, mask

    # pipeline
    def create_set(self, use_swt=True): 

        filenames = glob.glob(self.path_database + '**/*.edf', recursive=True)
        self.metadata_patient = [name.split("/")[-1] for name in filenames]

        time_frame = 60*60*256 #60 seconds * 60 minutes = 3600 seconds in one hour / 3600 seconds * 256 Hz = 921600 datapoints

        set_dict = dict()
        set_dict['ecg'] = []
        set_dict['label'] = []
        set_dict['feature'] = []
        set_dict['target'] = []
        set_dict['mask_array'] = []
        set_dict['filename'] = []
        set_dict['channel_id'] = []
        set_dict['hour_id'] = []

        for file in tqdm(filenames):
            ecg, label, fs, mask = self.load_data(file)
            
            hours_of_data = math.ceil(ecg.shape[1]/256/3600) 

            for hour_id in range(hours_of_data): 
                if hour_id == hours_of_data-1:
                    ecg_tmp = ecg[:,time_frame*(hour_id):]
                else:
                    ecg_tmp = ecg[:,time_frame*(hour_id):time_frame*(hour_id+1)]
                channel_id = 1
                for ecg_channel in ecg_tmp:
                    ecg_channel = self.resample_ecg(ecg_channel, fs, fs_resampling)
                    label = self.resample_label(label, fs, fs_resampling)
                    mask = self.resample_label(mask, fs, fs_resampling)

                    feature = self.transform(ecg_channel, use_swt=use_swt)
                    target = self.make_target(feature, label)
                    mask_array = self.make_target(feature, mask, w_size=25)

                    feature_diff = np.abs(feature[:,1]) # ignore flat areas (sum of absolute differences for 2 seconds < 0.1mV)
                    area_ignore = np.convolve(feature_diff, np.ones(fs_resampling*2), mode='same')
                    area_ignore = np.where(area_ignore < 0.1, 1, 0)
                    mask_array += area_ignore
                    mask_array = np.where(mask_array>0, 1, 0)

                    set_dict['ecg'].append(ecg_channel)
                    set_dict['label'].append(label)
                    set_dict['feature'].append(feature)
                    set_dict['target'].append(target)
                    set_dict['mask_array'].append(mask_array)
                    set_dict['filename'].append(file.split("/")[-1])
                    set_dict['channel_id'].append(channel_id)
                    set_dict['hour_id'].append(hour_id)

                    channel_id +=1



        return set_dict



    def visualise(self, set_dict, idx):
    
        ecg = set_dict["ecg"]
        feature = set_dict["feature"]
        pred = set_dict["pred"]
        filename = set_dict["filename"]
        channel_id = set_dict["channel_id"]
        hour_id = set_dict["hour_id"]
        # fig, axs = plt.subplots(5, constrained_layout=True)
        # #fig.tight_layout()
        # axs[0].plot(ecg[idx])
        # axs[1].plot(feature[idx][:,0])
        # axs[2].plot(feature[idx][:,1])
        # axs[3].plot(target[idx])
        # axs[4].plot(pred[idx][:target[idx].shape[0]])

        # fig.suptitle(f"{file}, strip: {strip_id[idx]}, channel: {channel_id[idx]} ", fontsize=8)
        # plt.savefig(f"figures/foo_{file}_{strip_id[idx]}_{channel_id[idx]}.png")
        # plt.close()
        

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

        fig.add_trace(
            go.Scatter(y=ecg[idx], name=f"ECG"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=feature[idx][:,0], name="feature 1"),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=feature[idx][:,1], name="feature 2"),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(y=pred[idx], name="prediction"),
            row=4, col=1
        )

        fig.update_layout(title_text=f"{filename[idx]}, hour: {hour_id[idx]}, channel: {channel_id[idx]}")
        fig.write_image(f"figures/edf/png/{idx}.png")
        fig.write_html(f"figures/edf/html/{idx}.html")