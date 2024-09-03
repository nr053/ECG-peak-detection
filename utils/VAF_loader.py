import numpy as np
import pandas as pd
import sys, os
import pywt
import wfdb
from scipy import signal
import glob
from tqdm import tqdm
import pickle

target_lv = 4
#target_lv = 1
fs_resampling = 360
duration = 0.15 # 150ms

class VAF_loading:
    def __init__(self):
        # path definition
        path_base = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        #self.path_database = '/home/rose/Cortrium/R-peak-detector/VAF_subset/test/'
        self.path_database = '/home/rose/Cortrium/R-peak-detector/10k/VAF_subset/'

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
        #feature = coeff_swt[1]
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

    def load_data(self, file):
        
        with open(file, 'rb') as f:
            vaf = pickle.load(f)
        
        #input data
        ecg = vaf["strip_ecg"]
        fs = 256 #Hz
        label = vaf["beat_positions"]
        start_times = vaf["strip_start_ms"]
        mask = np.array([])

        #remember the vaf is a dictionary containings lists of strip data. So ECG is a list of ECG strips, label is a list of beat positions lists. 

        return ecg, label, start_times, fs, mask

    # pipeline
    def create_set(self, use_swt=True): 

        file_names = glob.glob(self.path_database + '**/*.pkl', recursive=True)
        self.metadata_patient = [name.split("/")[-1] for name in file_names]

        set_dict = dict()
        set_dict['ecg'] = []
        set_dict['label'] = []
        set_dict['feature'] = []
        set_dict['target'] = []
        set_dict['mask_array'] = []



        for file in tqdm(file_names):
            ecg, label, start_times, fs, mask = self.load_data(file)
            
            ecg_resampled = []
            label_resampled = []
            features = []
            targets = []
            masks_array = []
            
            for strip in ecg:
                for i in range(3):
                    ecg_resampled.append(self.resample_ecg(strip[i], fs, fs_resampling)) #use each channel of ecg one at a time
            for strip, start_time in zip(label, start_times):
                resampled_label = self.resample_label(np.array([int((beat_position - start_time)*256/1000) for beat_position in strip]), fs, fs_resampling)
                for i in range(3):
                    label_resampled.append(resampled_label)
            mask = self.resample_label(mask, fs, fs_resampling)

            for strip in ecg_resampled:
                features.append(self.transform(strip, use_swt=use_swt))
            for feature, label in zip(features, label_resampled):
                targets.append(self.make_target(feature, label))
                mask_array = (self.make_target(feature, mask, w_size=25))
                
                feature_diff = (np.abs(feature[:,1])) # ignore flat areas (sum of absolute differences for 2 seconds < 0.1mV)
                area_ignore = np.convolve(feature_diff, np.ones(fs_resampling*2), mode='same')
                area_ignore = np.where(area_ignore < 0.1, 1, 0)
                mask_array += area_ignore
                mask_array = np.where(mask_array>0, 1, 0)
                
                masks_array.append(mask_array)

            for ecg, label, feature, target, mask_array in zip(ecg_resampled, label_resampled, features, targets, masks_array):

                set_dict['ecg'].append(ecg)
                set_dict['label'].append(label)
                set_dict['feature'].append(feature)
                set_dict['target'].append(target)
                set_dict['mask_array'].append(mask_array)

        return set_dict
