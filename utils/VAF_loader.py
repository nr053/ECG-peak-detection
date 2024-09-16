import numpy as np
import pandas as pd
import sys, os
import pywt
import wfdb
from scipy import signal
import glob
from tqdm import tqdm
import pickle
from utils.helper_functions import return_good_ecg_channel_idx_based_on_lead_off
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go


target_lv = 4
#target_lv = 1
fs_resampling = 360
duration = 0.15 # 150ms

class VAF_loading:
    def __init__(self, path_to_data):
        self.path_database = path_to_data

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
        lead_off = vaf["lead_off"]
        strip_type = vaf["strip_type"]
        strip_length = vaf["strip_length"]


        mask = np.array([])

        #remember the vaf is a dictionary containings lists of strip data. So ECG is a list of ECG strips, label is a list of beat positions lists. 

        return ecg, label, start_times, fs, lead_off, strip_type, mask

    # pipeline
    def create_set(self, use_swt=True, train=False): 

        file_names = glob.glob(self.path_database + '**/*.pkl', recursive=True)
        self.metadata_patient = [name.split("/")[-1] for name in file_names]

        set_dict = dict()
        set_dict['filename'] = []
        set_dict['strip_id'] = []
        set_dict['channel_id'] = []
        set_dict['ecg'] = []
        set_dict['label'] = []
        set_dict['feature'] = []
        set_dict['target'] = []
        set_dict['mask_array'] = []
        #metadata
        set_dict['strip_type'] = []
        #set_dict['strip_length'] = []
        set_dict['strip_position'] = []
        set_dict['beat_annotation'] = []
        #set_dict['signal_quality_level'] = []

        for file in tqdm(file_names):
            
            with open(file, 'rb') as f:
                vaf = pickle.load(f)
        
            #input data
            ecg = vaf["strip_ecg"]
            fs = 256 #Hz
            label = vaf["beat_positions"]
            start_times = vaf["strip_start_ms"]
            lead_off = vaf["lead_off"]
            mask = np.array([])
            #meta data
            strip_types = vaf["strip_type"]
            #strip_lengths = vaf["strip_length"]
            strip_positions = vaf["strip_position"]
            beat_annotations = vaf["beat_annotations"]
            #signal_quality_levels = vaf["signal_quality_level"]
            
            
            ecg_resampled = []
            filenames = []
            label_resampled = []
            features = []
            targets = []
            masks_array = []
            strip_ids = []
            channel_ids = []


            
            #print(file)
            #keep track of strip ID
            strip_id = 0

            zip_list = zip(ecg, label, start_times, lead_off, strip_types)

            for ecg_strip, label_strip, start_time, lead_off_list, strip_type in zip_list:
                strip_id += 1
                
                
                #skip the strip if the strip_type is "signal_quality_example" (there are no labelled beat positions)
                #if strip_type in {"signal_quality_example", "patient_event"}:
                #    continue

                #drop strips that are not 7seconds
                #if ecg_strip.shape[1] != 1792:
                #    continue
                

                usable_channels, _ = return_good_ecg_channel_idx_based_on_lead_off(ecg_strip, lead_off_list, 3)
                resampled_label = self.resample_label(np.array([int((beat_position - start_time)*256/1000) for beat_position in label_strip]), fs, fs_resampling)
                for idx in usable_channels:
                    
                    
                    
                    #reject strips that have amplitude range more than 5000mV or less than 80mV
                    #range = np.ptp(ecg_strip[idx])
                    #if range > 5000 or range < 80:
                    #    continue

                    ecg_resampled.append(self.resample_ecg(ecg_strip[idx], fs, fs_resampling)) #use each channel of ecg one at a time
                    label_resampled.append(resampled_label)
                    filenames.append(file)
                    strip_ids.append(strip_id)
                    channel_ids.append(idx)

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

            
            zip_info = zip(
                ecg_resampled, 
                label_resampled, 
                features, 
                targets, 
                masks_array, 
                filenames, 
                strip_ids, 
                channel_ids,
                strip_types,
                #strip_lengths,
                strip_positions,
                beat_annotations,
                #signal_quality_levels
                )
            
            for ecg, label, feature, target, mask_array, filename, strip_id, channel_id, strip_type, strip_position, beat_annotation in zip_info:

                set_dict['ecg'].append(ecg)
                set_dict['label'].append(label)
                set_dict['feature'].append(feature)
                set_dict['target'].append(target)
                set_dict['mask_array'].append(mask_array)
                set_dict['filename'].append(filename)
                set_dict['strip_id'].append(strip_id)
                set_dict['channel_id'].append(channel_id)
                #metadata
                set_dict['strip_type'].append(strip_type)
                #set_dict['strip_length'].append(strip_length)
                set_dict['strip_position'].append(strip_position)
                set_dict['beat_annotation'].append(beat_annotation)
                #set_dict['signal_quality_level'].append(signal_quality_level)


        if train == True:
            set_dict = pd.DataFrame.from_dict(set_dict)
        
        return set_dict


    def visualise(self, set_dict, idx):
    


        ecg = set_dict["ecg"]
        feature = set_dict["feature"]
        target = set_dict["target"]
        label = set_dict["label"]
        pred = set_dict["pred"]
        
        filename = set_dict["filename"]
        strip_id = set_dict["strip_id"]
        channel_id = set_dict["channel_id"]


        # fig, axs = plt.subplots(5, constrained_layout=True)
        # #fig.tight_layout()
        # axs[0].plot(ecg[idx])
        # axs[1].plot(feature[idx][:,0])
        # axs[2].plot(feature[idx][:,1])
        # axs[3].plot(target[idx])
        # axs[4].plot(pred[idx][:target[idx].shape[0]])

        file = filename[idx].split("/")[-1]
        strip_idx = strip_id[idx]-1

        ecg_v, label_v, start_times_v, fs_v, lead_off_v, strip_type_v, mask_v = self.load_data(filename[idx])
        usable_channels, _ = return_good_ecg_channel_idx_based_on_lead_off(ecg_v[strip_idx], lead_off_v[strip_idx], 3)
        usable_channels.remove(channel_id[idx])

        # fig.suptitle(f"{file}, strip: {strip_id[idx]}, channel: {channel_id[idx]} ", fontsize=8)
        # plt.savefig(f"figures/foo_{file}_{strip_id[idx]}_{channel_id[idx]}.png")
        # plt.close()
        

        fig = make_subplots(rows=5+len(usable_channels), cols=1, shared_xaxes=True)

        fig.add_trace(
            go.Scatter(y=ecg[idx], name=f"ECG {str(channel_id[idx])}"),
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
            go.Scatter(y=target[idx], name="target"),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(y=pred[idx][:target[idx].shape[0]], name="prediction"),
            row=5, col=1
        )
        extra_plot_number = 1
        for channel in usable_channels:
            ecg_data = self.resample_ecg(ecg_v[strip_idx][channel], 256, 360)
            fig.add_trace(
                go.Scatter(y=ecg_data, name=f"ECG {str(channel)}"),
                row=5+extra_plot_number, col=1
            )
            extra_plot_number+=1

        fig.update_layout(title_text=f"{file}, strip: {strip_id[idx]}, channel: {channel_id[idx]}")
        fig.write_image(f"figures/png/{file}_{strip_id[idx]}_{channel_id[idx]}.png")
        fig.write_html(f"figures/html/{file}_{strip_id[idx]}_{channel_id[idx]}.html")
