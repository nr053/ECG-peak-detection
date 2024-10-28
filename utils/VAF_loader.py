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
import yaml
from utils.db_loader import DB_loading

#config
with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

target_lv = cfg['target_level']
fs_resampling = cfg['fs_resampling']
duration = cfg['label_window_duration'] 



class VAF_loading(DB_loading):
    def __init__(self, path_to_data):
        self.path_database = path_to_data

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
        #strip_length = vaf["strip_length"]


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


        for file in tqdm(file_names):
            
            with open(file, 'rb') as f:
                vaf = pickle.load(f)
        
            #input data
            ecg = vaf["strip_ecg"]
            fs = 256 #Hz
            label = vaf["beat_positions"]
            start_times = vaf["strip_start_ms"]
            lead_off = vaf["lead_off"]
            strip_types = vaf["strip_type"]
            mask = np.array([])

            
            
            ecg_resampled = []
            filenames = []
            label_resampled = []
            features = []
            targets = []
            masks_array = []
            strip_ids = []
            channel_ids = []


            
            #print(file) n
            #keep track of strip ID
            strip_id = 0

            zip_list = zip(ecg, label, start_times, lead_off, strip_types)

            for ecg_strip, label_strip, start_time, lead_off_list, strip_type in zip_list:
                strip_id += 1
                
                
                #skip the strip if the strip_type is "signal_quality_example" (there are no labelled beat positions)
                if strip_type in {"signal_quality_example", "patient_event"}:
                    continue

                #drop strips that are not 7seconds
                #if ecg_strip.shape[1] != 1792:
                #    continue
                

                usable_channels, _ = return_good_ecg_channel_idx_based_on_lead_off(ecg_strip, lead_off_list, 3)
                resampled_label = self.resample_label(np.array([int((beat_position - start_time)*256/1000) for beat_position in label_strip]), fs, fs_resampling)
                for idx in usable_channels:
                    
                    
                    
                    #reject strips that have amplitude range more than 5000mV or less than 80mV
                    range = np.ptp(ecg_strip[idx])
                    if range > 5000 or range < 80:
                        continue

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
                channel_ids
                )
            
            for ecg, label, feature, target, mask_array, filename, strip_id, channel_id in zip_info:

                set_dict['ecg'].append(ecg)
                set_dict['label'].append(label)
                set_dict['feature'].append(feature)
                set_dict['target'].append(target)
                set_dict['mask_array'].append(mask_array)

                if not train:
                    #include metadata
                    set_dict['filename'].append(filename)
                    set_dict['strip_id'].append(strip_id)
                    set_dict['channel_id'].append(channel_id)

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
        fig.write_image(f"figures/vaf/png/{file}_{strip_id[idx]}_{channel_id[idx]}.png")
        fig.write_html(f"figures/vaf/html/{file}_{strip_id[idx]}_{channel_id[idx]}.html")
