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
import yaml
from utils.db_loader import DB_loading

#config
with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
target_lv = cfg['target_level']
fs_resampling = cfg['fs_resampling']
duration = cfg['label_window_duration'] 

class EDF_loading(DB_loading):
    """
    Data class for loading EDF data. 
    """
    def __init__(self, path_to_data):
        # path definition
        self.path_database = path_to_data
 
    def load_data(self, file, verbose=False):
        # load data
        record = mne.io.read_raw_edf(file)
        ecg = record.get_data()[:3,:]
        fs = record.info["sfreq"]

        # load annotation
        label = np.array([])
        mask = np.array([])

        return ecg, label, fs, mask

    # pipeline
    def create_set(self, use_swt): 
        """
        Create set_dict for EDF data.

        Args: 
            use_swt (bool): whether to use SWT peak enhancement processing step

        Returns: 
            dict: dictionary of sample data 
        """
        filenames = glob.glob(self.path_database + '**/*.edf', recursive=True)
        self.metadata_patient = [name.split("/")[-1] for name in filenames]

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

            hours_of_data = math.ceil(ecg.shape[1]/fs/3600) #calculate using original sampling freq
            time_frame = int(60*60*fs_resampling) #60 seconds * 60 minutes = 3600 seconds in one hour / 3600 seconds * 360 Hz = 1296000 datapoints (using resampling freq)
            print(f"HOURS OF DATA: {hours_of_data}")

            channel_id = 1
            for ecg_channel in ecg:
                print(f"Processing channel {channel_id}")
            
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

                #break recording into smaller chunks so plots are not too heavy
                for hour_id in range(hours_of_data): 
                    print(f"Processing hour: {hour_id}")
                    if hour_id == hours_of_data-1:
                        ecg_tmp = ecg_channel[time_frame*(hour_id):]
                        label_tmp = label[time_frame*(hour_id):]
                        feature_tmp = feature[time_frame*(hour_id):]
                        target_tmp = target[time_frame*(hour_id):]
                        mask_array_tmp = mask_array[time_frame*(hour_id):]
                        
                    else:
                        ecg_tmp = ecg_channel[time_frame*(hour_id):time_frame*(hour_id+1)]
                        label_tmp = label[time_frame*(hour_id):time_frame*(hour_id+1)]
                        feature_tmp = feature[time_frame*(hour_id):time_frame*(hour_id+1)]
                        target_tmp = target[time_frame*(hour_id):time_frame*(hour_id+1)]
                        mask_array_tmp = mask_array[time_frame*(hour_id):time_frame*(hour_id+1)]

                    set_dict['ecg'].append(ecg_tmp)
                    set_dict['label'].append(label_tmp)
                    set_dict['feature'].append(feature_tmp)
                    set_dict['target'].append(target_tmp)
                    set_dict['mask_array'].append(mask_array_tmp)
                    set_dict['filename'].append(file.split("/")[-1])
                    set_dict['channel_id'].append(channel_id)
                    set_dict['hour_id'].append(hour_id)
                channel_id +=1

        return set_dict



    def visualise(self, set_dict, idx):
        """
        Visualise a sample from the set dict

        Args: 
            set_dict (dict): set_dict created by create_set() function
            idx (int): index of sample to be plotted
        """
        ecg = set_dict["ecg"]
        feature = set_dict["feature"]
        pred = set_dict["pred"]
        filename = set_dict["filename"]
        channel_id = set_dict["channel_id"]
        hour_id = set_dict["hour_id"]

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

        fig.add_trace(
            go.Scatter(y=ecg[idx].astype('float32'),  x=[x/360 for x in list(range(0,len(ecg[idx])))], name=f"ECG")
        )
        fig.add_trace(
            go.Scatter(y=feature[idx][:,0],  x=[x/360 for x in list(range(0,len(feature[idx][:,0])))], name="feature 1"),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=feature[idx][:,1],  x=[x/360 for x in list(range(0,len(feature[idx][:,1])))], name="feature 2"),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(y=[round(x,2) for x in pred[idx]], x=[x/360 for x in list(range(0,len(pred[idx])))],name="prediction"),
            row=4, col=1
        )

        fig.update_layout(title_text=f"{filename[idx]}, hour: {hour_id[idx]}, channel: {channel_id[idx]}")
        fig.write_image(f"figures/edf/png/{idx}.png")
        fig.write_html(f"figures/edf/html/{idx}.html")