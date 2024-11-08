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
from utils.db_loader import DB_loading
import yaml


#config
with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

target_lv = cfg['target_level']
fs_resampling = cfg['fs_resampling']
duration = cfg['label_window_duration'] 

class NP_loading(DB_loading):
    def __init__(self, array):
        self.data = array

    def create_set(self, use_swt=True): 
        ecg = self.data
        fs = 256 #Hz

        set_dict = dict()
        set_dict['ecg'] = []
        set_dict['feature'] = []
        set_dict['target'] = []
        set_dict['label'] = []
        set_dict['mask_array'] = []

        ecg = self.resample_ecg(ecg, fs, fs_resampling)
        feature = self.transform(ecg, use_swt=use_swt)
        target = np.array([])
        label = np.array([])
        mask_array = np.array([])
         
        set_dict['ecg'].append(ecg)
        set_dict['feature'].append(feature)
        set_dict['target'].append(target)
        set_dict['label'].append(label)
        set_dict['mask_array'].append(mask_array)

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

        file = filename[idx].split("/")[-1]
        strip_idx = strip_id[idx]-1

        ecg_v, label_v, start_times_v, fs_v, lead_off_v, strip_type_v, mask_v = self.load_data(filename[idx])
        usable_channels, _ = return_good_ecg_channel_idx_based_on_lead_off(ecg_v[strip_idx], lead_off_v[strip_idx], 3)
        usable_channels.remove(channel_id[idx])

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
