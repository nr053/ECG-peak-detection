import pickle
from plotly.subplots import make_subplots
import plotly.graph_objects as go
#from helper_functions import return_good_ecg_channel_idx_based_on_lead_off
import mne
import numpy as np
import argparse
from utils.evaluator import Evaluator

#edf
#raw = mne.io.read_raw_edf("/home/rose/Cortrium/ECG-peak-detection/edf1.edf", preload=True)
#data = raw.get_data()

#vaf
with open("/home/rose/Cortrium/R-peak-detector/VAF_subset/test/00a4aad0-5d1b-4456-a6f0-8e5c13d0feda.pkl", 'rb') as file:
    vaf = pickle.load(file)



ecg = vaf["strip_ecg"][0][0]

detector = Evaluator(ecg)
detector.load(ecg)
peaks = detector.find_peaks()