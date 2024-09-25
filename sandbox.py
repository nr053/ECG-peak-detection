import pickle
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils.helper_functions import return_good_ecg_channel_idx_based_on_lead_off
import mne
import numpy as np
import argparse
from utils.VAF_loader import VAF_loading
from utils.db_generator import Train_Generator

# parser = argparse.ArgumentParser()
# parser.add_argument("-v", "--Visualise", help="visualise examples of poor performance", action='store_true')
# parser.add_argument("-d", "--Data", help="database to use")
# args = parser.parse_args()

# print(args.Visualise)
# print(args.Data)

#vaf
with open("/home/rose/Cortrium/R-peak-detector/VAF_subset/test/00a4aad0-5d1b-4456-a6f0-8e5c13d0feda.pkl", 'rb') as file:
    vaf = pickle.load(file)


from utils.evaluator import Evaluator

#edf
#raw = mne.io.read_raw_edf("/home/rose/Cortrium/ECG-peak-detection/edf1.edf", preload=True)
#data = raw.get_data()


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
