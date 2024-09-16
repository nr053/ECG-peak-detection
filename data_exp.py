from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import mne
import numpy as np
import argparse
from utils.VAF_loader import VAF_loading
from utils.db_generator import Train_Generator


data = VAF_loading('/home/rose/Cortrium/10K_VAF_subset/VAF_subset/test/')
df = data.create_set(train=True)

#count of beat types in each strip
beats = [x for xs in df["beat_annotation"].values.tolist() for x in xs]
beat_types = set(beats)
beat_counts = []
for type in beat_types:
    beat_counts.append(beats.count(type))
beat_dist = dict(zip(beat_types, beat_counts))

fig = px.bar(beat_dist)
fig.write_image("foo.png")