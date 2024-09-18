from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import mne
import numpy as np
import argparse
from utils.VAF_loader import VAF_loading
from utils.db_generator import Train_Generator
import pandas as pd
from collections import Counter

data = VAF_loading('/home/rose/Cortrium/10K_VAF_subset/VAF_subset/test/')
df = data.create_set(train=True)

#count of beat types in each strip
beats = [x for xs in df["beat_annotation"].values.tolist() for x in xs]
beat_types = set(beats)
beat_counts = []
for type in beat_types:
    beat_counts.append(beats.count(type))
dict_tmp = dict()
dict_tmp["beat_types"] = list(beat_types)
dict_tmp["beat_counts"] = beat_counts
df_tmp = pd.DataFrame(dict_tmp)

fig = px.bar(df_tmp, x="beat_types", y="beat_counts")
fig.write_image("beat_type_count.png")

#count number of beats in a strip
strip_lengths = [len(x) for x in df["beat_annotation"].values.tolist()]
strip_lengths_unique = set(strip_lengths)
strip_length_counts = []
for length in strip_lengths_unique:
    strip_length_counts.append(strip_lengths.count(length))
dict_tmp = dict()
dict_tmp["strip_length"] = list(strip_lengths_unique)
dict_tmp["strip_length_counts"] = strip_length_counts
df_tmp = pd.DataFrame(dict_tmp)

fig = px.bar(df_tmp, x="strip_length", y="strip_length_counts")
fig.write_image("strip_beat_count.png")


#distribution of strip types
strip_types = df["strip_type"].values.tolist()
strip_types_unique = set(strip_types)
strip_type_counts = []
for type in strip_types_unique:
    strip_type_counts.append(strip_types.count(type))
dict_tmp = dict()
dict_tmp["strip_type"] = list(strip_types_unique)
dict_tmp["strip_type_count"] = strip_type_counts
df_tmp = pd.DataFrame(dict_tmp)

fig = px.bar(df_tmp, x="strip_type", y="strip_type_count")
fig.write_image("strip_type_dist.png")


#distribution of strip length
strip_lengths = [len(x) for x in df["ecg"].values.tolist()]
strip_lengths_unique = set(strip_lengths)
strip_length_counts = []
for length in strip_lengths_unique:
    strip_length_counts.append(strip_lengths.count(length))
dict_tmp = dict()
dict_tmp["strip_lengths"] = list(strip_lengths_unique)
dict_tmp["strip_length_counts"] = strip_length_counts
df_tmp = pd.DataFrame(dict_tmp)

fig = px.bar(df_tmp, x="strip_lengths", y="strip_length_counts")
fig.write_image("strip_length_count.png")

#number of different types of beat in a strip
df["counts"] = df["beat_annotation"].apply(lambda x: len(Counter(x)))
unique_counts = df["counts"].values.tolist()

unique = set(unique_counts)
counts = []
for count in unique:
    counts.append(unique_counts.count(count))
dict_tmp = dict()
dict_tmp["unique_beats_in_strip"] = list(unique)
dict_tmp["unique_beats_in_strip_count"] = counts
df_tmp = pd.DataFrame(dict_tmp)

fig = px.bar(df_tmp, x="unique_beats_in_strip", y="unique_beats_in_strip_count")
fig.write_image("unique_beats_in_strip_count.png")