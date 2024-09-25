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
import yaml
import argparse
import pandas as pd

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

path = config['path']


parser = argparse.ArgumentParser(
                        prog="data_exp.py",
                        description="explore distribution of beat and strip types in VAF dataset"
)
parser.add_argument("set", help="data set to perform analysis on (train/test/validation)")
args = parser.parse_args()

dataset = args.set


data = VAF_loading(path + '/10K_VAF_subset/VAF_subset/' + dataset + '/')
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

fig = px.bar(df_tmp, 
    x="beat_types", 
    y="beat_counts", 
    text="beat_counts", 
    title=f"distribution of beat type within a strip: {dataset}"
    )
fig.write_image("figures/data_exp/beat_type_count_" + dataset + ".png")



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

fig = px.bar(df_tmp, 
    x="strip_length", 
    y="strip_length_counts", 
    title=f"count of number of beats within a strip: {dataset}"
    )
fig.write_image("figures/data_exp/strip_beat_count_" + dataset + ".png")



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

total = sum(dict_tmp["strip_type_count"])
percentages = [int(x/total * 100) for x in dict_tmp["strip_type_count"]]

fig = px.bar(
    df_tmp, 
    x="strip_type", 
    y="strip_type_count", 
    text=percentages, 
    title=f"distribution of strip types: {dataset}"
    )
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.write_image("figures/data_exp/strip_type_dist_" + dataset + ".png")



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

fig = px.bar(df_tmp, 
x="strip_lengths", 
y="strip_length_counts", 
title=f"distribution of strip length: {dataset}"
)
fig.write_image("figures/data_exp/strip_length_count_" + dataset + ".png")



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

fig = px.bar(
    df_tmp, 
    x="unique_beats_in_strip", 
    y="unique_beats_in_strip_count", 
    title=f"count of different beat types within a strip: {dataset}"
    )
fig.write_image("figures/data_exp/unique_beats_in_strip_count_" + dataset + ".png")
fig.write_html("figures/data_exp/unique_beats_in_strip_count_" + dataset + ".html")