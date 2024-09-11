import json
import math

from lib.util_csv import *
from lib.util_data import *
from lib.util_tree import build_tree
from lib.util_viz_tree import *
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt



if __name__ == "__main__":
    in_file = "./data/more_attrs_counts.csv"
    out_file = "./data/processed_sample_more_attrs.csv"
    # in_file = "./data/sample_test.csv"
    # out_file = "./data/proc_sample_test.csv"
    LOOKUPS = json.load(open(f"./json/lookups.json", "r"))
    DECODE = json.load(open(f"./json/decode.json", "r"))

    # ==================================================
    # All Features
    # Each tuple has instructions on processing this feature
    # Lookups will binarify categorical variables
    # null_not_null will convert null values to 0 not null to 1
    # edge will call an edge cases function
    # classifier is the attribute to predict
    # ==================================================
    features = [
        ("alien_state", "lookup", "value"),
        ("nat", "lookup", "value"),
        ("lang", "lookup", "value"),
        ("custody", "lookup", "key"),
        ("e_28_date", "null_not_null",""),
        ("is_stay", "classifier",""),
        ("osc_date", "no_action",""),
        ("crim_ind", "edge", ""),
        ("input_date", "no_action",""),
        ("comp_date", "no_action",""),
        ("charges", "no_action",""),
        ("base_city_code", "no_action",""),
        ("hearing_loc_code", "no_action",""),
        ("absentia", "edge",""),
        ("ij_code", "no_action",""),
        ("hearing_date", "no_action",""),
        ("c_asy_type", "null_not_null",""),
        ("total_applications", "no_action",""),
        ("grant_proportion", "no_action",""),
        ("denial_proportion", "no_action",""),
        ("is_family", "no_action",""),
        ("applications", "no_action",""),
        ("continuances", "no_action",""),
        ("charges", "edge","")
    ]

    # ==================================================
    # Prep Decode Values
    # ==================================================
    init_features = [f[0] for f in features]

    # Trim to feature name and categories (e.g. 'alien_state':['MA', 'NY'])
    f_lookup = [f[0] for f in features if f[1] == "lookup"]
    feature_lookup_pairs = {k:LOOKUPS.get(v) for k,v in DECODE.items() if k in f_lookup}

    f_binary = [f[0] for f in features if f[1] == "null_not_null"]

    f_edge = [f[0] for f in features if f[1] == "edge"]

    raw_data = csv_gen(in_file)
    print("Cleaning data...")
    data = list(process_data(raw_data, init_features, feature_lookup_pairs, f_binary, f_edge))


    filtered_data = [row for row in data if not row[0]['length_of_proc'] == None]
    print(str(len(data) - len(filtered_data)) + 'Removed Rows')
    data = filtered_data
    filtered_data = [row for row in data if not row[0]['length_of_proc'] == None]
    # embed(config=c)

    print("Data processed. Cleaning continuous data...")
    data = continuous_to_percentile(data, "length_of_proc", n_bins=5)
    data = continuous_to_percentile(data, "denial_proportion", n_bins=5)



    pop_list = ['admin_closure_proportion', 'other_proportion']
    for row in data:
        for feature in pop_list:
            row[0].pop(feature)


    print(f"Writing data...")
    bad_rows = 0
    for i, row in enumerate(data):
        if len(row[0]) != 58:
            bad_rows +=1
            continue
        row[0].update({'is_stay':row[1]})
        write_data(row[0], out_file)
    print("Data cleaned!")
    print(f"{bad_rows} bad rows removed")

