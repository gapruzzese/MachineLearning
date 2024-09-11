import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import csv

def rank_and_index_array_with_features(arr, features):
    # Check if the number of features matches the dimensions of the array
    if arr.shape[0] != len(features) or arr.shape[1] != len(features):
        raise ValueError(f"The number of features ({len(features)}) must match the dimensions of the array ({arr.shape[0]}, {arr.shape[1]}).")
    
    # Step 1: Prepare data
    # Get absolute values
    absolute_values = np.abs(arr)
    
    # Convert feature names into a 2D array for rows and columns
    feature_names_array = np.array(features)
    
    # Flatten all components
    values_flat = arr.flatten()
    absolute_values_flat = absolute_values.flatten()
    rows_flat, cols_flat = np.indices(arr.shape)
    
    # Map row and column indices to feature names
    row_names_flat = feature_names_array[rows_flat.flatten()]
    col_names_flat = feature_names_array[cols_flat.flatten()]
    
    # Step 2: Sort by absolute values and rank
    sorted_indices = np.argsort(absolute_values_flat)
    ranks = np.argsort(sorted_indices)  # This gives us the ranks for the original array
    reversed_ranks = len(ranks) - ranks  # Reverse the rank order
    
    # Step 3: Construct the output array
    # Combine the data into a single array with the required structure
    output = np.column_stack((reversed_ranks, values_flat, row_names_flat, col_names_flat))
    
    # Step 4: Remove rows where features in column 3 and 4 are the same
    mask_diff_features = output[:, 2] != output[:, 3]
    output = output[mask_diff_features]
    
    # Step 5: Remove symmetric pairs
    seen_pairs = set()
    final_output = []
    for row in output:
        pair = tuple(sorted((row[2], row[3])))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            final_output.append(row)

    # Step 6: Sort the output array by ranks
    final_output = np.array(final_output)
    final_output_sorted = final_output[final_output[:, 0].astype(int).argsort()]

    return final_output_sorted

def get_csv_headers(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)[:-1]
    return headers

if __name__ == "__main__":
    file_path = "./data/processed_sample_more_attrs.csv"
    raw = np.loadtxt(file_path, delimiter=",", skiprows=1)
    num_feats = len(get_csv_headers(file_path))
    features = get_csv_headers(file_path)
    x = raw[:, 0:num_feats]
    X = pd.DataFrame(x, columns=features)
    corr = spearmanr(X).correlation
    corr = np.nan_to_num(corr, copy=False)
    ranked = rank_and_index_array_with_features(corr, features)
    df = pd.DataFrame(ranked, columns=["Rank", "Value", "RowFeature", "ColFeature"])
    df.to_csv("./data/corr_ranked.csv", index=False)
