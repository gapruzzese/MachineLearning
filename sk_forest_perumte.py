import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
import pandas as pd
# from sklearn.inspection import permutation_importance
# from scipy.cluster import hierarchy
# from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
# from collections import defaultdict
import csv

# def plot_permutation_importance(clf, X, y, ax):
#     result = permutation_importance(clf, X, y, n_repeats=3, random_state=42, n_jobs=-1)
#     perm_sorted_idx = result.importances_mean.argsort()

#     ax.boxplot(
#         result.importances[perm_sorted_idx].T,
#         vert=False,
#         labels=X.columns[perm_sorted_idx],
#     )
#     ax.axvline(x=0, color="k", linestyle="--")
#     return ax

def get_csv_headers(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)[:-1]
    return headers

if __name__ == "__main__":
    
    # change file_path to reflect which subset is being run
    file_path = "./data/processed_sample_more_attrs.csv"
    raw = np.loadtxt(file_path, delimiter=",", skiprows=1)
    num_feats = len(get_csv_headers(file_path))
    features = get_csv_headers(file_path)
    
    x = raw[:, 0:num_feats]
    y = np.ravel(raw[:, num_feats:])
    y = pd.DataFrame(y)
    X = pd.DataFrame(x, columns=features)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    forest = RandomForestClassifier()
    forest.fit(X_train, y_train)
    
    num_trees = 3
    fig, axes = plt.subplots(nrows=num_trees, ncols=1, figsize=(30, 40))
    
    for i in range(num_trees):
        tree = forest.estimators_[i]
        plot_tree(tree, ax=axes[i],
                  feature_names=features, 
                  class_names=["Leave", "Stay"], 
                  filled=True, max_depth=2, fontsize=27)
        axes[i].set_title(f'Tree {i+1}')
    
    plt.tight_layout()
    plt.show()
    
    y_pred = forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    feature_imp = pd.Series(forest.feature_importances_, 
                            index=features).sort_values(ascending=False)
    
    
    # clf = forest
    
    # mdi_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    # tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
    # tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5
    
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    # mdi_importances.sort_values().plot.barh(ax=ax1)
    # ax1.set_xlabel("Gini importance")
    # plot_permutation_importance(clf, X_train, y_train, ax2)
    # ax2.set_xlabel("Decrease in accuracy score")
    # fig.suptitle(
    #     "Impurity-based vs. permutation importances on multicollinear features (train set)"
    # )
    # _ = fig.tight_layout()
    
    
    # fig, ax = plt.subplots(figsize=(7, 6))
    # plot_permutation_importance(clf, X_test, y_test, ax)
    # ax.set_title("Permutation Importances on multicollinear features\n(test set)")
    # ax.set_xlabel("Decrease in accuracy score")
    # _ = ax.figure.tight_layout()
        
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 16))
    # corr = spearmanr(X).correlation
    
    # # Ensure the correlation matrix is symmetric
    # corr = (corr + corr.T) / 2
    # np.fill_diagonal(corr, 1)
    # corr = np.nan_to_num(corr, copy=False)

    
    # # We convert the correlation matrix to a distance matrix before performing
    # # hierarchical clustering using Ward's linkage.
    # distance_matrix = 1 - np.abs(corr)
    # dist_linkage = hierarchy.ward(squareform(distance_matrix))
    # dendro = hierarchy.dendrogram(
    #     dist_linkage, labels=X.columns.to_list(), ax=ax1, leaf_rotation=90
    # )
    # dendro_idx = np.arange(0, len(dendro["ivl"]))
    
    # ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    # ax2.set_xticks(dendro_idx)
    # ax2.set_yticks(dendro_idx)
    # ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    # ax2.set_yticklabels(dendro["ivl"])
    # _ = fig.tight_layout()
        
    fig, ax = plt.subplots(figsize=(20, 16))

    # Calculate Spearman correlation
    corr = spearmanr(X).correlation
    
    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    corr = np.nan_to_num(corr, copy=False)
    
    # Plot heatmap
    heatmap = ax.imshow(corr, cmap='coolwarm')
    
    # Customize the heatmap appearance
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_yticklabels(features)
    

    # Ensure proper layout
    plt.tight_layout()
    
    plt.show()

