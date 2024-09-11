import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
import pandas as pd

from IPython import embed
from traitlets.config import get_config

c = get_config()
c.InteractiveShellEmbed.colors = "Linux"


if __name__ == "__main__":
    raw = np.loadtxt(
        "./data/proc_sample_test.csv", delimiter=",", skiprows=1
    )
    features = [
        'Not Widely Spoken',
        'Widely Spoken',
        'Central & South America',
        'Central Asia',
        'Sub-Saharan Africa',
        'Southeast Asia',
        'South Asia',
        'Middle East & North Africa',
        'East Asia',
        'South Pacific',
        'The North & Australasia',
        'UNKNOWN REGION',
        'Released',
        'Detained',
        'Never Detained',
        'Midwest',
        'Northeast',
        'Southwest',
        'Puerto Rico',
        'Northern Mariana Islands',
        'Southeast',
        'West',
        'Virgin Islands',
        'Mexico',
        'Canada',
        'Foreign Address',
        'e_28_date - binary',
        'c_asy_type - binary',
        'app_clinton',
        'app_bush',
        'app_obama',
        'app_trump',
        'app_biden',
        'drug infraction',
        'national security infraction',
        'immigration infraction',
        'criminal infraction',
        'inadmissible',
        'deportable',
        'detained_docket',
        'asylum_rule_docket',
        'juvenile',
        'dedicated_docket',
        'port_of_entry',
        'low_asylum_grant',
        'low_experience',
        'absentia',
        'is_family',
        'length_of_proc_1_5',
        'length_of_proc_2_5',
        'length_of_proc_3_5',
        'length_of_proc_4_5',
        'length_of_proc_5_5',
        'denial_proportion_1_5',
        'denial_proportion_2_5',
        'denial_proportion_3_5',
        'denial_proportion_4_5',
        'denial_proportion_5_5']

    num_feats = len(features)
    x = raw[:, 0:num_feats]
    y = np.ravel(raw[:, num_feats:])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    forest = RandomForestClassifier(max_depth=3)
    forest.fit(X_train, y_train)

    num_trees = 3
    fig, axes = plt.subplots(nrows=1, ncols=num_trees, figsize=(40, 10))

    # Plot each tree
    for i in range(num_trees):
        tree = forest.estimators_[i]
        plot_tree(
            tree,
            ax=axes[i],
            feature_names=features,
            class_names=["Leave", "Stay"],
            filled=True,
            max_depth=3,
            fontsize=10,
        )
        axes[i].set_title(f"Tree {i+1}")

    plt.tight_layout()
    plt.show()

    y_pred = forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    feature_imp = pd.Series(forest.feature_importances_, index=features).sort_values(
        ascending=False
    )
    embed(config=c)
