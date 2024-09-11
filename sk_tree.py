import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

if __name__ == "__main__":
    raw = np.loadtxt("./data/processed_sample.csv", delimiter=",")
    x = raw[:, 0:28]
    y = raw[:, 28:]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)
    features = [
        "Not Widely Spoken",
        "Widely Spoken",
        "Central Asia",
        "Middle East & North Africa",
        "The North & Australasia",
        "South Asia",
        "Sub-Saharan Africa",
        "Central & South America",
        "East Asia",
        "Southeast Asia",
        "Released",
        "Detained",
        "Never Detained",
        "Midwest",
        "Southwest",
        "West",
        "Northeast",
        "Southeast",
        "Canada",
        "Mexico",
        "e_28_date - binary",
        "c_asy_type - binary",
        "crim_chgs",
        "length_of_proc_1_5",
        "length_of_proc_2_5",
        "length_of_proc_3_5",
        "length_of_proc_4_5",
        "length_of_proc_5_5",
    ]
    # Feature importance metric to summarize all the splits in the tree and show that these seem quite important in making the decision
    # groups that are heterogenous enough and see how the feature importance
    # Dedicated docket 
    # Time dimension and survival models (Cos regression, if we think about court's decision time, what delays  the court)
    # Random forest on all people who stayed and have the outcome variable be length of their proceeding
    # Judge fixed effects, probit regression outcome variable with fixed effects.
    # Linear regression a bunch of columns one hot coded for, a column with a judge id
    # Judge experience, asylum approval rate, 
    # One hot encoding judges.
    # Not one hot encoding hearing location, but splitting on detention facility or immigration court
    # Unsupervised clustering to see similarities between certain judges
    # Bias in, Bias out

    tree.plot_tree(
        clf,
        feature_names=features,
        class_names=["Leave", "Stay"],
        max_depth=3,
        filled=True,
        fontsize=8,
        label="none",
    )
    plt.show()
