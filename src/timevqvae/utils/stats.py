import numpy as np
from sklearn.ensemble import IsolationForest


def remove_outliers(data: np.ndarray):
    """Filter outliers from (B, D) data using IsolationForest."""
    iso_forest = IsolationForest(max_samples=0.9, contamination=0.1, random_state=0)
    inliers = iso_forest.fit_predict(data) == 1
    return data[inliers]
