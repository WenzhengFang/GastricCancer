import sys
import os

scriptpath = os.path.abspath(__file__)
scridir = os.path.dirname(scriptpath)
sys.path.append(os.path.join(scridir, '..', '..'))

from sklearn.feature_selection import VarianceThreshold


def low_variance_feature_selection(X, threshold):
    """
    This function implements the low_variance feature selection (existing method in scikit-learn)

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    p:{float}
        parameter used to calculate the threshold(threshold = p*(1-p))

    Output
    ------
    X_new: {numpy array}, shape (n_samples, n_selected_features)
        data with selected features
    """
    sel = VarianceThreshold(threshold)
    return sel.fit_transform(X)