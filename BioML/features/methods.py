"""
This module contains the methods used for feature selection.
"""
from pathlib import Path
import numpy as np
from typing import Iterable
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, r_regression
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.linear_model import RidgeClassifier, Ridge
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import pandas as pd


def calculate_shap_importance(model: xgb.XGBClassifier | xgb.XGBRegressor, X_train: pd.DataFrame | np.ndarray, 
                              Y_train: pd.Series | np.ndarray, feature_names: Iterable[str]):
    """
    Calculates the SHAP importance values for the given XGBoost model and training data.

    Parameters
    ----------
    model : xgb.XGBClassifier or xgb.XGBRegressor
    The trained XGBoost model to calculate SHAP importance for.
    X_train : pd.DataFrame or np.ndarray
        The training data used to train the model.
    Y_train : pd.Series or np.ndarray
        The target values used to train the model.
    feature_names : Iterable[str]
        The names of the features in the training data.

    Returns
    -------
    Tuple[pd.Series, np.ndarray]
        A tuple containing the SHAP importance values as a pandas Series and the SHAP values as a numpy array.
    """
    xgboost_explainer = shap.TreeExplainer(model, X_train, feature_names=feature_names)
    shap_values = xgboost_explainer.shap_values(X_train, Y_train)
    shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), feature_names).sort_values(ascending=False) # type: ignore
    shap_importance = shap_importance.loc[lambda x: x > 0]
    return shap_importance, shap_values


def plot_shap_importance(shap_values: np.ndarray, feature_names: Iterable[str], output_path: Path, 
                         X_train: pd.DataFrame | np.ndarray | None=None, plot_num_features: int=20, dpi=500):
    """
    Plots the SHAP importance values for the given feature names.

    Parameters
    ----------
    shap_values : np.ndarray
        The SHAP values to plot.
    feature_names : Iterable[str]
        The names of the features in the SHAP values.
    output_path : Path or None, optional
        The output directory to save the plot in, by default None.
    X_train : pd.DataFrame or np.ndarray, optional
        The training data used to train the model, by default None.
    plot_num_features : int, optional
        The number of features to plot, by default 20.
    dpi : int, optional
        The resolution of the saved plot, by default 500.

    Returns
    -------
    None
    """
    
    shap_dir = (output_path / "shap_features")
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type='bar', show=False,
                        max_display=plot_num_features)
    plt.savefig(shap_dir / f"shap_top_{plot_num_features}_features.png", dpi=dpi)
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False,
                        max_display=plot_num_features)
    plt.savefig(shap_dir / f"feature_influence_on_model_prediction.png", dpi=dpi)


def fechner_corr(x, y):
    """Calculate Sample sign correlation (Fechner correlation) for each
    feature. Bigger absolute values mean more important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> fechner_corr(x, y)
    array([-0.2,  0.2, -0.4, -0.2, -0.2])
    """
    y_dev = y - np.mean(y)
    x_dev = x - np.mean(x, axis=0)
    return np.sum(np.sign(x_dev.T * y_dev), axis=1) / x.shape[0]


def kendall_corr(x, y):
    """Calculate Sample sign correlation (Kendall correlation) for each
    feature. Bigger absolute values mean more important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> kendall_corr(x, y)
    array([-0.1,  0.2, -0.4, -0.2,  0.2])
    """
    def __kendall_corr(feature):
        k_corr = 0.0
        for i, feat in enumerate(feature):
            k_corr += np.sum(np.sign(feat - feature[i + 1:])
                             * np.sign(y[i] - y[i + 1:]))
        return 2 * k_corr / (feature.shape[0] * (feature.shape[0] - 1))

    return np.apply_along_axis(__kendall_corr, 0, x)


def chi2_measure(x, y):
    """Calculate the Chi-squared measure for each feature. Bigger values mean
    more important features. This measure works best with discrete features due
    to being based on statistics.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    http://lkm.fri.uni-lj.si/xaigor/slo/clanki/ijcai95z.pdf

    Example
    -------
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> chi2_measure(x, y)
    array([ 1.875     ,  0.83333333, 10.        ,  3.75      ,  6.66666667])
    """
    def __chi2(feature):
        values, counts = np.unique(feature, return_counts=True)
        values_map = {val: idx for idx, val in enumerate(values)}
        splits = {cl: np.array([values_map[val] for val in feature[y == cl]]) 
            for cl in classes}
        e = np.vectorize(
            lambda cl: prior_probs[cl] * counts,
            signature='()->(1)')(classes)
        n = np.vectorize(
            lambda cl: np.bincount(splits[cl], minlength=values.shape[0]),
            signature='()->(1)')(classes)
        return np.sum(np.square(e - n) / e)

    classes, counts = np.unique(y, return_counts=True)
    prior_probs = {cl: counts[idx] / x.shape[0] for idx, cl
        in enumerate(classes)}
    
    return np.apply_along_axis(__chi2, 0, x)


def classification_filters(X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, 
                           num_features: int, feature_names: Iterable[str], filter_name: str) -> pd.Series:
    """
    Perform univariate feature selection.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        The training feature data.
    Y_train : pd.Series or np.ndarray
        The training label data.
    num_features : int
        The number of features or columns to select.
    feature_names : Iterable[str]
        The names of the features or columns.
    filter_name : str
        The name of the statistical filter to use.

    Returns
    -------
    pd.Series
        A series containing the feature scores, sorted in descending order.
    """
    filters = {"Fscore": f_classif, "mutual_info": mutual_info_classif, "chi2": chi2_measure, 
               "FechnerCorr": fechner_corr, "KendallCorr": kendall_corr}
    ufilter = SelectKBest(filters[filter_name], k=num_features)
    ufilter.fit(X_train, Y_train)
    scores = dict(zip(feature_names, ufilter.scores_))

    # sorting the features
    if filter_name in ["KendallCorr", "FechnerCorr"]:
        scores = pd.Series(dict(sorted(scores.items(), key=lambda items: abs(items[1]), reverse=True)))
    else:
        scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1], reverse=True)))
    return scores


def random_forest(X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray,
                  feature_names: Iterable[str], treemodel: rfc | rfr=rfc, # type: ignore
                  seed: int=123, num_threads: int=-1) -> pd.Series:
    """
    Perform feature selection using a random forest model.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        The training feature data.
    Y_train : pd.Series or np.ndarray
        The training label data.
    feature_names : list or np.ndarray
        The names of the features.
    treemodel : RandomForestClassifier o RandomForestRegressor, optional
        The random forest model to use. The default is rfc.
    seed : int, optional
        The random seed to use. The default is 123.
    num_threads : int, optional
        The number of threads to use for parallel processing. The default is -1.

    Returns
    -------
    pd.Series
        A series containing the feature importances, sorted in descending order.
    """
    forest_model = treemodel(random_state=seed, max_features=0.7, max_samples=0.8, min_samples_split=6, n_estimators=200, 
                             n_jobs=num_threads, min_impurity_decrease=0.1) # type: ignore
    forest_model.fit(X_train, Y_train)
    gini_importance = pd.Series(forest_model.feature_importances_, index=feature_names) # type: ignore
    gini_importance.sort_values(ascending=False, inplace=True)

    return gini_importance


def xgbtree(X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, 
            xgboosmodel: xgb.XGBClassifier | xgb.XGBRegressor,  # type: ignore
            feature_names: Iterable[str], seed: int=123, num_threads: int=-1) -> xgb.XGBClassifier | xgb.XGBRegressor:
    """
    Perform feature selection using a xgboost model.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        The training feature data.
    Y_train : pd.Series or np.ndarray
        The training label data.
    xgboostmodel : xgb.XGBClassifier or xgb.XGBRegressor, optional
        The type of problem to solve. Defaults to XGBClassifier.
    seed : int, optional
        The random seed to use. The default is 123.
    num_threads : int, optional
        The number of threads to use for parallel processing. The default is -1.

    Returns
    -------
    pd.Series
        A series containing the feature importances, sorted in descending order.
    """

    XGBOOST = xgboosmodel(learning_rate=0.01, n_estimators=200, max_depth=4, gamma=0,
                            subsample=0.8, colsample_bytree=0.8, nthread=num_threads, seed=seed) # type: ignore
    # Train the model
    XGBOOST.fit(X_train, Y_train)
    shap_importance, shap_values = calculate_shap_importance(XGBOOST, X_train, Y_train, feature_names)

    return shap_importance, shap_values


def rfe_linear(X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, num_features: int, seed: int,
                feature_names: Iterable[str], step: int=30, ridgemodel: RidgeClassifier | Ridge =RidgeClassifier) -> list[str]: # type: ignore
    """
    Perform feature selection using recursive feature elimination with a linear model.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        The training feature data.
    Y_train : pd.Series or np.ndarray
        The training label data.
    num_features : int
        The number of features to select.
    feature_names : list or np.ndarray
        The names of the features.
    step : int, optional
        The number of features to remove at each iteration. Defaults to 30.
    ridge_model : RidgeClassifier or Ridge
        The model for the selection, it depends on the problem, classification or regression.

    Returns
    -------
    list
        A list of the selected feature names.
    """
    linear_model = ridgemodel(random_state=seed, alpha=4)
    rfe = RFE(estimator=linear_model, n_features_to_select=num_features, step=step)
    rfe.fit(X_train, Y_train)
    features = rfe.get_feature_names_out(feature_names)
    return features
 

def regression_filters(X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, feature_nums: int, 
                    feature_names: Iterable[str], reg_func: str) -> pd.Series:
    """
    Perform feature selection using regression filters.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        The training feature data.
    Y_train : pd.Series or np.ndarray
        The training label data.
    feature_nums : int
        The number of features to select.
    feature_names : list or np.ndarray
        The names of the features.
    reg_func : str
        The name of the regression filter to use. Available options are "mutual_info" and "Fscore".

    Returns
    -------
    pd.Series
        A series containing the feature scores, sorted in descending order.
    """
    reg_filters = {"mutual_info": mutual_info_regression, "Fscore":f_regression, "PCC": r_regression}
    sel = SelectKBest(reg_filters[reg_func], k=feature_nums)
    sel.fit(X_train, Y_train)
    scores = sel.scores_
    scores = dict(zip(feature_names, scores)) 
    scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1], reverse=True)))
    return scores