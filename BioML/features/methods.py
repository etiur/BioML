
import pandas as pd
import numpy as np
from typing import Iterable
from ITMO_FS.filters.multivariate import STIR, TraceRatioFisher
from ITMO_FS.filters.univariate import UnivariateFilter, select_k_best
from ITMO_FS.filters.unsupervised import TraceRatioLaplacian
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from pathlib import Path
from sklearn.linear_model import RidgeClassifier, Ridge


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
    shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), feature_names).sort_values(ascending=False)
    shap_importance = shap_importance.loc[lambda x: x > 0]
    return shap_importance, shap_values


def plot_shap_importance(shap_values, feature_names: Iterable[str], output_path: Path, 
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


def univariate(X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, 
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
    ufilter = UnivariateFilter(filter_name, select_k_best(num_features))
    ufilter.fit(X_train, Y_train)
    scores = {x: v for x, v in zip(feature_names, ufilter.feature_scores_)}
    # sorting the features
    if filter_name != "LaplacianScore":
        if filter_name in ["SpearmanCorr", "PearsonCorr", "KendallCorr", "FechnerCorr"]:
            scores = pd.Series(dict(sorted(scores.items(), key=lambda items: abs(items[1]), reverse=True)))
        else:
            scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1], reverse=True)))
    else:
        scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1])))
    return scores


def multivariate(X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, 
                    num_features: int, feature_names: Iterable[str], filter_name: str) -> pd.Series:
    """
    Perform multivariate feature selection.

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
    filter_name : str
        The name of the statistical filter to use.

    Returns
    -------
    pd.Series
        A series containing the feature scores, sorted in descending order.
    """
    if filter_name == "STIR":
        ufilter = STIR(num_features, k=5).fit(X_train, Y_train)
        scores = {x: v for x, v in zip(feature_names, ufilter.feature_scores_)}
    else:
        ufilter = TraceRatioFisher(num_features).fit(X_train, Y_train)
        scores = {x: v for x, v in zip(feature_names, ufilter.score_)}
    scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1], reverse=True)))
    return scores


def unsupervised(X_train: pd.DataFrame | np.ndarray, num_features: int, 
                    feature_names: Iterable[str], filter_name: str) -> pd.Series:
    """
    Perform unsupervised feature selection using a statistical filter.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        The training feature data.
    num_features : int
        The number of features to select.
    feature_names : list or np.ndarray
        The names of the features.
    filter_name : str
        The name of the statistical filter to use.

    Returns
    -------
    pd.Series
        A series containing the feature scores, sorted in descending order.
    """
    if "Trace" in filter_name:
        ufilter = TraceRatioLaplacian(num_features).fit(X_train)
        scores = {x: v for x, v in zip(feature_names, ufilter.score_)}

    scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1], reverse=True)))

    return scores


def random_forest(X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray,
                  feature_names: Iterable[str], treemodel: rfc | rfr=rfc, 
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
                                n_jobs=num_threads, min_impurity_decrease=0.1)
    forest_model.fit(X_train, Y_train)
    gini_importance = pd.Series(forest_model.feature_importances_, index=feature_names)
    gini_importance.sort_values(ascending=False, inplace=True)

    return gini_importance


def xgbtree(X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, 
            xgboosmodel: xgb.XGBClassifier | xgb.XGBRegressor= xgb.XGBClassifier, 
            seed: int=123, num_threads: int=-1) -> xgb.XGBClassifier | xgb.XGBRegressor:
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
                            subsample=0.8, colsample_bytree=0.8, nthread=num_threads, seed=seed)
    # Train the model
    XGBOOST.fit(X_train, Y_train)

    return XGBOOST


def rfe_linear(X_train: pd.DataFrame | np.ndarray, Y_train: pd.Series | np.ndarray, num_features: int, seed: int,
                feature_names: Iterable[str], step: int=30, ridgemodel: RidgeClassifier | Ridge =RidgeClassifier) -> list[str]:
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
    reg_filters = {"mutual_info": mutual_info_regression, "Fscore":f_regression}
    sel = SelectKBest(reg_filters[reg_func], k=feature_nums)
    sel.fit(X_train, Y_train)
    scores = sel.scores_
    scores = {x: v for x, v in zip(feature_names, scores)}
    scores = pd.Series(dict(sorted(scores.items(), key=lambda items: items[1], reverse=True)))
    return scores