import numpy as np
import pandas as pd
from typing import Iterable
from scipy.stats import pearsonr


def ndcg_at_k(true_relevance: pd.Series, predicted_ranking: Iterable[int | float], 
              k: int=10, form: str="exp", penalty: int=15, **kwargs):
    """
    Compute the Normalized Discounted Cumulative Gain at k (NDCG@k) for a given ranking.

    Parameters
    ----------
    true_relevance : pd.Series
        The true relevance scores for the items in the ranking.
    predicted_ranking : Iterable[int  |  float]
        The predicted relevance score of the items.
    k : int, optional
        How many items to consider, by default 6
    form : str, optional
        What form to use, by default "exp"

    Returns
    -------
    _ : float
        The NDCG@k score.
    """
    # Sort items by predicted ranking

    true_relevance_scaled = (true_relevance - min(true_relevance)) / (max(true_relevance) - min(true_relevance))
    
    sorted_indices = np.argsort(predicted_ranking)[::-1]  # sort the predictions in descending order
    true_relevance_sorted = true_relevance_scaled.iloc[sorted_indices] 
    # It is scaled but the ranking is preserved

    # sort the true relevance according to the predicted ranking
    if form == "exponential" or form == "exp":
        # Calculate DCG@k
        # I don't use the predicted relevance score to calculate the score just the order
        # With higher penalty we give more relevance to the relevance score not only order 
        dcg = sum((penalty ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(true_relevance_sorted[:k]))
        
        true_relevance_sorted = true_relevance_sorted.sort_values(ascending=False)
        # Calculate IDCG@k
        idcg = sum((penalty ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(true_relevance_sorted[:k]))
    elif form == "linear":
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(true_relevance_sorted[:k]))
        
        true_relevance_sorted = true_relevance_sorted.sort_values(ascending=False)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(true_relevance_sorted[:k]))
    
    # Calculate NDCG@k
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return ndcg

def pear_r(true_y: Iterable[int], y_pred: Iterable[int], alternative: str="greater", 
           penalize: bool=True, **kwargs):
    """
    Compute the Pearson correlation coefficient between the true and predicted values.
    It can be used to see if NDCG@k is a good metric for the model.

    Parameters
    ----------
    true_y : Iterable[int]
        The true values.
    y_pred : Iterable[int]
        The predicted values.
    alternative : str, optional
        The alternative hypothesis, by default "greater"
    penalize : bool, optional
        Whether to penalize the correlation coefficient by p-value, by default True
        
    Returns
    -------
    _ : float
        The Pearson correlation coefficient.
    """
    correlate = pearsonr(true_y, y_pred, alternative=alternative)
    if penalize:
        p = 0.5 if correlate[1] > 0.05 else 1
        return correlate[0] * p
    return correlate[0]