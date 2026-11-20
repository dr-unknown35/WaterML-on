from ..metrics.Classification import *
from ..metrics.Regression import *
import numpy as np
import copy




def cross_val_score(model, X, y, metric,response,splitter):
    scores=[]
    for train_idx,test_idx in splitter.split(X,y):
        model_copy=copy.deepcopy(model)
        model_copy.fit(X[train_idx],y[train_idx])
        if not hasattr(model_copy,response):
            raise AttributeError(f"Model has no corresponding response: '{response}'.")
        preds=getattr(model_copy,response)(X[test_idx])
        scores.append(metric(y[test_idx],preds))
    return np.mean(scores)