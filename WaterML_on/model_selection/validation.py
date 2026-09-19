from ..data_utils.splitter import Kfolds_split
from ..metrics.Classification import *
from ..metrics.Regression import *
import numpy as np
import copy




def cross_val_score(model, X, y, nb_folds, metric, random_state=None):
    scores=[]
    for train_idx,test_idx in Kfolds_split(X.shape[0],nb_folds,random_state):
        model_copy=copy.deepcopy(model)
        model_copy.fit(X[train_idx],y[train_idx])
        preds=model_copy.predict(X[test_idx])
        scores.append(metric(y[test_idx],preds))
    return np.mean(scores)