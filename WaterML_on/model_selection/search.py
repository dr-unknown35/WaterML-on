from .validation import cross_val_score
from ..data_utils.splitter import Kfolds
import numpy as np
import itertools








def grid_search(model_factory, param_grid, X, y, metric,response="predict", splitter=None,greater_is_better=True):
    scores= []
    splitter=splitter if splitter is not None else Kfolds(5)
    keys= list(param_grid.keys())
    combinations= list(itertools.product(*param_grid.values()))
    judge=np.nanargmax if greater_is_better else np.nanargmin
    for combo in combinations:
        params= dict(zip(keys, combo))
        model= model_factory(**params)
        if not hasattr(model,response):
                    raise AttributeError(f"Model has no corresponding response: '{response}'.")
        scores.append(cross_val_score(model,X,y,metric,response,splitter))
    best_idx=judge(scores)
    best_params = dict(zip(keys, combinations[best_idx]))
    return best_params,scores[best_idx],list(zip(keys,scores))
