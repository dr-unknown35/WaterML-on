from .validation import cross_val_score
from ..data_utils.splitter import Kfolds
import numpy as np
import itertools








def grid_search(model_factory, param_grid, X, y, metric,response="predict", splitter=None,greater_is_better=True):
    scores= []
    param_list= []
    splitter=splitter if splitter is not None else Kfolds(5)
    keys= list(param_grid.keys())
    combinations= list(itertools.product(*param_grid.values()))
    judge=np.nanargmax if greater_is_better else np.nanargmin
    
    
    for combo in combinations:
        params= dict(zip(keys, combo))
        param_list.append(params)
        model= model_factory(**params)
        scores.append(cross_val_score(model,X,y,metric,response,splitter))
        
        
        
    best_idx=judge(scores)
    return param_list[best_idx],scores[best_idx],list(zip(param_list,scores))
