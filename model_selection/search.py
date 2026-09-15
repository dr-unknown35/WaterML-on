from .validation import cross_val_score
import numpy as np
import itertools








def grid_search(model_factory, param_grid, X, y, nb_folds, metric, random_state=None, greater_is_better=True):
    scores= []
    keys= list(param_grid.keys())
    combinations= list(itertools.product(*param_grid.values()))
    judge=np.argmax if greater_is_better else np.argmin
    for combo in combinations:
        params= dict(zip(keys, combo))
        model= model_factory(**params)
        scores.append(cross_val_score(model,X,y,nb_folds,metric,random_state))
    best_idx=judge(scores)
    best_params = dict(zip(keys, combinations[best_idx]))
    return best_params,scores[best_idx]
