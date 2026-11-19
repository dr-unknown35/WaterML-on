import numpy as np




def MSE(y_pred,y_target):
    return np.mean((y_target-y_pred)**2)


def MAE(y_pred,y_target):
    return np.mean(np.abs(y_target-y_pred))


def RMSE(y_pred,y_target):
    return np.sqrt(MSE(y_pred,y_target))

def r2_score(y_pred,y_target):
    y_mean=np.mean(y_target,axis=0)
    ssr=(np.sum((y_target-y_pred)**2))
    sst=(np.sum((y_target-y_mean)**2))
    return 1- (ssr/sst)

def MAPE(y_pred,y_target):
    return np.mean(np.abs(y_target-y_pred) /y_target)


def poisson_deviance(y_true,y_pred):
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)

    
    if y_true.shape!=y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")


    y_true=y_true.ravel()
    y_pred=np.maximum(y_pred.ravel(),1e-15)
    safe_true=np.where(y_true>0,y_true,1)
    return 2 * np.sum(y_true*np.log(safe_true/y_pred) -(y_true-y_pred))

     