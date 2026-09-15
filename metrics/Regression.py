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
    return 100* np.mean(np.abs(y_target-y_pred) /y_target)