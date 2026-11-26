import numpy as np




def MSE(y_pred,y_target):
    return np.mean((y_target-y_pred)**2)


def MAE(y_pred,y_target):
    return np.mean(np.abs(y_target-y_pred))


def RMSE(y_pred,y_target):
    return np.sqrt(MSE(y_pred,y_target))

def r2_score(y_pred,y_target):
    y_pred=np.asarray(y_pred)
    y_target=np.asarray(y_target)
    
    
    if y_pred.shape!=y_target.shape:
        raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} != y_target {y_target.shape}")
    
    
    y_mean=np.mean(y_target,axis=0)
    ssr=(np.sum((y_target-y_pred)**2))
    sst=(np.sum((y_target-y_mean)**2))
    
    
    return 1.0 if sst==0 and ssr==0 else 0.0 if sst==0 else  1- (ssr/sst)





def MAPE(y_pred,y_target,ignore_zeros=False):
    y_pred=np.asarray(y_pred)
    y_target=np.asarray(y_target)
    
    if y_pred.ndim==0 or y_target.ndim==0:
            raise ValueError(f"Shape error: argument should be array but got scalar instead")
    if y_pred.shape!=y_target.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} != y_target {y_target.shape}")
        
        
    if np.any(y_target==0):
        if not ignore_zeros:
            raise ValueError("MAPE is undefined when y_target contains zeros. Try ignore_zeros=True")
        mask=y_target!=0
        if not np.any(mask):
            raise ValueError("MAPE is undefined when all y_target values are zeros")
        
        y_target,y_pred=y_target[mask],y_pred[mask]
    
    
    return np.mean(np.abs((y_target-y_pred) /y_target))


def poisson_deviance(y_true,y_pred):
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)

    
    if y_true.shape!=y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")


    y_true=y_true.ravel()
    y_pred=np.maximum(y_pred.ravel(),1e-15)
    safe_true=np.where(y_true>0,y_true,1)
    return 2 * np.sum(y_true*np.log(safe_true/y_pred) -(y_true-y_pred))

     