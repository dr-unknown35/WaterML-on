import numpy as np
from ..data_utils.preprocessor import one_hot





def confusion_matrix(y_target,y_pred):
    return (np.c_[(1-y_target),y_target].T @ np.c_[(1-y_pred),y_pred]).flatten()


def accuracy(cm):
     return (cm[0]+cm[3])/np.sum(cm)


def precision(cm):
     return (cm[3])/(cm[1]+cm[3])


def recall(cm):
    return (cm[3])/(cm[2]+cm[3])

                
def specificity(cm):
     return (cm[0])/(cm[0]+cm[1])


def f1_score(cm):
     return (2*cm[3])/(2*cm[3] +cm[1]+cm[2])


def deviance(y_target,p_pred):
     p_pred=np.maximum(p_pred,1e-15)
     return -2* np.sum(y_target*np.log(p_pred) +np.maximum((1-y_target),1e-15)*np.log(np.maximum((1-p_pred),1e-15)))






def log_loss(y_target,p_pred):
     p_pred=p_pred.ravel() if np.ndim(p_pred)==2 and p_pred.shape[1]==1 else p_pred
     p_pred=np.column_stack([1-p_pred,p_pred]) if np.ndim(p_pred)==1 else p_pred
     p_pred=np.clip(p_pred, 1e-15, 1)
     one_hot_y=one_hot(y_target.astype(int),p_pred.shape[1])
     return -1/(y_target.shape[0]) *np.sum(one_hot_y*np.log(p_pred))

