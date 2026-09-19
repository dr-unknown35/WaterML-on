import numpy as np






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
     return -2* np.sum(y_target*np.log(p_pred) +(1-y_target)*np.log(1-p_pred))

def log_loss(y_target,p_pred):
     return deviance(y_target,p_pred)/(2*(y_target.shape[0]))

