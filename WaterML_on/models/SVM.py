import numpy as np
from ..optim.gradient_methods import GradientDescent
from .base import baseSVM





#in Primal Sub-Gradient-Descent SVM , we assume the binary classified data is not perfectly separable by a linear boundary but that it would still be enough accounting for errors.
#the goal is to find the best boundary such that the boundary and the closest point from each class have as much "breathing room" as possible


#a hyperplane is defined as $$W^Tx+b=0$$ where $w$ is the normal vector perpendicular to the hyperplane which determines it's orinetation
# $b$is the offset that shifts the hyperplane from the origin
# for a datapoint $x_i$, the sign of $W^Tx_i+b$ tells us which side the datapoint is on. this is our classification mechanism
# predict $y=1$if $W^Tx_i+b>0$else $y=-1$ (SVMs use 1/-1 for binary classes as opposed to 0/1) 


#for a datapoint $x_i$the functional margin is $\Large {\hat y_{i} =y_{i} \cdot (W^Tx_i+b) \enspace}$

#if the classifier predicts correctly , ie $y_i$and $W^Tx+b$ have the same sign, then their product that is the functional margin will be positive.
# the bigger that quantity is , the more confident the prediction is.
#however , this quantity can be artifically inflated by scaling w and b. which does nothing for the hyperplane but still gives us "better" predictions.

#to mitigate this problem , the geometric margin divides out the scale of $w$
#$\quad\Large {\gamma_i = y_{i} \cdot (W^Tx_i+b) \frac{1}{\|{w}\|} \enspace}$ , the Euclidean distance from a datapoint $x_i$to the hyperplane


# $\Large\gamma $ is the minimum geometric margin over all data points across all classes. meaning the closest point to the hyperplane
# by symmetry of the optimal hyperplane , the closest points end up having the same distance from the hyperplane.

#this is where the optimization problem reveals itself , to get the best boundary , we need the hyperplane that maximizes the minimum distance from $\Large\gamma$
#ie as much breathing room as possible between the boundary and the nearest point of either class


class PrimalSubGD(baseSVM):
    def __init__(self,lr,batch_size,epochs,decay,C,solver):
        self.lr=lr
        self.batch_size=batch_size
        self.epochs=epochs
        self.decay=decay
        self.C=C
        self.solver=solver
        self.weights=None
        self.bias=None
        
        
    def fit(self,X,y):
        solver=GradientDescent(self.epochs,self.batch_size,learning_rate=self.lr,penalty='L2',alpha=1/(self.C*X.shape[0]),decay=self.decay) if self.solver is None else self.solver
        self.bias,self.weights=solver.optimize(self,X,np.where(y>0,1,-1))
        return self
    
    
    
    def get_gradient(self,X,y,weights,bias):
        #because maximal marginal classifiers have low bias , they will inevitablely have high variance 
        #if a point is misclassified is will single handedly shift the entire hyperplane away from it's "correct" class and will cause it to misclassify future points.
        #which is why we use a soft margin classifier :we let points violate the margin and pay a penalty. A point’s cost is determined by it's 'correctness score' , the functional margin.
        # loss_i = $\Large max(0, 1 − y_i(w^T x_i + b))\qquad\qquad$
        
        #if the functional margin is $>1$then the point is on the correct side of the decision boundary and outside the safety margin. not penalized
        #if the functional margin is $=1$the point lies exactly on the margin support boundary. not penalized 
        #if the functional margin is between $0 ≤ Value < 1$then the point is correctly classified but is too close to the decision boundary (inside the margin).
        #if the functional margin is $<0$then the point falls on the wrong side of the decision boundary entirely. heavily penalized
        
        
        violating_margin= y *(X@weights +bias) <1 #get points that violate the margin 
        #the functional margin isn't exaclty differentiable for all its values since it has a sharp corner for $m<1$$l=1-m$ and $m>=1$$l=0$  with $\large m = y_i(w^T x_i + b)\enspace$
        #so we use a subgradient for our weights and bias gradients , for $w$ the gradient is  $−y_i x_i$ for $m<1$ and $0$ otherwise
        #for $b$ the gradient is $-y_i$ for $m<1$ and $0$ otherwise 
        
        return (-np.sum(y[violating_margin]) / X.shape[0],-(X[violating_margin].T @ y[violating_margin]) / X.shape[0])
    

    def predict(self,X):
        return np.where(X@self.weights +self.bias >=0, 1,0)



class MulticlassOvR(baseSVM):
    
    # for $K$ classes, train $K$ SVMs using PrimalSGD where one class gets the positive label and the rest get the negative one. 
    
    def __init__(self, lr, batch_size, epochs, decay, C, solver):
        self.lr=lr
        self.batch_size=batch_size
        self.epochs=epochs
        self.decay=decay
        self.C=C
        self.solver=solver
        self.machines=None
        self.classes=None
    
    def fit(self,X,y):
        self.machines=[]
        self.classes=np.unique(y)
        for c in self.classes:
            y_c=np.where(y==c ,1,-1)
            machine=PrimalSubGD(self.lr,self.batch_size,self.epochs,self.decay,self.C,self.solver)
            self.machines.append(machine.fit(X,y_c))
        return self
        
    def predict_raw(self, X):
        return np.array([machine.predict_raw(X) for machine in self.machines]).T
    
    
    # for each trained SVM , you get your raw predictions and find the most likely class (highest raw prediction)
    def predict(self,X):
        predictions=self.predict_raw(X)
        return self.classes[np.argmax(predictions,axis=1)]
            
    
        
        