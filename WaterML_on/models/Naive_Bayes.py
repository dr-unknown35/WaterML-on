import numpy as np






class BernoulliNB:
    def __init__(self):
        self.priors=None
        self.probabilities=None



    def fit(self, X,y):
        counts=np.bincount(y)
        #count number of targets from each class
        self.priors=counts/len(y)
        #count fractions of targets from each class
        self.probabilities=np.array([(np.sum(np.where(X[y==0]==1,1,0),axis=0)+1)/(counts[0]+2),(np.sum(np.where(X[y==1]==1,1,0),axis=0)+1)/(counts[1]+2)])
        #find the probability of a datapoint Xi=1 given Y=y 
    
        #  $$\large \phi_{i|y=1} = P(x_i=1|y=1)=\frac{P(x_i=1,y=1)}{P(y=1)}$

    def predict(self,X):
        scores=[]
        for k in range(len(self.priors)):
            #work in log space. monotonic so argmax is unaffected and we don't have to multiply.

            #for a given class k, datapoint Xi has probabilities $P(Xi=1|y=k)=$ self.probabilities[k][i]
            #and $P(Xi=0|y=k)=$ 1-self.probabilities[k][i]

            #our likelihood for Xi becomes  $\large L(X_i)={X_i\times P(Xi=0|y=k) +(1-X_i)\times P(Xi=i|y=k)}$
            #= X* self.probabilities[k] + (1-X) * (1-self.probabilities[k])
            #likelihood for a given class K usually requires to multiply the above quantity for all i with the priors , but since we're working in Log space
            
            #log likelyhood for class K =$\large{\phi_k +\sum_{i=0}^{n} L(X_i)}$
            


            scores.append(np.log(self.priors[k])+np.sum(np.log(X*self.probabilities[k]+ (1-X)*(1-self.probabilities[k])),axis=1))


        return np.argmax(np.array(scores),axis=0)
        #pick the highest likelyhood