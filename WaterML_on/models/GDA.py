import numpy as np



class Linear:
    def __init__(self):
        self.priors=None
        self.means=None
        self.covariances=None

    def fit(self, X, y):
            counts=np.bincount(y)
            self.priors=counts/len(y)
            #count each class' occurence and find the proportion of that class in y

            self.means=[]
            for i in range(len(counts)):
                  self.means.append(np.where(y==i,1,0)@X*(1/counts[i]))
            self.means=np.array(self.means)
            #identity function over y=i , dot with X to sum rows of X having class i . divide by number of occurences of class i to obtain class mean.


            X_centered=X-self.means[y]
            #get the mean corresponding to each row of X and subtract it to obtain centered X.
            self.covariance=(X_centered.T@X_centered) /len(y)
            #multiply mean centered data matrix and it's transpose to calculate the product between the value for feature i and j . divide by number of rows to obtain covariance , or variance in case of diagonal entries.
            #The covariance matrix Σ for a random vector $$\large(X = [X_1, X_2, \dots, X_n]^T)\quad $$ is defined as: 

            # $$\large \Sigma = \operatorname{Cov}(X) = \mathbb{E}\left[(X - \boldsymbol{\mu})(X - \boldsymbol{\mu})^T\right]  \quad \quad \quad \quad \quad \quad \quad\quad\quad$$
      


    def predict(self, X):
            #$$\large f(x; \mu, \Sigma) = \frac{1}{(2\pi)^{k/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)$$


            scale_factor=1/(((2*np.pi)**((X.shape[1])/2))* np.sqrt(np.linalg.det(self.covariance)))
            inv=np.linalg.inv(self.covariance)
            #scale factor for multinomial gaussian distribution , to normalize the probabilities into [0,1]
            conditionals=[]
            for k in range(len(self.priors)):
                  diff=X-self.means[k]
                  #calculate mean centered data points
                  m=diff@inv
                  #diff with shape (N,P), inverse cov matrix with shape (P,P). matrix multiply to standardize features and rotate data to reduce correlation. shape(N,P)
                  conditionals.append(self.priors[k]*scale_factor * np.exp((-1/2) * np.sum(m * diff, axis=1)))
                  #the term inside the exponential is called the squared Mahalanobis distance $$D_M^2 = (x - \mu)^T \Sigma^{-1} (x - \mu)$$
                  #which measures how far a point is from the mean of a multivariate distribution


                  #multiply each feature for each datapoint with it's covariance , sum on rows to obtain the squared distance from the mean for each datapoint.
                  #this distance is passed through a gaussian bell curve so that datapoints that are farther from the mean have lower probability.
                  #multiply the prior , scale factor and conditional to obtain the posterior probability that a given datapoint belongs to a certain class. shape(D,N)


            return np.argmax(np.array(conditionals),axis=0)
            #pick which of the D classes has the highest probability for each of the N rows.


class Quadratic:
    def __init__(self):
        self.priors=None
        self.means=None
        self.covariances=None

    def fit(self, X, y):
            counts=np.bincount(y)
            self.priors=counts/len(y)


            self.means=[]
            self.covariances=[]


            for i in range(len(counts)):
                  self.means.append(np.where(y==i,1,0)@X*(1/counts[i])) 


            for i in range(len(counts)):
                  X_k=X[y==i]
                  X_centered=X_k-self.means[i]
                  self.covariances.append((X_centered.T@X_centered) /counts[i])


            self.means,self.covariances=np.array(self.means),np.array(self.covariances)

    def predict(self, X):
            conditionals=[]
            for k in range(len(self.priors)):
                  scale_factor=1/(((2*np.pi)**((X.shape[1])/2))* np.sqrt(np.linalg.det(self.covariances[k])))
                  inv=np.linalg.inv(self.covariances[k])
                  diff=X-self.means[k]
                  
                  m=diff@inv

                  conditionals.append(self.priors[k]*scale_factor * np.exp((-1/2) * np.sum(m * diff, axis=1)))
                  

            return np.argmax(np.array(conditionals),axis=0)
            #pick which of the D classes has the highest probability for each of the N rows.
