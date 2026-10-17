import numpy as np






class BernoulliEM:
    def __init__(self):
        self.priors=None
        self.probabilities=None



    def fit(self, X,y):
        counts=np.bincount(y)
        #count number of targets from each class
        self.priors=counts/len(y)
        #count fractions of targets from each class
        self.probabilities=np.array([(np.sum(np.where(X[y==0]==1,1,0),axis=0)+1)/(counts[0]+len(counts)),(np.sum(np.where(X[y==1]==1,1,0),axis=0)+1)/(counts[1]+len(counts))])
        #find the probability of a datapoint Xi=1 for either class
    
        #  $\large \phi_{i|y=1} = P(x_i=1|y=1)=\frac{P(x_i=1,y=1)}{P(y=1)}$

    def predict(self,X):
        scores=[]
        for k in range(len(self.priors)):
            #work in log space. monotonic so argmax is unaffected and we don't have to multiply.

            #for a given class k, datapoint Xi has probabilities $P(Xi=1|y=k)=$ self.probabilities[k][i]
            #and $P(Xi=0|y=k)=$ 1-self.probabilities[k][i]

            #likelihood for Xi becomes  $\large L(X_i)={X_i\times P(Xi=0|y=k) +(1-X_i)\times P(Xi=i|y=k)}$
            #= X* self.probabilities[k] + (1-X) * (1-self.probabilities[k])
            #likelihood for a given class K usually requires us to multiply the above quantity for all i with the priors , but since we're working in Log space
            
            #log likelyhood for class K =$\large{\phi_k +\sum_{i=0}^{n} L(X_i)}$
            


            scores.append(np.log(self.priors[k])+np.sum(np.log(X*self.probabilities[k]+ (1-X)*(1-self.probabilities[k])),axis=1))


        return np.argmax(np.array(scores),axis=0)
        #pick the highest likelyhood







class MultinomialEM:
    def __init__(self):
        self.priors=None
        self.probabilities=[]


    def fit(self,X,y):
        counts=np.bincount(y)
        #count number of targets from each class
        self.priors=counts/len(y)
        #count fractions of targets from each class



        #X is a (n,|V|) matrix , where each row represents a "document" and each column represents a word 
        #the sum of a column represents how many times it appeared across all documents.
        #the operation below counts the number of times a word k appears for each class y=c, then divides by the total number of words across all documents of class y.
        for c in range(len(self.priors)):
            self.probabilities.append((np.sum(X[y==c],axis=0)+1)/(np.sum(X[y==c],axis=None)+X.shape[1]))
        self.probabilities=np.array(self.probabilities)

    def predict(self,X):
        #for every word in the vocabulary, how many times did it appear in document i, times the log-probability of that word under class c, summed up.
        #a dot product between our X (matrix (n,|V|) where each row is the number of times words appeared in a given document
        #and (the log of) the probability matrix (k,|V|) where each row is the probabilities of words appearing in a given class
        #we want an output shape of (n,k) , each document's score for each class (each word's number of occurences gets multiplied by its probability , then summed) 
        #so X@self.probabilities.T
        scores=np.log(self.priors)+(X @ np.log(self.probabilities.T))
        return np.argmax(scores,axis=1)