import torch

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        s = X@self.w

        return s
        pass 

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        s = self.score(X)
        y_hat = torch.where(s<=0, 0.0, 1.0)

        # y_hat = (s>0)*(1.0)
        
        return y_hat
        pass 

class MBPerceptron(LinearModel):

    def loss(self, X, y):
        """
        Compute the misclassification rate. A point i is classified correctly if it holds that s_i*y_i_ > 0, where y_i_ is the *modified label* that has values in {-1, 1} (rather than {0, 1}). 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        HINT: In order to use the math formulas in the lecture, you are going to need to construct a modified set of targets and predictions that have entries in {-1, 1} -- otherwise none of the formulas will work right! An easy to to make this conversion is: 
        
        y_ = 2*y - 1
        """
        # convert label from 0, 1 to -1, 1
        y_hat = 2 * self.predict(X) - 1
        missclass = 1.0*(y_hat * y < 0)
        return missclass.mean()

    def grad(self, X, y, k, alpha):
        y_ = (X@self.w*y < 0)*y
        reshaped_y_ = torch.reshape(y_, (k,1))
        return (alpha/k)*(X*reshaped_y_).sum(axis = 0)


class MBPerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y, k, alpha):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """
        loss = self.model.loss(X, y)
        self.model.w += torch.reshape(self.model.grad(X, y, k, alpha),(self.model.w.size()[0],))
        new_loss = self.model.loss(X, y)
        return abs(loss - new_loss)
    
    # 
    # find the sign for x_1, x_2, x_3..., x_n for the update. Using this sign create a matrix so that the column means are the grad updates. 
