import torch


class LinearModel:

    def __init__(self):
        self.w = None 
        self.prev_w = None

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

class LogisticRegression(LinearModel):
    def loss(self, X, y):
        # calling the score function
        s = self.score(X)

        # sigmoid function of the score
        sigma_s = torch.sigmoid(s)

        # calculate loss using logistic regresison function
        loss = torch.mean(-(y * torch.log(sigma_s) + (1 - y) * torch.log(1 - sigma_s)))
        return loss
    
    def grad(self, X, y):
        # calling the score function
        s = self.score(X)

        # the gradient function  p.s. v[:, None] converts a tensor v with size (n,) to a tensor with shape (n, 1)
        v = torch.sigmoid(s) - y
        return torch.mean(v[:, None] * X, dim = 0)

class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model 
        
    
    def step(self, X, y, alpha, beta):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """

        # call loss function
        loss = self.model.loss(X, y)
        
        # record the current w
        now_w = self.model.w.clone()
        
        # in case it is the first update
        if self.model.prev_w == None:
            self.model.w += -alpha * self.model.grad(X, y)
        else:
            self.model.w += -alpha*self.model.grad(X, y) + beta*(now_w - self.model.prev_w) # Spicy Gradeint Descent Function
        # the current w becomes the past w
        self.model.prev_w = now_w
        
        return loss
    



