import torch
class LinearModel:

    def __init__(self):
        self.w = None 


    def score(self, X):
        if self.w is None: 
            self.w = (torch.rand((X.size()[1])) - 0.5) / X.size()[1]
            
        return X@self.w.double()
        pass 

    def predict(self, X):

        s = self.score(X)
        y_hat = 1.0*(s >= 0)
        
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

class AdamOptimizer:

    def __init__(self, model):
        self.model = model
        self.prev_m = None
        self.prev_v = None
        self.t = 0
        
    def step(self, X, y, alpha, beta1, beta2):
        self.t += 1

        # compute gradient
        g = self.model.grad(X, y)

        # let m and v equal to zero if they are None
        if self.prev_m is None:
            self.prev_m = torch.zeros_like(g)
            self.prev_v = torch.zeros_like(g)
        
        # update m and v
        self.prev_m = beta1 * self.prev_m + (1 - beta1) * g
        self.prev_v = beta2 * self.prev_v + (1 - beta2) * (g ** 2)

        # compute m_hat and v_hat
        m_hat = self.prev_m / (1 - (beta1 ** self.t))
        v_hat = self.prev_v / (1 - (beta2 ** self.t))

        # update weight
        self.model.w -= alpha * m_hat / (torch.sqrt(v_hat) + 1e-8)

        # compute loss
        loss = self.model.loss(X, y)

        return loss