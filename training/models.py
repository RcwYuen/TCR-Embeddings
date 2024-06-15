import torch
from sparsemax import Sparsemax

class ReducedClassifier(torch.nn.Module):
    def __init__(self):
        super(ReducedClassifier, self).__init__()
        self.last_scores = None
        self.last_weights = None

        self.attention = torch.nn.Linear(5, 1)
        self.classifying = torch.nn.Linear(5, 1)
        self.sparsemax = Sparsemax(dim = 0)
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
    
    def forward(self, x):
        self.last_scores = self.attention(x)
        self.last_weights = self.sparsemax(self.last_scores.T).T
        aggregated = torch.sum(self.last_weights * x, dim = 0, keepdim = True)
        return self.sig(self.classifying(aggregated))
    
    
class NormalClassifier(torch.nn.Module):
    def __init__(self, in_dim):
        super(NormalClassifier, self).__init__()
        self.last_scores = None
        self.last_weights = None

        self.attention = torch.nn.Linear(in_dim, 1)
        self.classifying = torch.nn.Linear(in_dim, 1)
        self.sparsemax = Sparsemax(dim = 0)
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
    
    def forward(self, x):
        self.last_scores = self.attention(x)
        self.last_weights = self.sparsemax(self.last_scores.T).T
        aggregated = torch.sum(self.last_weights * x, dim = 0, keepdim = True)
        return self.sig(self.classifying(aggregated))