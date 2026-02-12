import torch.nn as nn

class LogisticRegression(nn.Module):

  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.linear = nn.Linear(input_dim, output_dim)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    out = self.linear(x)
    prob = self.sigmoid(out)
    return prob