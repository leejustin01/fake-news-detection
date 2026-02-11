import torch.nn as nn

class ShallowNetwork(nn.Module):
    def __init__(self, input_dim, hidden_width, output_dim):
        super().__init__()
        
        self.hidden = nn.Linear(input_dim, hidden_width)
        self.act = nn.ReLU()
        self.output = nn.Linear(hidden_width, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z1 = self.hidden(x)
        a1 = self.act(z1)
        out = self.output(a1)
        prob = self.sigmoid(out)
        return prob

