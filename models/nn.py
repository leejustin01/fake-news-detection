import torch.nn as nn

class FeedForwardNetwork(nn.Module):

  def __init__(self, input_dim, layer_widths, output_dim, activation=nn.ReLU):
    super().__init__()
    assert(len(layer_widths) >= 1)
    
    self.act = activation()
    
    layers = []
    prev = input_dim
    for width in layer_widths:
        layers.append(nn.Linear(prev, width))
        layers.append(activation())
        prev = width
    layers.append(nn.Linear(prev, output_dim))
    
    self.hidden_layers = nn.Sequential(*layers)
    self.sigmoid = nn.Sigmoid()


  def forward(self, x):
    out = self.hidden_layers(x)
    prob = self.sigmoid(out)
    return prob