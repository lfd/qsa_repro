import torch
import numpy as np

class Scale(torch.nn.Module):
    def __init__(self):
        super(Scale, self).__init__()
        self.factor = torch.nn.Parameter(data=torch.Tensor(np.ones(1)), requires_grad=True)


    def forward(self, inputs):
        return self.factor * inputs