import torch
import torch.nn as nn
import torch.nn.functional as F

class GeMPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6, learn_p=True):
        super().__init__()
        self.eps = eps
        if learn_p:
            self.p = nn.Parameter(torch.tensor(float(p)))
        else:
            self.register_buffer("p", torch.tensor(float(p)))

    def forward(self, x):
        p = torch.clamp(self.p, min=1e-3, max=6.0)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.pow(1.0 / p)
        return x
