import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3., inplace=True) / 6.
