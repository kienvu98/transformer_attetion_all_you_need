import torch.nn as nn
import torch
import math

"""
viết 2 cách
1: dùng thư viện norm sẵn của pytorch
2: tự tùy chỉnh
"""


class LayerNormPytorch(nn.Module):
    
    '''
    dùng thư viện của pytorch
    '''
    
    def __init__(self, embdding_dim):
        super(LayerNormPytorch, self).__init__()
        self.norm = nn.LayerNorm(embdding_dim)
        
    def forward(self, tensor):
        return self.norm(tensor)
    
    
class LayerNorm(nn.Module):
    
    '''
    tự cài đặt
    '''
    
    def __init__(self, embembdding_dimd, eps = 1e-12):
        super().__init__(LayerNorm, self)
        self.gamma = nn.Parameter(torch.ones(embembdding_dimd))
        self.beta = nn.Parameter(torch.zeros(embembdding_dimd))
        self.eps = eps
        
    def forward(self, tensor):
        mean = tensor.mean(-1, keepdim = True) # keepdim = True không làm mất chiều của tensor
        var = tensor.var(-1, unbiased = False, keepdim = True) # unbiased = False (1/N) còn = True(1/N-1)
        
        out = (tensor - mean) / (math.sqrt(var + self.eps)) # boatcating tự kéo về đúng chiều
        out = self.gamma * out + self.beta
        return out
        