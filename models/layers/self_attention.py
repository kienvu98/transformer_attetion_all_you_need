import torch.nn as nn
import math
import torch


class SelfAttention(nn.Module):
    '''
        Class tính sefl attention
    '''
    
    def __init__(self, dropout = 0.1):
        super(SelfAttention, self).__init__() #  gọi constructor của lớp cha 
        self.dropout = nn.Dropout(dropout) # tắt 1 số node lúc train
        
    def forward(self, query, key, value, mask=None):
        # input 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = key.size()
        
        # dot product query * ket^T
        scores = (query @ key.transpose(2,3)) / math.sqrt(d_tensor)
        
        # masking với decode
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e9"))
            
        # tính sofmax
        scores = self.dropout(torch.softmax(scores, dim = -1))
        
        # output = scores * values
        output = scores @ value
        return output
        
        
        
        
        