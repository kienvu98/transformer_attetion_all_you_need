import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    
    '''
    sinusoi tinh toan vi tri cua tu trong cau
    '''
    
    def __init__(self, embedding, max_length, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_length, embedding, device=device)
        self.encoding.requires_grad = False # khong dao ham hoc vi day tinh toan thu cong qua sin, cos
        
        pos = torch.arange(0, max_length, device=device) # vector positon tung token
        pos = pos.float().unsqueeze(dim=1) # them 1 chieu 
        
        _2i = torch.arange(0, embedding, step=2, device=device).float()
        
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embedding)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embedding)))
        
        
    def forward(self, x):
        
        batch_size, seq_len = x.size()
        
        return self.encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
    
    