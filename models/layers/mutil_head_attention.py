import torch.nn as nn
import torch
from transfromer_learn.models.layers.self_attention import SelfAttention


class MultilHeadAttention(nn.Module):
    
    '''
     Class block multil haed attention
     chia du lieu thanh nhieu khoi block de attention
    '''
    
    def __init__(self, embedding_dim, num_head, dropout = 0.1):
        super(MultilHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_head # so luong head
        self.dim_per_head = embedding_dim // num_head # dim tren tung head
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.self_attention = SelfAttention(dropout)
        
    
    def forward(self, query, key, value, mask = None):
        '''
        hàm mutil-head attention 
        '''
        # apply linner
        query = self.w_q(query)
        key = self.w_v(key)
        value = self.w_v(value)
        
        # spit tensor thanh number number of head
        query = self.split(query)
        key = self.split(key)
        value = self.split(value)
        
        # attention
        out = self.self_attention.forward(query, key, value, mask=mask)
        
        # concat
        out = self.concat(out)
        out = self.out(out)
        return out
    
    
    def split(self, tensor):
        '''
        hàm chia tensor thành nhiều head tensor
        chuyển từ [batch_size, length, embedding] -> [batch_size, lenght, head, d_tensor] -> [batch_size, head, length, d_tensor]
        
        '''
        batch_size = tensor.size(0)
        length = tensor.size(1)
        
        '''
        PyTorch thường yêu cầu shape: (batch_size, head, length, d_tensor) để thực hiện tính attention trên từng "đầu".
        '''
        tensor = tensor.view(batch_size, length, self.num_head, self.dim_per_head).transpose(1,2)
        return tensor
    
    
    def concat(self, tensor):
        '''
        hàm concat sau khi attention
        chuyển [batch_size, head, lenght, d_tensor] -> [batch_size, lenght, head, d_tensor] -> [batch_size, length, embedding] 
        '''
        
        batch_size = tensor.size(0)
        length = tensor.size(1)
        tensor = tensor.transpose(1,2).contiguous().view(batch_size, length, self.embedding_dim)
        return tensor
        