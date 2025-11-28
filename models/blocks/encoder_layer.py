import torch.nn as nn

from transfromer_learn.models.layers.mutil_head_attention import MultilHeadAttention
from transfromer_learn.models.layers.norm import LayerNorm, LayerNormPytorch
from transfromer_learn.models.layers.feed_forward import FeedForward

class EncoderLayer(nn.Module):
    
    def __init__(self, embedding_dim, n_head, hidden_dim, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.mutil_head_attention = MultilHeadAttention(embedding_dim=embedding_dim, num_head=n_head, dropout=dropout)
        self.norm1 = LayerNormPytorch(embedding_dim)
        self.norm2 = LayerNormPytorch(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim = embedding_dim, hidden_dim = hidden_dim, dropout = dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, tensor, mask = None):
        _tensor = tensor
        # tính mutil_haed sefl attetion
        tensor = self.mutil_head_attention.forward(query=tensor, key=tensor, value=tensor, mask=mask)
        
        # add & norm
        tensor = self.dropout1(tensor)
        tensor = self.norm1(_tensor + tensor)
        
        _tensor = tensor
        
        # feed forward
        tensor = self.feed_forward.forward(tensor)
        
        # add & norm
        tensor = self.dropout2(tensor)
        tensor = self.norm2(_tensor + tensor)
        return tensor