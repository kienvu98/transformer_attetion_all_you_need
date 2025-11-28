import torch.nn as nn

from transfromer_learn.models.layers.mutil_head_attention import MultilHeadAttention
from transfromer_learn.models.layers.norm import LayerNorm, LayerNormPytorch
from transfromer_learn.models.layers.feed_forward import FeedForward

class DecoderLayer(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, n_head, dropout = 0.1):
        super(DecoderLayer, self).__init__()
        self.mutil_head_attetion = MultilHeadAttention(embedding_dim=embedding_dim, num_head=n_head, dropout=dropout)
        self.enc_dec_attention = MultilHeadAttention(embedding_dim=embedding_dim, num_head=n_head, dropout=dropout)
        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.norm3 = LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim = embedding_dim, hidden_dim = hidden_dim, dropout = dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    
    def forward(self, tensor_dec, tensor_enc, source_mask, target_mask):
        _tensor = tensor_dec
        
        # mutil head attention
        tensor = self.enc_dec_attention.forward(query=tensor_dec, key=tensor_dec, value=tensor_dec, mask=target_mask)
        
        # add & norm
        tensor = self.dropout1(tensor)
        tensor = self.norm1(_tensor + tensor)
        
        # enncoder and decoder mutil haed attention
        # nếu chỉ sinh (mô hình GPT không có encoder) -> tensor_enc = None
        if tensor_enc is not None:
            _tensor = tensor
             
            # encoder decoder mutil head attetion
            tensor = self.enc_dec_attention.forward(query=tensor, key=tensor_enc, value=tensor_enc, mask=source_mask)
            
            # add & norm
            tensor = self.dropout2(tensor)
            tensor = self.norm2(_tensor + tensor)
        
        _tensor = tensor    
        
        # feed forward
        tensor = self.feed_forward.forward(tensor)
        
        # add $ norm
        tensor = self.dropout3(tensor)
        tensor = self.norm3(_tensor + tensor)
        
        return tensor
        
        
            
            
            
        
        