import torch.nn as nn

from transfromer_learn.models.embedding.transformer_encoding import TransformerEncoding
from transfromer_learn.models.blocks.encoder_layer import EncoderLayer

class Encoder(nn.Module):
    
    def __init__(self, enc_voc_size, max_length, embedding, ffn_hidden, n_head, n_layers, dropout, device):
        super(Encoder, self).__init__()
        self.emb = TransformerEncoding(vocab_size=enc_voc_size, embedding=embedding, max_length=max_length, dropout=dropout, device=device)
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim=embedding, n_head=n_head, hidden_dim=ffn_hidden, dropout=dropout)
                                    for _ in range(n_layers)])
        
    
    def forward(self, x, src_mask):
        
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
            
        return x
    
        
    