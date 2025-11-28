import torch.nn as nn

from transfromer_learn.models.embedding.token_embedding import TokenEmbedding
from transfromer_learn.models.embedding.positional_encoding import PositionalEncoding

class TransformerEncoding(nn.Module):
    
    def __init__(self, vocab_size, embedding, max_length, dropout, device):
        super(TransformerEncoding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size=vocab_size, embedding=embedding)
        self.pos_emb = PositionalEncoding(embedding=embedding, max_length=max_length, device=device)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.dropout(tok_emb + pos_emb)