import torch.nn as nn

from transfromer_learn.models.embedding.transformer_encoding import TransformerEncoding
from transfromer_learn.models.blocks.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    
    def __init__(self, dec_voc_size, max_length, embedding, ffn_hidden, n_head, n_layers, dropout, device):
        super(Decoder, self).__init__()
        self.emb = TransformerEncoding(vocab_size=dec_voc_size, embedding=embedding, max_length=max_length, dropout=dropout, device=device)
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim=embedding, hidden_dim=ffn_hidden, n_head=n_head, dropout=dropout)
                                     for _ in n_layers])
        self.linear = nn.Linear(embedding, dec_voc_size)
        
    def forward(self, trg, enc_src, trg_mark, src_mask):
        trg = self.emb(trg)
        
        for layer in layer:
            trg = layer(trg, enc_src, src_mask, trg_mark)
        
        out = self.linear(trg)
        return out