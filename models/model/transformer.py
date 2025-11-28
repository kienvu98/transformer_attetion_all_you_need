import torch
import torch.nn as nn

from transfromer_learn.models.model.decoder import Decoder
from transfromer_learn.models.model.encoder import Encoder


class Transformer(nn.Module):
    
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, embedding,
                 n_head, max_length, ffn_hidden, n_layers, dropout, device):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(enc_voc_size=enc_voc_size, max_length=max_length, embedding=embedding,
                               ffn_hidden=ffn_hidden, n_head=n_head, n_layers=n_layers, dropout=dropout, device=device)
        
        self.decoder = Decoder(dec_voc_size=dec_voc_size, max_length=max_length, embedding=embedding,
                               ffn_hidden=ffn_hidden, n_head=n_head, n_layers=n_layers, dropout=dropout, device=device)
        
        
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.encoder(src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        pass
    
    
    
    def make_src_mask(self, src):
        '''
        ham mask tao padding mask cho encoder
        '''
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # tao ra mask co size [batch_size, 1, 1, src_len]
        return src_mask
    

    def make_trg_mask(self, trg):
        '''
        ham tao mask cho decoder
        '''
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)  # -> [batch_size, 1, trg_len, 1] tao padding mask
        
        # tao sub mask tam giacs duoi che tu dang truoc --> du doan tu dang truoc
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.bool).to(self.device) 
        
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask