import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    
    '''
    token embedding using torch.nn
    bieu dietu thong qua ma tran trong so duoc hoc
    '''
    
    def __init__(self, vocab_size, embedding, pad_id=1):
        '''
        vocab_size kich thuoc cua vocabulary
        embedding kich thuoc cua embedding dau ra cua tu
        '''
        super(TokenEmbedding, self).__init__(vocab_size, embedding_dim=embedding, padding_idx=pad_id)