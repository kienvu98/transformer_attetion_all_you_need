import torch.nn as nn

class FeedForward(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, dropout = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tensor):
        tensor = self.linear1(tensor)
        tensor = self.relu(tensor)
        tensor = self.dropout(tensor)
        tensor = self.linear2(tensor)
        return tensor

