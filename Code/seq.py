import torch.nn as nn, torch
class SeqModel(nn.Module):
    def __init__(self,in_dim,hidden=384):
        super().__init__(); self.rnn=nn.GRU(in_dim,hidden,batch_first=True)
    def forward(self,x): out,h=self.rnn(x); return out,h.squeeze(0)
