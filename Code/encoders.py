import torch, torch.nn as nn
class ImageEncoder(nn.Module):
    def __init__(self,in_ch=3,feat_dim=256):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_ch,64,4,2,1),nn.ReLU(True),
            nn.Conv2d(64,128,4,2,1),nn.BatchNorm2d(128),nn.ReLU(True),
            nn.Conv2d(128,256,4,2,1),nn.BatchNorm2d(256),nn.ReLU(True),
            nn.Conv2d(256,512,4,2,1),nn.BatchNorm2d(512),nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        ); self.fc=nn.Linear(512,feat_dim)
    def forward(self,x): return self.fc(self.net(x).flatten(1))
class TextEncoder(nn.Module):
    def __init__(self,vocab_size,emb_dim=192,hidden=256,pad_idx=0):
        super().__init__()
        self.emb=nn.Embedding(vocab_size,emb_dim,padding_idx=pad_idx)
        self.rnn=nn.GRU(emb_dim,hidden,batch_first=True)
    def forward(self,tokens):
        _,h=self.rnn(self.emb(tokens)); return h.squeeze(0)
