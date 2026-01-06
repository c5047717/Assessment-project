import torch.nn as nn, torch
class CaptionDecoder(nn.Module):
    def __init__(self,vocab_size,emb_dim=192,hidden=384,pad_idx=0):
        super().__init__(); self.emb=nn.Embedding(vocab_size,emb_dim,padding_idx=pad_idx); self.rnn=nn.GRU(emb_dim,hidden,batch_first=True); self.fc=nn.Linear(hidden,vocab_size)
    def forward(self,tgt_tokens,init_state=None):
        h0=init_state.unsqueeze(0) if init_state is not None else None
        out,_=self.rnn(self.emb(tgt_tokens),h0); return self.fc(out)
class ImageDecoder(nn.Module):
    def __init__(self,latent_dim=384,out_ch=3,channels=[256,128,64,32]):
        super().__init__(); C=channels; self.fc=nn.Linear(latent_dim,4*4*C[0])
        self.up=nn.Sequential(
            nn.ConvTranspose2d(C[0],C[1],4,2,1),nn.ReLU(True),
            nn.ConvTranspose2d(C[1],C[2],4,2,1),nn.ReLU(True),
            nn.ConvTranspose2d(C[2],C[3],4,2,1),nn.ReLU(True),
            nn.ConvTranspose2d(C[3],out_ch,4,2,1),nn.Sigmoid()
        )
    def forward(self,z): return self.up(self.fc(z).reshape(z.size(0),-1,4,4))
