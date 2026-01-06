import torch.nn as nn, torch
class ConvAutoencoder(nn.Module):
    def __init__(self,in_ch=3,latent_dim=384):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Conv2d(in_ch,64,4,2,1),nn.ReLU(True),
            nn.Conv2d(64,128,4,2,1),nn.BatchNorm2d(128),nn.ReLU(True),
            nn.Conv2d(128,256,4,2,1),nn.BatchNorm2d(256),nn.ReLU(True),
            nn.Conv2d(256,512,4,2,1),nn.BatchNorm2d(512),nn.ReLU(True),
        )
        self.enc_fc=nn.Linear(512*4*4,latent_dim)
        self.dec_fc=nn.Linear(latent_dim,512*4*4)
        self.dec=nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1),nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1),nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1),nn.ReLU(True),
            nn.ConvTranspose2d(64,in_ch,4,2,1),nn.Sigmoid()
        )
    def encode(self,x): return self.enc_fc(self.enc(x).reshape(x.size(0),-1))
    def decode(self,z): return self.dec(self.dec_fc(z).reshape(z.size(0),512,4,4))
    def forward(self,x): z=self.encode(x); xr=self.decode(z); return z,xr
