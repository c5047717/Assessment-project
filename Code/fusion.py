import torch.nn as nn, torch
class FusionBlock(nn.Module):
    def __init__(self,img_dim,txt_dim,out_dim):
        super().__init__(); self.fc=nn.Sequential(nn.Linear(img_dim+txt_dim,out_dim),nn.ReLU(True))
    def forward(self,img_feat,txt_feat): return self.fc(torch.cat([img_feat,txt_feat],dim=-1))
