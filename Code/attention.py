import torch.nn as nn, torch
class CrossModalAttention(nn.Module):
    def __init__(self,dim_q,dim_kv,attn_dim=192,out_dim=384):
        super().__init__()
        self.wq=nn.Linear(dim_q,attn_dim); self.wk=nn.Linear(dim_kv,attn_dim); self.wv=nn.Linear(dim_kv,attn_dim)
        self.proj=nn.Linear(attn_dim,out_dim)
    def forward(self,q,kv):
        Q=self.wq(q).unsqueeze(1); K=self.wk(kv); V=self.wv(kv)
        attn=(Q*K).sum(-1).softmax(dim=-1); ctx=(attn.unsqueeze(-1)*V).sum(1)
        return self.proj(ctx)
