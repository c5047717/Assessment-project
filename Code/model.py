import os, torch
import torch.nn as nn
from .encoders import ImageEncoder, TextEncoder
from .fusion import FusionBlock
from .seq import SeqModel
from .decoders import CaptionDecoder, ImageDecoder
from .attention import CrossModalAttention
from .autoencoder import ConvAutoencoder
class MMSequenceModel(nn.Module):
    def __init__(self,cfg,vocab_size):
        super().__init__(); self.cfg=cfg; pad=cfg['data']['pad_idx']
        self.img_enc=ImageEncoder(in_ch=cfg['data']['channels'],feat_dim=cfg['model']['img_feat_dim'])
        self.txt_enc=TextEncoder(vocab_size=vocab_size,emb_dim=cfg['model']['embed_dim'],hidden=cfg['model']['text_hidden'],pad_idx=pad)
        self.fuse=FusionBlock(cfg['model']['img_feat_dim'],cfg['model']['text_hidden'],cfg['model']['fusion_dim'])
        self.seq=SeqModel(in_dim=cfg['model']['fusion_dim'],hidden=cfg['model']['seq_hidden'])
        self.xattn=CrossModalAttention(dim_q=cfg['model']['seq_hidden'],dim_kv=cfg['model']['img_feat_dim'],attn_dim=cfg['model']['attn_dim'],out_dim=cfg['model']['seq_hidden'])
        self.cap_dec=CaptionDecoder(vocab_size=vocab_size,emb_dim=cfg['model']['embed_dim'],hidden=cfg['model']['seq_hidden'],pad_idx=pad)
        self.img_dec=ImageDecoder(latent_dim=cfg['model']['seq_hidden'],out_ch=cfg['data']['channels'],channels=cfg['model']['image_decoder_channels'])
        self.use_ae=bool(cfg.get('ae',{}).get('use',False))
        if self.use_ae:
            self.ae=ConvAutoencoder(in_ch=cfg['data']['channels'],latent_dim=cfg['model']['seq_hidden'])
            ckpt=cfg.get('ae',{}).get('ckpt',None)
            if ckpt and os.path.exists(ckpt):
                try:
                    state=torch.load(ckpt,map_location='cpu'); self.ae.load_state_dict(state['model']); print(f"[MM] Loaded AE from {ckpt}")
                except Exception as e: print(f"[MM] AE load failed: {e}")
            if cfg.get('ae',{}).get('freeze_decoder',True):
                for p in self.ae.parameters(): p.requires_grad=False
                self.ae.eval()
        H=cfg['model']['seq_hidden']
        self.to_latent=nn.Sequential(nn.Linear(H,H),nn.ReLU(True),nn.Linear(H,H),nn.Tanh())
    def forward(self,imgs,caps,tgt_cap):
        B,K=imgs.size(0),imgs.size(1)
        img_feats=[self.img_enc(imgs[:,t]) for t in range(K)]
        txt_feats=[self.txt_enc(caps[:,t]) for t in range(K)]
        import torch as T
        img_feats=T.stack(img_feats,1); txt_feats=T.stack(txt_feats,1)
        fused=self.fuse(img_feats,txt_feats); _,seq_h=self.seq(fused)
        kv=T.stack([img_feats[:,-1],txt_feats[:,-1]],1); ctx=self.xattn(seq_h,kv)
        cap_logits=self.cap_dec(tgt_cap,init_state=ctx)
        if self.use_ae:
            pred_latent=2.0*self.to_latent(seq_h)
            img_pred=self.ae.decode(pred_latent)
            return cap_logits,img_pred,pred_latent
        else:
            img_pred=self.img_dec(ctx); return cap_logits,img_pred,None
