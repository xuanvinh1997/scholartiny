"""Mamba-2 adapter and an independent, slow recurrent reference.

Reference scope: ngroups=1, d_ssm=expand*d_model, D_has_hdim=False,
RMSNorm enabled, norm_before_gate=False, dt_limit=(0, infinity).
State layout matches upstream Mamba2 for this subset. It is NOT the parallel
SSD kernel; it must not be used for production throughput comparisons.
Sources: docs/SOURCES.md. No upstream CUDA/Triton implementation is vendored.
"""
import math
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .layers import RMSNorm

class Mamba2Reference(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_inner = cfg.hidden_size * cfg.ssm_expand
        self.headdim, self.d_state = cfg.ssm_head_dim, cfg.state_size
        self.nheads = self.d_inner // self.headdim
        self.d_conv = cfg.conv_kernel
        self.conv_dim = self.d_inner + 2*self.d_state
        self.in_proj = nn.Linear(cfg.hidden_size, 2*self.d_inner+2*self.d_state+self.nheads,
                                 bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim,self.conv_dim,self.d_conv,
                               groups=self.conv_dim,bias=cfg.conv_bias)
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        self.A_log = nn.Parameter(torch.empty(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        for p in (self.dt_bias,self.A_log,self.D):
            p._no_weight_decay = True
        self.norm = RMSNorm(self.d_inner, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner,cfg.hidden_size,bias=False)
        with torch.no_grad():
            dt = torch.exp(torch.empty(self.nheads).uniform_(math.log(.001),math.log(.1)))
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            self.A_log.copy_(torch.empty(self.nheads).uniform_(1,16).log())

    def forward(self, u, inference_params=None):
        b,t,_ = u.shape
        z,xbc,dt = torch.split(self.in_proj(u),
                              [self.d_inner,self.conv_dim,self.nheads],dim=-1)
        if inference_params is None or self.layer_idx not in inference_params.key_value_memory_dict:
            hist = u.new_zeros(b,self.conv_dim,self.d_conv)
            state = torch.zeros(b,self.nheads,self.headdim,self.d_state,
                                device=u.device,dtype=torch.float32)
        else:
            hist,state = inference_params.key_value_memory_dict[self.layer_idx]
        A = -self.A_log.float().exp()
        outputs = []
        for i in range(t):
            hist = torch.cat((hist[:,:,1:],xbc[:,i,:,None]),dim=-1)
            mixed = (hist*self.conv1d.weight[:,0,:]).sum(-1)
            if self.conv1d.bias is not None:
                mixed = mixed + self.conv1d.bias
            x,B,C = torch.split(F.silu(mixed),[self.d_inner,self.d_state,self.d_state],dim=-1)
            x = x.view(b,self.nheads,self.headdim).float()
            delta = F.softplus(dt[:,i].float()+self.dt_bias.float())
            state = (state*torch.exp(delta*A)[...,None,None]
                     + delta[...,None,None]*x[...,None]*B.float()[:,None,None,:])
            y = (state*C.float()[:,None,None,:]).sum(-1) + self.D.float()[None,:,None]*x
            outputs.append(y.flatten(1))
        y = torch.stack(outputs,dim=1)
        # Upstream norm_before_gate=False: gate BEFORE RMS normalization.
        y = self.norm(y*F.silu(z.float())).to(u.dtype)
        if inference_params is not None:
            inference_params.key_value_memory_dict[self.layer_idx] = (hist,state)
        return self.out_proj(y)

def make_ssm(cfg, layer_idx):
    if cfg.ssm_backend == 'reference':
        return Mamba2Reference(cfg,layer_idx)
    try:
        from mamba_ssm import Mamba2
    except (ImportError, OSError) as exc:
        raise RuntimeError('Mamba-2 CUDA backend unavailable. Install mamba-ssm and '
                           'causal-conv1d; select reference only for CPU/debug tests.') from exc
    return Mamba2(d_model=cfg.hidden_size,d_state=cfg.state_size,d_conv=cfg.conv_kernel,
                  expand=cfg.ssm_expand,headdim=cfg.ssm_head_dim,ngroups=1,
                  bias=False,conv_bias=cfg.conv_bias,rmsnorm=True,norm_before_gate=False,
                  chunk_size=cfg.ssm_chunk_size,layer_idx=layer_idx)

def new_ssm_cache(max_seq_len, batch_size):
    # Only the fields consumed by the supported upstream Mamba2 interface.
    return SimpleNamespace(max_seqlen=max_seq_len,max_batch_size=batch_size,
                           seqlen_offset=0,batch_size_offset=0,
                           key_value_memory_dict={},lengths_per_sample=None)
