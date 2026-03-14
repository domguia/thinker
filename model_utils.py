
import copy
import torch
from torch import nn
from torch import Tensor
from typing import Any, Mapping, Optional
from torch.nn import TransformerDecoder, TransformerDecoderLayer, LayerNorm
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hid: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hid, bias=False)
        self.w2 = nn.Linear(d_model, d_hid, bias=False)
        self.w3 = nn.Linear(d_hid, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

def apply_rotary_emb(q, k, cos, sin):
    # q, k: [B, num_heads, seq_len, head_dim]
    # cos, sin: [B, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) # [seq_len, dim]
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :] # [1, 1, seq_len, dim]

class CustomFlexDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_hid, dropout=0.1, skip_self_attn=False, ff_in_self_attn=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.skip_self_attn = skip_self_attn
        self.ff_in_self_attn = ff_in_self_attn
        
        # Self-attention
        if not skip_self_attn:
            self.sa_q_proj = nn.Linear(d_model, d_model, bias=False)
            self.sa_k_proj = nn.Linear(d_model, d_model, bias=False)
            self.sa_v_proj = nn.Linear(d_model, d_model, bias=False)
            self.sa_o_proj = nn.Linear(d_model, d_model, bias=False)
            self.norm1 = RMSNorm(d_model)
            if ff_in_self_attn:
                self.ff_sa = SwiGLU(d_model, d_hid)
                self.norm4 = RMSNorm(d_model)
                
        # Cross-attention
        self.mha_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.mha_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.mha_v_proj = nn.Linear(d_model, d_model, bias=False)
        self.mha_o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        
        # Feed-forward
        self.ff = SwiGLU(d_model, d_hid)
        self.norm3 = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, rope_cos=None, rope_sin=None, **kwargs):
        x = tgt
        B, T, D = x.shape
        
        if not self.skip_self_attn:
            res = x
            x = self.norm1(x)
            q = self.sa_q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            k = self.sa_k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            v = self.sa_v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            
            if rope_cos is not None and rope_sin is not None:
                q, k = apply_rotary_emb(q, k, rope_cos[:,:,:T,:], rope_sin[:,:,:T,:])
            
            # tgt_mask adjustment for SDPA
            sa_mask = None
            if tgt_mask is not None:
                sa_mask = tgt_mask.unsqueeze(1) if tgt_mask.dim() == 3 else tgt_mask.unsqueeze(0).unsqueeze(0)
                sa_mask = sa_mask == 1
                
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=sa_mask)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
            x = res + self.sa_o_proj(attn_out)
            
            if self.ff_in_self_attn:
                x = x + self.ff_sa(self.norm4(x))
                
        # Cross Attention
        res = x
        x = self.norm2(x)
        S = memory.shape[1]
        
        q = self.mha_q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.mha_k_proj(memory).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.mha_v_proj(memory).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        
        ca_mask = None
        if memory_mask is not None:
            ca_mask = memory_mask.unsqueeze(1) if memory_mask.dim() == 3 else memory_mask.unsqueeze(0).unsqueeze(0)
            ca_mask = ca_mask == 1
            
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=ca_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = res + self.mha_o_proj(attn_out)
        
        # Feed forward
        x = x + self.ff(self.norm3(x))
        return x

class CustomFlexDecoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = RMSNorm(layer.d_model)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, rope_cos=None, rope_sin=None, **kwargs):
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, rope_cos=rope_cos, rope_sin=rope_sin)
        return self.norm(x)


class FlexTransformerDecoderLayer(TransformerDecoderLayer): # Not the real flex model! the real one is really flexible!!!
    r"""
        Look at original implementation below
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer
    """

    def __init__(self, *args, **kwargs):
        self.skip_self_attn = kwargs.pop('skip_self_attn', False)
        self.ff_in_self_attn = kwargs.pop('ff_in_self_attn', False)
        assert not (self.skip_self_attn and self.ff_in_self_attn), "skip_self_attn and ff_in_self_attn cannot be True at the same time"

        super().__init__(*args, **kwargs)

        if self.skip_self_attn:
            # remove parameters that we don't need
            self.skip_self_attn = True
            self.self_attn = nn.Identity()
            self.norm1 = nn.Identity()
            self.dropout1 = nn.Identity()
            self.self_attn.batch_first = True

        if self.ff_in_self_attn:
            self.norm4 = copy.deepcopy(self.norm1)
        self.norm5 = copy.deepcopy(self.norm1)
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        skip_self_attn: bool = False,
    ) -> Tensor:
        r"""
            Look at original implementation below
            https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer.forward
        """
        x = tgt
        if self.skip_self_attn:
            skip_self_attn = True
        if self.norm_first:
            if not skip_self_attn: # disable self attention block
                x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
                if self.ff_in_self_attn:
                    x = x + self._ff_block(self.norm4(x))
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else: # default case we are runing in this project
            if not skip_self_attn: # disable self attention block
                x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)) # latent -> _sa_block: where I am? -> what should I do
                if self.ff_in_self_attn:
                    x = self.norm4(x + self._ff_block(x)) # -> _ff_block: what should I look for? ->
            x = self.norm2(x + (x2 := self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))) # _mha_block: info lookup
            x = self.norm3(x + (x3 := self._ff_block(x))) # _ff_block: what I'm doing with what I have found?

        # if self.ff_in_self_attn:
        #     return x, self.norm5(x2)
        # else:
        #     return x, self.norm5(x3)
        return x

class FlexTransformerDecoder(TransformerDecoder):
    r"""
        Look at original implementation below
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoder
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False, skip_self_attn: bool = False) -> Tensor:
        r"""
            Look at original implementation below
            https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoder.forward
        """
        output = tgt

        seq_len = nn.modules.transformer._get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = nn.modules.transformer._detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal,
                         memory_is_causal=memory_is_causal,
                         skip_self_attn=skip_self_attn)

        if self.norm is not None:
            output = self.norm(output)

        return output
    


class LeveledPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, level: int, base: int=2, overlap: int =0, dropout: float = 0.1, max_len: int = 5000, randomised: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.emb = nn.Embedding(level*base, d_model)

        self.level = level
        self.base = base
        self.overlap = overlap # offset between levels
        self.max_len = max_len
    
    def forward(self, x: Tensor) -> Tensor:
        B, T = x.shape
        device, dytpe = x.device, x.dtype
        
        max_level = math.ceil(math.log(T, self.base))

        # compute level
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0).repeat(B,1)
        level = torch.arange(0, max_level, dtype=torch.long, device=device).unsqueeze(1).repeat(1,T)
    
        pos = pos * (level + 1)
        pos = pos % self.base
        pos = pos + level * self.base

        # compute embedding
        x = self.emb(pos)

        return self.dropout(x)

class TokenProject(nn.Module):
    def __init__(self, d_model: int, outdim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, outdim)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        # load weights from state_dict
        if assign:
            self.proj.weight = nn.Parameter(state_dict['proj.weight'])
            self.proj.bias = nn.Parameter(state_dict['proj.bias'])
        else:
            self.proj.load_state_dict(state_dict)
        
        # freeze the weights
        for param in self.parameters():
            param.requires_grad = False

        return self 
    
    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)
