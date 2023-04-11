import torch
import torch.nn as nn
from torch.nn import functional as F

# FlexProjection Layer
class FlexProjection(nn.Module):
    def __init__(self, max_in_dim, max_out_dim):
        super(FlexProjection, self).__init__()
        self.matrix = nn.Parameter(torch.randn(max_in_dim, max_out_dim)) #TODO - initialize this with a better value

    def forward(self, x, in_dim=None, out_dim=None):
        if in_dim is not None and in_dim > self.matrix.shape[0]:
            raise ValueError("in_dim is too large for the projection matrix")
        if out_dim is not None and out_dim > self.matrix.shape[1]:
            raise ValueError("out_dim is too large for the projection matrix")
        
        in_dim = in_dim or x.shape[-1]
        out_dim = out_dim or self.matrix.shape[1]
        return x[..., :in_dim] @ self.matrix[:in_dim, :out_dim]

# FlexMultiheadAttention Layer
class FlexMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, flex_projection=None, **kwargs):
        super(FlexMultiheadAttention, self).__init__(embed_dim, num_heads, **kwargs)
        self.flex_projection = flex_projection or FlexProjection(embed_dim, embed_dim)

    def forward(self, query, key, value, **kwargs):
        q_in_dim, k_in_dim, v_in_dim = query.shape[-1], key.shape[-1], value.shape[-1]
        q_proj = self.flex_projection(query, in_dim=q_in_dim)
        k_proj = self.flex_projection(key, in_dim=k_in_dim)
        v_proj = self.flex_projection(value, in_dim=v_in_dim)
        
        return super(FlexMultiheadAttention, self).forward(q_proj, k_proj, v_proj, **kwargs)

class Th1nkerModel(nn.Module):
    def __init__(self, max_in_dim, max_out_dim, embed_dim, num_heads):
        super(Th1nkerModel, self).__init__()
        self.flex_projection = FlexProjection(max_in_dim, max_out_dim)
        self.flex_multihead_attention = FlexMultiheadAttention(embed_dim, num_heads, flex_projection=self.flex_projection)

    def forward(self, input_data):
        input_embeds = [self.flex_projection(inp) for inp in input_data]
        K = torch.stack(input_embeds, axis=1)
        V = K.clone()

        latent = torch.zeros(K.shape[0], K.shape[1], K.shape[2], dtype=K.dtype, device=K.device)

        Q = latent
        K_concat = torch.cat((K,latent), axis=1)
        V_concat = torch.cat((V,latent), axis=1)
        attn_output, _ = self.flex_multihead_attention(Q, K_concat, V_concat)

        # You can add additional layers and processing steps here as needed.

        return attn_output
