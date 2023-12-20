
from torch import nn
from torch import Tensor
from typing import Optional
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class FlexTransformerDecoderLayer(TransformerDecoderLayer): # Not the real flex model! the real one is really flexible!!!
    r"""
        Look at original implementation below
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer
    """

    def __init__(self, *args, **kwargs):
        self.skip_self_attn = False
        if 'skip_self_attn' in kwargs:
            self.skip_self_attn = kwargs['skip_self_attn']
            del kwargs['skip_self_attn']

        super().__init__(*args, **kwargs)

        if self.skip_self_attn:
            # remove parameters that we don't need
            self.skip_self_attn = True
            self.self_attn = nn.Identity()
            self.norm1 = nn.Identity()
            self.dropout1 = nn.Identity()
            self.self_attn.batch_first = True
    
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
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else: # default case we runing in this project
            if not skip_self_attn: # disable self attention block
                x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

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