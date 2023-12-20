import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class ToyThinker(nn.Module):

    def __init__(self, vocab_size: int, max_latent: int, max_input_len: int, output_len: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, n_probe: int = 0, dropout: float = 0.1):
        super().__init__()
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout, activation=F.gelu, batch_first=True)
        self.compute_step = TransformerDecoder(decoder_layers, nlayers)

        decoder_layers = LiteTransformerDecoderLayer(d_model, nhead, d_model, dropout, activation=F.gelu, batch_first=True) # n_hid = d_model*2 because shoul be smaller
        self.compute_output = TransformerDecoder(decoder_layers, 1) # only one layer

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.latent_embedding = nn.Embedding(max_latent, d_model)
        self.pos_embedding_in  = nn.Embedding(max_input_len, d_model)
        self.pos_embedding_out = nn.Embedding(output_len, d_model)
        
        self.linear = nn.Linear(d_model, vocab_size + n_probe)

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_input_len = max_input_len
        self.output_len = output_len
        self.n_probe = n_probe

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        def _apply_init_weights(module):
            module.weight.data.uniform_(-initrange, initrange)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        [_apply_init_weights(module) for module in (
            self.embedding,
            self.latent_embedding,
            self.pos_embedding_in,
            self.pos_embedding_out,
            self.linear,
        )]

    def forward(self, x: Tensor, n_latent: int = None, n_target: int = None, n_step: int = 1, read_step: int = 1e4, step_output:int = 1, n_memory:int = -1) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, vocab_size]``
        """
        B, T = x.shape

        pos = torch.arange(0, max(T,n_latent,n_target), dtype=torch.long, device=x.device).unsqueeze(0).repeat(B,1) # shape (1, t)
        # x = self.pos_embedding_in(pos) + self.embedding(x)
        x = self.embedding(x) + self.pos_embedding_in(pos[:,:T])

        # x = self.embedding(x) * math.sqrt(self.d_model)
        # x = self.pos_encoder(x)
        if n_target == None: n_target = T
        if n_latent == None: n_latent = 8

        latent = self.latent_embedding(pos[:,:n_latent])
        out_query = self.pos_embedding_out(pos[:,:n_target])

        latents = []
        outputs = []
        memory = x
        for i in range(n_step):
            latent = self.compute_step(latent, memory)

            # append to latents memory
            latents.append(latent)
            if len(latents) > n_memory: latents.pop(0) # first in first out

            # compute context : memory + input
            memory = latents if i>read_step else latents + [x] 
            memory = torch.cat(memory, dim=1)

            if i >= (n_step - step_output):
                # compute output at the last step
                output = self.compute_output(out_query, memory) # B, T, H
                outputs.append(output)

        outputs = torch.cat(outputs, dim=1) # B, S, T, H
        outputs = self.linear(outputs)

        # split outputs
        logits = outputs[:, :, :self.vocab_size] # B, S, T, vocab_size
        probes = outputs[:, :, -self.n_probe:]   # B, S, T, n_probe

        return logits, probes
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class LiteTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        # we dont need self attention
        self.self_attn = None
        self.norm1 = None
        self.dropout1 = None
    
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
    ) -> Tensor:
        
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer.forward
        x = tgt
        if self.norm_first:
            # disable self attention block
            # x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else: # default case we runing in this project
            # disable self attention block
            # x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x

if __name__ == '__main__':
    model = TransformerModel(vocab_size=17, d_model=32, output_len=5, nhead=4, d_hid=8,
                 nlayers=1, dropout=0.1)
    x = torch.randint(0,16, (2,5))
    y = model(x, n_latent = 3, n_target = 5, n_step = 4, read_step = 2)
    print("y.shape:", y.shape)

