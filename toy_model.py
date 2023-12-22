import math
from typing import Optional #, Any, Union, Callable

import torch
from torch import nn, Tensor
# from torch.nn import TransformerDecoder, TransformerDecoderLayer
from model_utils import FlexTransformerDecoder, FlexTransformerDecoderLayer

class ToyThinker(nn.Module):
    # inspered from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, vocab_size: int, max_latent: int, max_input_len: int, max_output_len: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, n_probe: int = 0, dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout=0)
        decoder_layers = FlexTransformerDecoderLayer(d_model, nhead, d_hid, dropout, activation=nn.functional.gelu, batch_first=True)
        self.compute_step = FlexTransformerDecoder(decoder_layers, nlayers)

        decoder_layers = FlexTransformerDecoderLayer(d_model, nhead, d_model, dropout, activation=nn.functional.gelu, batch_first=True) #, skip_self_attn=True) # n_hid = d_model*2 because shoul be smaller
        self.compute_output = FlexTransformerDecoder(decoder_layers, 1) # only one layer

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.latent_embedding = nn.Embedding(max_latent, d_model)
        self.pos_embedding_in  = nn.Embedding(max_input_len, d_model)
        self.pos_embedding_out = nn.Embedding(max_output_len, d_model)
        
        self.linear = nn.Linear(d_model, vocab_size + n_probe)

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
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

    def forward(self, x: Tensor, n_latent: int = None, n_target: int = None, n_step: int = 1, read_step: int = 1e4, n_keep_output:int = 1, n_memory:int = 1e4, output_step:int = 1) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, vocab_size]``
        """
        B, T = x.shape

        pos = torch.arange(0, max(T,n_latent,n_target), dtype=torch.long, device=x.device).unsqueeze(0).repeat(B,1) # shape (1, t)

        offset = torch.randint(self.max_input_len-T, size=(B,1), device=x.device)
        x = self.embedding(x) + self.pos_embedding_in(offset+pos[:,:T])

        # x = self.embedding(x) * math.sqrt(self.d_model)
        # x = self.pos_encoder(x)

        if n_target == None: n_target = T
        if n_latent == None: n_latent = 8

        # define init latent
        latent = self.latent_embedding(pos[:,:n_latent])

        # define output query
        offset = torch.randint(self.max_output_len-n_target, size=(B,1), device=x.device)
        out_query = self.pos_embedding_out(offset + pos[:,:n_target])

        latents = []
        outputs = []
        memory = x
        for i in range(n_step):
            latent = self.compute_step(latent, memory, skip_self_attn = not i) # skip only for first layer

            # append to latents memory
            latents.append(latent)      # remove old memorised latents
            if len(latents) > n_memory: latents.pop(0) # first in first out

            # compute context : memory + input
            memory = latents if i>=read_step else latents + [x] 
            memory = torch.cat(memory, dim=1)
            # print(i, f': process step: mem*{len(latents)}') if i>=read_step else print(i, f': process step: read + mem*{len(latents)}')

            if i >= (n_step - n_keep_output):
                output = out_query # avoid ereasing out_query 
                for j in range(output_step):
                    # compute output at the last step
                    # print(i, j, ': output compute step')
                    output = self.compute_output(output, memory) # B, T, H
                    if j >= (output_step - 1):
                        # print(i, j, ': keep output')
                        outputs.append(output) # keep output

        outputs = torch.stack(outputs, dim=1) # B, S, T, H
        outputs = self.linear(outputs)

        # split outputs
        logits = outputs[:, :, :, :self.vocab_size] # B, S, T, vocab_size
        probes = outputs[:, :, :, self.vocab_size:] # B, S, T, n_probe

        return logits, probes
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 2, d_model) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # seq_len = x.size(0)
        # offset = torch.randint(self.max_len-seq_len, size=(1,)).item()
        # x = x + self.pe[offset : offset + seq_len]
        x = x + self.pe[:, :x.size(0)]
        return self.dropout(x)


if __name__ == '__main__':
    model = ToyThinker(
         vocab_size=17, max_latent=4, max_input_len=7, max_output_len=15,
         d_model=32, nhead=4, d_hid=32*4, nlayers=1, n_probe=0, dropout=0.1)
    
    import torchinfo
    torchinfo.summary(model, depth = 4) # should print model summary
    
    # manual computer parameters
    d_model=32
    d_hid=32*4
    
    # multiheadattention param count
    mha  = d_model*d_model # q_proj_weight: d_model * d_model
    mha += d_model*d_model # k_proj_weight: d_model * kdim
    mha += d_model*d_model # v_proj_weight: d_model * vdim
    mha += d_model*3       # in_proj_bias
    # out_proj = NonDynamicallyQuantizableLinear
    mha += d_model*d_model # out_proj.weight
    mha += d_model         # out_proj.bias
    print('multi head attention param count:', mha)

    decoder  = mha # self attention
    decoder += mha # cross attention
    decoder += d_model * d_hid  # linear1 weight
    decoder += d_hid   # linear1 bias
    decoder += d_hid * d_model  # linear2 weight
    decoder += d_model  # linear2 bias
    decoder += d_model*2  # layer norm 1
    decoder += d_model*2  # layer norm 2
    decoder += d_model*2  # layer norm 3
    print('decoder param count:', decoder)

    x = torch.randint(0,16, (2,5))
    print()
    output, probe = model(x,
                 n_latent = 3, n_target = 5,
                 n_step = 5, read_step = 2, n_keep_output = 2,
                 n_memory = 3, output_step = 2)
    print()
    print("     x.shape:", x.shape)
    print("output.shape:", output.shape)
    print(" probe.shape:", probe.shape)



