import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class ToyThinker(nn.Module):

    def __init__(self, ntoken: int, max_latent: int, max_input_len: int, output_len: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.compute_step = TransformerDecoder(decoder_layers, nlayers)

        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_model, dropout, batch_first=True) # n_hid = d_model*2 because shoul be smaller
        self.compute_output = TransformerDecoder(decoder_layers, 1) # only one layer

        self.embedding = nn.Embedding(ntoken, d_model)
        self.latent_embedding = nn.Embedding(max_latent, d_model)
        self.pos_embedding_in  = nn.Embedding(max_input_len, d_model)
        self.pos_embedding_out = nn.Embedding(output_len, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.linear_probe = nn.Linear(d_model, 1)

        self.ntoken = ntoken
        self.max_input_len = max_input_len
        self.output_len = output_len

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.latent_embedding.weight.data.uniform_(-initrange, initrange)
        self.pos_embedding_in.weight.data.uniform_(-initrange, initrange)
        self.pos_embedding_out.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor, n_latent: int = None, n_target: int = None, n_step: int = 1) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        B, T = x.shape

        pos = torch.arange(0, max(T,n_latent,n_target), dtype=torch.long, device=x.device).unsqueeze(0).repeat(B,1) # shape (1, t)
        # x = self.pos_embedding_in(pos) + self.embedding(x)
        x = self.embedding(x) + self.pos_embedding_in(pos[:,:T])

        # x = self.embedding(x) * math.sqrt(self.d_model)
        # x = self.pos_encoder(x)
        if n_target == None: n_target = T
        if n_latent == None: n_latent = 8

        out_query = self.pos_embedding_out(pos[:,:n_target])
        latent = self.latent_embedding(pos[:,:n_latent])

        memory = x
        latent = out_query
        for i in range(n_step):
            # self.transformer_decoder.forward(tgt, memory, ...
            latent = self.compute_step(latent, memory)
            memory = torch.cat((memory,latent), dim=1)

        output_emb = self.compute_output(out_query, memory)
        output = self.linear(output_emb)
        output_probe = self.linear_probe(output_emb)

        return output, output_probe
    
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
    

if __name__ == '__main__':
    model = TransformerModel(ntoken=17, d_model=32, output_len=5, nhead=4, d_hid=8,
                 nlayers=1, dropout=0.1)
    x = torch.randint(0,16, (2,5))
    y = model(x)
    print("y.shape:", y.shape)

