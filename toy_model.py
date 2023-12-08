import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, output_len: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout) # should be nn.TransformerDecoder
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken + output_len, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.ntoken = ntoken
        self.output_len = output_len

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        B, T = src.shape
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        out_query = self.ntoken + torch.arange(T, device=src.device)
        out_query = self.embedding(out_query.repeat(B,1))
        input = torch.cat((src, out_query), dim=1)
        
        output = self.transformer_encoder(input)
        output = self.linear(output[:,T:])
        return output
    
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

