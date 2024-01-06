import math
from typing import Optional #, Any, Union, Callable

import torch
from torch import nn, Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer, Transformer
from model_utils import FlexTransformerDecoder, FlexTransformerDecoderLayer, TokenProject

class ToyThinker(nn.Module):
    # inspered from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, vocab_size: int, max_latent: int, max_input_len: int, max_output_len: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, n_probe: int = 0, dropout: float = 0.1, static_mem_len:int = 0,
                 skip_self_attn:bool = False, ff_in_self_attn:bool = False):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout=0)
        # decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout, activation=nn.functional.gelu, batch_first=True)
        # self.attn_compute = TransformerDecoder(decoder_layers, nlayers) if nlayers>1 else decoder_layers

        decoder_layers = FlexTransformerDecoderLayer(d_model, nhead, d_hid, dropout, activation=nn.functional.gelu,
                                                     batch_first=True, skip_self_attn=skip_self_attn, ff_in_self_attn=ff_in_self_attn)
        self.attn_compute = FlexTransformerDecoder(decoder_layers, nlayers) if nlayers>1 else decoder_layers

        # decoder_layers = FlexTransformerDecoderLayer(d_model, nhead, d_model, dropout, activation=nn.functional.gelu, batch_first=True) #, skip_self_attn=True) # n_hid = d_model*2 because shoul be smaller
        # self.compute_output = FlexTransformerDecoder(decoder_layers, 1) # only one layer

        self.embd_vocab   = nn.Embedding(vocab_size, d_model)
        self.embd_latent  = nn.Embedding(max_latent, d_model)
        self.embd_in_pos  = nn.Embedding(max_input_len, d_model)
        self.embd_out_pos = nn.Embedding(max_output_len, d_model)

        # self.embd_vocab_out = nn.Embedding(vocab_size, d_model)
        self.emb_static_mem = nn.Embedding(static_mem_len, d_model)
        
        self.linear = nn.Linear(d_model, vocab_size + n_probe)

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.n_probe = n_probe

        self.ouput_is_causal = False
        self.randomise_output = False
        self.perturb_prob = 0.1

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        def _apply_init_weights(module):
            module.weight.data.uniform_(-initrange, initrange)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        [_apply_init_weights(module) for module in (
            self.embd_vocab,
            self.embd_latent,
            self.embd_in_pos,
            self.embd_out_pos,
            self.emb_static_mem,
            self.linear,
        )]

    def insert_or_remove_latent(self, latent, n_latent):
        '''
        This function is used to insert or remove latent variables from the given tensor.

        Parameters:
        latent (torch.Tensor): The input tensor from which latent variables are to be inserted or removed.
        n_latent (int): The number of latent variables to be inserted if positive, or removed if negative. 
                        If a decimal number is provided, it is used as a ratio while staying below max latent.

        Returns:
        torch.Tensor: The updated tensor after inserting or removing the latent variables.
        '''
        B, T, H = latent.shape
        if n_latent > 0:
            # insert n_latent
            new_latent = torch.randn((B, n_latent, H), device=latent.device)
            new_latent *= latent.detach().std(dim=1, keepdim=True)
            new_latent += latent.detach().mean(dim=1, keepdim=True)
            latent = torch.cat([latent, new_latent], dim=1)
        elif n_latent < 0:
            # remove n_latent
            for i in range(B):
                indices = torch.randperm(T)[:T + n_latent]
                latent[i] = latent[i, indices, :]
        return latent

    def pertub(self, latent, pertub_prob=0.1, n_latent_change_ratio=0.5):
        '''
        This function is used to add perturbation to the latent variables in the given tensor.

        Parameters:
        latent (torch.Tensor): The input tensor to which perturbation is to be added.
        pertub_prob (float or list): The probability of perturbation. If a list is provided, it gives the range of perturbation probability.
        n_latent_change_ratio (float or list): The probability of inserted or removed latent. If a list is provided, it gives the range of change ratio.

        Returns:
        torch.Tensor: The updated tensor after adding the perturbation.
        '''
        B, T, H = latent.shape
        if isinstance(pertub_prob, (list, tuple)):
            pertub_prob = torch.rand((B, 1, 1), device=latent.device) * (pertub_prob[1] - pertub_prob[0]) + pertub_prob[0]
        pertub = torch.randn_like(latent, device=latent.device)
        pertub = pertub * latent.detach().std(dim=1, keepdim=True) + latent.detach().mean(dim=1, keepdim=True)
        latent = torch.where(torch.rand((B, T, 1), device=latent.device) < pertub_prob, pertub, latent)

        if ratio := n_latent_change_ratio:
            if isinstance(ratio, float):
                ratio = [ratio, ratio]
            n_latent_change = torch.randint(int(-T*ratio[0]), int(T*ratio[1]), size=(1,)).item()
            latent = self.insert_or_remove_latent(latent, n_latent_change)

        return latent


    def forward(self, x: Tensor, target = None,
                n_latent = None, n_step: int = 1, read_step: int = 1e4, n_keep_output:int = 1,
                n_memory:int = 1e4, output_step:int = 1, knowledge_trigger = torch.LongTensor([]),
                is_full_ar:bool = False, is_output_ar:bool = False) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, vocab_size]``
        """
        B, T = x.shape

        if isinstance(target, torch.Tensor): n_target = target.shape[1]
        else: n_target = target if isinstance(target,int) else T
        print('n_target:',n_target)

        pos = torch.arange(0, max(T,n_latent,n_target), dtype=torch.long, device=x.device).unsqueeze(0).repeat(B,1) # shape (1, t)

        # offset = torch.randint(self.max_input_len-T, size=(B,1), device=x.device)
        # x = self.embd_vocab(x) + self.embd_in_pos(offset+pos[:,:T])

        x = self.embd_vocab(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # define init latent
        if n_latent == None: n_latent = 8
        latent = self.embd_latent(pos[:,:n_latent]) # B, L, H

        # define output query
        out_query = self.embd_out_pos(pos[:,:n_target])
        if isinstance(target, torch.Tensor):
            if isinstance(target, torch.FloatTensor):
                out_query += target
            elif isinstance(target, torch.LongTensor):
                assert is_output_ar, "is_output_ar must be True for target sequence of tokens"
                out_query += self.embd_vocab(target)

        if is_full_ar or is_output_ar:
            tgt_mask = Transformer.generate_square_subsequent_mask(n_target, latent.device)
        else: tgt_mask = None

        # autogressive ouput like Perceiver AR
        outputs_ar = None
        if is_full_ar: 
            assert isinstance(target, torch.Tensor) and target.dtype == torch.long, "target must be a sequence of token index"
            output_ar = self.embd_vocab(target) * math.sqrt(self.d_model)
            output_ar = self.pos_encoder(output_ar, offset=T)
            outputs_ar = []

        latents = []
        outputs = []
        memory = x
        static_mem = self.emb_static_mem(knowledge_trigger)
        # static_mem = self.emb_static_mem.data.weight if knowledge_trigger=='all' else self.emb_static_mem(knowledge_trigger)
        for i in range(n_step):
            # add pertubation to latent
            # if self.perturb_prob > random.random():
            #     latent = self.pertub(latent)

            # compute step
            latent = self.attn_compute(latent, memory)
            if is_full_ar:
                output_ar = self.attn_compute(output_ar, memory, tgt_mask=tgt_mask, tgt_is_causal=True) # Fix: avoid twice compute with mask attention
                outputs_ar.append(output_ar)

            # append to latents memory
            latents.append(latent)      # remove old memorised latents
            if len(latents) > n_memory: latents.pop(0) # first in first out

            # define context : memory + input
            memory = latents if i>=read_step else [x] + latents 
            # add static memory
            memory = memory + [static_mem] if static_mem.nelement() else memory
            memory = torch.cat(memory, dim=1)
            # print(i, f': process step: mem*{len(latents)}') if i>=read_step else print(i, f': process step: read + mem*{len(latents)}')

            if i >= (n_step - n_keep_output):
                output = out_query # avoid ereasing out_query 
                for j in range(output_step):
                    # compute output at the last step
                    # print(i, j, ': output compute step')
                    output = self.attn_compute(output, memory, tgt_mask=tgt_mask, tgt_is_causal=is_output_ar) # B, T, H
                    if j >= (output_step - 1):
                        # print(i, j, ': keep output')
                        outputs.append(output) # keep output

        outputs = torch.stack(outputs, dim=1) # B, S, T, H
        logits = self.linear(outputs)

        # split outputs
        probes = logits[:, :, :, self.vocab_size:] # B, S, T, n_probe
        logits = logits[:, :, :, :self.vocab_size] # B, S, T, vocab_size

        logits_ar, probes_ar = None, None
        if is_full_ar:
            outputs_ar = torch.stack(outputs_ar, dim=1) # B, S, T, H
            logits_ar = self.linear(outputs_ar)

            probes_ar = logits_ar[:, :, :, self.vocab_size:] # B, S, T, n_probe
            logits_ar = logits_ar[:, :, :, :self.vocab_size] # B, S, T, vocab_size

        return outputs, logits, probes, outputs_ar, logits_ar, probes_ar
    
    def compute_step(self, latent, memory, ouput_query=None):
        B, L, _ = latent.shape

        mask = None
        in_latent = latent

        # generate attention mask
        if ouput_query:
            _, C, _ = ouput_query.shape # number of causal elements
            mask = mask_attn((B,L+C,L+C), C, latent.device)
            in_latent = torch.cat((latent, ouput_query), dim=1)

        out_latent = self.attn_compute(in_latent, memory, tgt_mask=mask)

        # extract latent and output
        latent = out_latent[:,:L,:]
        output_emb = out_latent[:,L:,:]

        return latent, output_emb

def mask_attn(size, n_causal:int = 0, n_output:int = 0, is_causal_ouput:bool = False, device=None):
    # size = (B, T, T) or (T, T)
    # n_causal is number element were causal mask is applied, if None use all L elements
    #         0 0 0|0 0 0|0 0 1    1 1 1    
    #       O 0 0 0|0 0 0|0 1 1 or 1 1 1 for full attention output   
    #       __0_0_0|0_0_0|1_1_1    1_1_1    
    #         0 0 0|0 0 1|0 0 0    
    # key-> C 0 0 0|0 1 1|0 0 0    
    #       __0_0_0|1_1_1|0_0_0    
    #         1 1 1|1 1 1|1 1 1    L : Lantent full attention
    #       L 1 1 1|1 1 1|1 1 1    C : Causal attention, with full ar pass
    #         1 1 1|1 1 1|1 1 1    O : Output attention
    # query->   L  |  C  |  O       
    # TODO: important check tgt_is_causal=True with this kind of mask but I dont think so 
    
    # define param
    T = size[1]
    B = size[0] if len(size)==3 else 0
    C, O = n_causal, n_output
    L = T - C - O
    
    # init mask
    mask = torch.ones(size, device=device)

    # mask orginin is bottom, left
    mask[   L: ,    :  ] = 0 # mask output
    mask[ L+C: , L+C:  ] = torch.tril(torch.ones((O,O)), diagonal=O) if is_causal_ouput else 1
    mask[   L:C,   L:C ] = torch.tril(torch.ones((C,C)), diagonal=C) # fill causal attention

    if len(size)==3:
        mask = mask.repeat(B,1,1)
    return mask

def all_losses_compute(outs, target, target_emb=None, last_step_only = True):
    outputs, logits, probes, outputs_ar, logits_ar, probes_ar = outs
    # outputs: B, S, T, H
    # probes: B, S, T, n_probe
    # outputs_ar: B, S, T, H
    # logits_ar: B, S, T, vocab_size
    # probes_ar: B, S, T, n_probe
    # target: B, T

    def compute_token_pred_loss(logits, target, last_step_only=True):
        B, S, T, vocab_size = logits.shape
        # logits: B, S, T, vocab_size
        # target: B, T

        target_ = target.repeat(S,1,1).permute(1,0,2) # B, S, T wich is B, T -> S, B, T -> B, S, T
        logits_ = logits.permute(0,3,1,2) # B, vocab_size, S, T
        losses = nn.functional.cross_entropy(logits_, target_, reduction='none', ignore_index=-100)
        
        loss = losses[:,-1].mean() if last_step_only else losses.mean()

        return loss, losses.detach().cpu()
    
    def compute_emdedding_loss(output_emb, target_emb, last_step_only=True,
                               proj_model:nn.Module = None, proj_target:nn.Module = None,
                               method:str = 'mse'):
        # output_emb: B, S, T, H = hdim_model
        # target_emb: B, S, T, H = hdim_target
        hdim_model  = output_emb.size(-1)
        hdim_target = target_emb.size(-1)


        if hdim_model != hdim_target:
            assert proj_model, "proj_model should be provided if model hdim and target are different"
            # project model embedding to target
            output_emb = proj_model(output_emb) # B, S, T, H = hdim_target from hdim_model

        if method == 'mse':
            losses = nn.functional.mse_loss(output_emb, target_emb, reduction='none') # B, S, T, H
            losses = losses.mean(dim=-1) # B, S, T
        elif method == 'cross_entropy':
            assert proj_target, "proj_target should be provided if method is cross_entropy"
            
            # project target embedding to model
            target_emb = proj_target(target_emb) # B, S, T, H = proj_dim
            assert output_emb.size(-1) == target_emb.size(-1), "proj_dim should be the same for model and target"
            
            output_logits = output_emb.softmax(dim=-1).permute(0,3,1,2) # B, H, S, T
            target_logits = target_emb.softmax(dim=-1).permute(0,3,1,2) # B, H, S, T
            
            losses = nn.functional.cross_entropy(output_logits, target_logits, reduction='none') # B, S, T
        else:
            raise Exception(f"method {method} is not supported")

        loss = losses[:,-1].mean() if last_step_only else losses.mean()

        return loss, losses.detach().cpu()
    
    def compute_probe_loss(probes, target, last_step_only=True, factor=1):
        # probes: B, S, T, P = n_probe
        # target: B, T
        B, S, T, P = probes.shape

        target = target[:, None, :, None].repeat(1,S,1,P).float()
        losses = nn.functional.mse_loss(probes, target/factor, reduction='none') # B, S, T, P
        losses = losses.mean(dim=3) # B, S, T

        loss = losses[:,-1].mean() if last_step_only else losses.mean()

        return loss, losses.detach().cpu()

    # visualisation losses
    def compute_viz_loss(losses):
        # losses: B, S, T
        # compute loss over S and T
        losses = losses.detach().mean(dim=0)

        viz = dict()
        viz['last_step_seq_losses'] = losses[-1,:]
        viz['mean_losses_per_step'] = losses.mean(dim=1)

        return viz

    # Load projection model
    
    # proj_model  = TokenProject(outputs.size(-1), probes.size(-1))
    # proj_target = TokenProject(target_emb.size(-1), probes.size(-1))

    # proj_model.load_state_dict()
    # proj_target.load_state_dict()

    logs = []
    logs += [('token_pred',    compute_token_pred_loss(logits,    target,     last_step_only))]
    logs += [('token_pred_ar', compute_token_pred_loss(logits_ar, target,     last_step_only))] if logits_ar is not None else []
    logs += [('embd',          compute_emdedding_loss(outputs,    target_emb, last_step_only))] if target_emb is not None else []
    logs += [('embd_ar',       compute_emdedding_loss(outputs_ar, target_emb, last_step_only))] if not (outputs_ar is None or target_emb is None) else []
    logs += [('probe',         compute_probe_loss(probes,         target,     last_step_only, factor=4))]
    logs += [('probe_ar',      compute_probe_loss(probes_ar,      target,     last_step_only, factor=4))] if probes_ar is not None else []

    # compute loss for loss.backward()
    aggregated_loss = sum([loss for name, (loss, losses) in logs])

    # compute loss for visualization
    logs = [dict(name=name, loss=loss.detach().cpu(), losses=losses, **compute_viz_loss(losses)) for name, (loss, losses) in logs]

    return aggregated_loss, logs

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, randomised: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 2, d_model) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_len = max_len
        self.randomised = randomised

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        B, T, H = x.shape
        
        assert self.randomised==False or offset==0, "randomise can't work simultaneously with offset"

        if self.randomised:
            # resolution = torch.randint(1, int(T/self.max_len) + 1, size=(B,))
            # offset = torch.rand(size=(B,))
            # offset = offset*(self.max_len-T*resolution).clamp(min=0, max=self.max_len)
            # index = torch.arange(T).unsqueeze(0).repeat(B,1)
            # index = (index*resolution.unsqueeze(-1) + offset.unsqueeze(-1)).long()
            # x = x + self.pe[:, index, :].expand_as(x) # advanced indexing
            offset = torch.randint(self.max_len-T, size=(B,))
            x = x + self.pe[offset.unsqueeze(-1), :T].expand_as(x)  # apply offset to each batch item
            # x = x + self.pe[offset.unsqueeze(-1), :T].expand_as(x) # advanced indexing
        else:
            x = x + self.pe[:, offset:T+offset, :]
        return self.dropout(x)


if __name__ == '__main__':
    d_model = 512 # + 512
    d_hid = d_model * 4
    nhead = 32 * 1 # 1024/32=32
    nlayers = 1
    static_mem_len = 100 # 1024*1024
    
    def manual_parameters_count(d_model, d_hid, nlayers, static_mem_len):
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

        static_mem = static_mem_len*d_model
        print('static_mem:', static_mem)

        total_param = nlayers*decoder + static_mem
        print(f'Total {total_param:,}')

        return total_param

    manual_parameters_count(d_model, d_hid, nlayers, static_mem_len)

    model = ToyThinker(
         vocab_size=17, max_latent=4, max_input_len=7, max_output_len=15,
         d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers, n_probe=1, dropout=0.1,
         static_mem_len=static_mem_len)
    
    import torchinfo
    torchinfo.summary(model, depth = 4) # should print model summary

    from time import time
    start_time = time()
    x = torch.randint(0, 16, (2, 5))
    target = torch.randint(0, 16, (2, 3))
    knowledge_trigger = torch.randint(0, 16, (2, 10))
    print()
    outs = model(x, target = target, n_latent = 3,
                 n_step = 5, read_step = 2, n_keep_output = 2,
                 n_memory = 3, output_step = 2,
                 knowledge_trigger = knowledge_trigger,
                 is_full_ar = True, is_output_ar=True)
    outputs_emb, logits, probes, outputs_ar, logits_ar, probes_ar = outs


    print()
    print("Time:", time() - start_time)
    print("          x.shape:", x.shape)
    print("      ---:---     ")
    print("outputs_emb.shape:", outputs_emb.shape)
    print("     logits.shape:", logits.shape)
    print("     probes.shape:", probes.shape)

    if logits_ar is not None:
        print("      ---:---     ")
        print(" outputs_ar.shape:", outputs_ar.shape)
        print("  logits_ar.shape:", logits_ar.shape)
        print("  probes_ar.shape:", probes_ar.shape)


    loss, logs = all_losses_compute(outs, target, None, last_step_only=True)

    print('loss:', loss)
    print('logs:', logs)





