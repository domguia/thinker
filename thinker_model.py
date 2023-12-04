import math
import numpy as np
import torch
import torch.nn as nn

class Th1nker(nn.Module):
    def __init__(self, config):
        super(Th1nker, self).__init__()

        self.input_embedding = nn.Embedding(config.vocab_size, config.hdim)
        self.causal_output_embedding = nn.Embedding(config.vocab_size, config.hdim)
        self.query_output = nn.Embedding(config.max_output_len + 1, config.hdim)
        self.trained_latent = nn.Embedding(config.max_latent_size, config.hdim)
        
        self.position_encoder = PositionalEncoding(config.hdim, config.input_cache_size)

        ### BEGIN Block
        self.attn_kv_input = nn.Linear(config.hdim, 2 * config.hdim, bias=config.hdim)
        self.attn_kv_memory = nn.Linear(config.hdim, 2 * config.hdim, bias=config.bias)
        self.attn_sc = nn.Linear(config.hdim, 3 * config.hdim, bias=config.bias)

        self.ln_1 = nn.LayerNorm(config.hdim)
        self.ln_2 = nn.LayerNorm(config.hdim)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.hdim, 4 * config.hdim),
            c_proj  = nn.Linear(4 * config.hdim, config.hdim),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

        self.probe_proj = nn.Linear(config.hdim, 1)
        self.output_probe_proj = nn.Linear(config.hdim, 1)
        self.output_proj = nn.Linear(config.hdim, config.vocab_size)
        
        # init causal mask
        max_causal = config.max_output_len - 1
        max_q = config.max_latent_size + config.max_output_len
        max_k = config.input_cache_size + config.mem_cache_size + config.max_latent_size + config.max_output_len
        self.cached_mask = CachedAttentionMask(max_q, max_k, max_causal)
        ### END Block
        
        self.config = config
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def init(self, batch_size, latent_size):
        self.latent_size = latent_size
        self.output = None

        self.cached_mask.set_batch_size(batch_size)
        self.init_kv_cache(batch_size)
        self.init_latent(batch_size, latent_size)

    def init_kv_cache(self, batch_size):
        cache_size = self.config.input_cache_size + self.config.mem_cache_size + self.config.max_latent_size
        self.k_cache = torch.empty((batch_size, cache_size, self.config.hdim)) #, dtype=torch.bfloat16)
        self.v_cache = torch.empty((batch_size, cache_size, self.config.hdim)) #, dtype=torch.bfloat16)
        self.cache_input_length = 0 
        self.cache_mem_length = 0

    def init_latent(self, batch_size, latent_size=None):
        # we can use trained latent but randomly add and removed
        if latent_size==None:
            latent_size = self.config.max_latent_size
        self.latent = torch.randn((batch_size, latent_size, self.config.hdim))
        # broadcast first latent + some random
        self.latent = self.latent + self.trained_latent(torch.LongTensor([0]))

    def extend_kv_from_cache(self, k,v, input_lookup=True, mem_lookup=True):
        begin_input_idx = self.config.input_cache_size - self.cache_input_length
        end_input_idx = self.config.input_cache_size
        
        begin_mem_idx = end_input_idx
        end_mem_idx = self.config.input_cache_size + self.cache_mem_length
        
        begin_insert_idx = end_mem_idx
        end_insert_idx = begin_insert_idx + k.size(1)

        self.k_cache[:, begin_insert_idx:end_insert_idx] = k
        self.v_cache[:, begin_insert_idx:end_insert_idx] = v
        
        if mem_lookup: # NOTE: you can't lookup memory without lookup input
            begin_idx = begin_mem_idx
        if input_lookup:
            begin_idx = begin_input_idx
        k = self.k_cache[:, begin_idx:end_insert_idx]
        v = self.v_cache[:, begin_idx:end_insert_idx]
        return k,v

    def load_input(self, inputs):
        inputs = self.input_embedding(inputs)
        inputs = self.position_encoder(inputs)
        k, v = self.attn_kv_input(inputs).split(self.config.hdim, dim=2)

        # find where to insert in cache
        B, T, C = inputs.size()
        end_idx = self.config.input_cache_size - self.cache_input_length
        begin_idx = end_idx - T

        if begin_idx < 0:
            raise("KV Cache out of bounds, for input")
        
        self.k_cache[:, begin_idx:end_idx, :] = k
        self.v_cache[:, begin_idx:end_idx, :] = v

        self.cache_input_length += T
    
    def latent_to_memory(self, latent):
        k, v = self.attn_kv_memory(latent).split(self.config.hdim, dim=2)

        # find where to insert in cache
        B, T, C = latent.size()
        begin_idx = self.config.input_cache_size + self.cache_mem_length
        end_idx = begin_idx + T

        if begin_idx > self.config.input_cache_size + self.cache_mem_length:
            raise("KV Cache out of bounds, for memory")
            # TODO: implement rotation when cache is full
        
        self.k_cache[:, begin_idx:end_idx, :] = k
        self.v_cache[:, begin_idx:end_idx, :] = v

        self.cache_mem_length += T
    
    # def add_latent_probe(self, latent):
    def add_latent(self, latent, with_output=None, latent_probe=True, reverse=False):
        B, L, H = latent.shape
        causal_mask_len = None
        latents = [latent]
        
        begin, end = 1, 1
        if latent_probe:
            begin = 0
            # L += 1
            # latent.resize_((B, L, H))
            # latent[:,-1,:] = self.query_output[0]
            # latents.append(self.query_output(torch.LongTensor([0])))
        if isinstance(with_output, int):
            end = with_output + 1 # because we sart by 1 since 0 is for the probe
            # latent.resize_((B, L+n, H))
            # latent[:,L:L+n,:] = self.query_output[1:n]
        
        temp_latent = self.query_output(torch.arange(begin, end, dtype=torch.int32))
        latents.append(temp_latent.repeat(B, 1, 1))
        
        if isinstance(with_output, torch.Tensor):
            causal_mask_len = with_output.size(1)
            # latent.resize_((B, L+n, H))
            # latent[:,L:L+1,:] = self.query_output[1]
            # latent[:,L+1:L+n,:] = with_output[:,:-1,:] # do not consider the last element

            latents.append(self.query_output(torch.LongTensor([1])).repeat(B, 1, 1))
            latents.append(self.causal_output_embedding(with_output[:,:-1]))

        latent = torch.cat(latents, dim=1)
        
        return latent, causal_mask_len

    # def get_causal_mask(self, query, key, causal_mask_len):
    #     if causal_mask_len == None:
    #         return None
    #     # IMPROVEMENT: could be use mask cache that will be slice
    #     B = query.size(0)
    #     L, S = query.size(-2), key.size(-2)
    #     attn_bias = torch.ones(L, S, dtype=torch.bool).tril(diagonal=S-causal_mask_len)
    #     return attn_bias

    # @torch.compile
    def compute_step(self, with_output=None, input_lookup=True, mem_lookup=True, latent_to_memory=True, latent_probe=True, latent=None):
        latent = latent if latent else self.latent
        
        B, L, H = latent.shape
        latent = self.ln_1(latent)

        latent, causal_mask_len = self.add_latent(latent, with_output, latent_probe)

        q, k ,v  = self.attn_sc(latent).split(self.config.hdim, dim=2)
        k, v = self.extend_kv_from_cache(k, v, input_lookup, mem_lookup)

        n_head = self.config.number_of_head
        q = q.view(B, q.size(1), n_head, H // n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, k.size(1), n_head, H // n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, v.size(1), n_head, H // n_head).transpose(1, 2) # (B, nh, T, hs)
        
        ## get mask
        # mask = self.get_causal_mask(q,k, causal_mask_len)
        mask = self.cached_mask.get_mask(q.size(-2), k.size(-2), causal_mask_len)
        scale = 1.0 / math.sqrt(self.config.head_size)

        ## perfom attention.contiguous().view(B, -1, C)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=scale, attn_mask=mask, dropout_p=self.config.attn_pdrop
        )
        attn = attn.transpose(1, 2).contiguous().view(B, -1, H)
        attn = self.mlpf(attn)
        latent = latent + self.mlpf(self.ln_2(attn))

        ## latent post-processing
        self.latent = latent[:, :L, :]
        self.output_latent = latent[:, L:, :]
        self.latent_probe = latent_probe
        output = self.get_output(latent[:, L:, :], latent_probe)
        
        if latent_to_memory:
            self.latent_to_memory(self.latent)

        return output

    def forward(self, *args, **kwargs):
        return self.compute_step(*args, **kwargs)
    
    # @torch.compile
    def get_output(self, out_latent=None, latent_probe=None):
        if out_latent==None: out_latent = self.output_latent
        if latent_probe==None: latent_probe = self.latent_probe

        skip_first = 0
        probe, logits, outputs_probe = None, None, None

        if latent_probe:
            skip_first = 1
            probe = self.probe_proj(out_latent[:, 0, :])
        
        if out_latent.size(1) > skip_first:
            outputs_probe = self.output_probe_proj(out_latent[:, skip_first:, :]).squeeze()
            logits = self.output_proj(out_latent[:, skip_first:, :])
            # logits = nn.functional.softmax(outputs, dim=2)
        
        return probe, logits, outputs_probe

class LLamaLogprob(nn.Module):
    def __init__(self):
        self.proj = nn.Linear(4096, 32000)
    def forward(self, embed, softmax=True):
        logits = self.proj(embed)
        if softmax:
            return nn.functional.softmax(logits)
        return logits
    
class CachedAttentionMask:
    def __init__(self, max_q, max_k, max_causal):
        mask = torch.ones(max_q, max_k+max_causal, dtype=torch.bool).tril(diagonal=max_k-1)
        c = -max_causal # causal slice
        mask[:,c:].logical_not_()
        mask[:,c:] = mask[:,c:].flip(1)
        self.mask = mask
        self.max_q, self.max_k, self.max_causal = max_q, max_k, max_causal
    
    def set_batch_size(self, batch_size):
        if len(self.mask.shape) == 3:
            self.mask = self.mask[0,:,:]
        self.mask = self.mask.unsqueeze_(0).expand(batch_size, -1, -1)

    def get_mask(self, Q, K, causal_output_len):
        if causal_output_len==None: return None

        begin_q = self.max_causal - causal_output_len
        end_q = begin_q + Q

        end_k = self.max_k + self.max_causal - begin_q
        begin_k = end_k - K

        # print(f"{begin_q}:{end_q}, {begin_k}:{end_k}")
        if len(self.mask.shape) == 3:
            return self.mask[:, begin_q:end_q, begin_k:end_k]
        else:
            return self.mask[begin_q:end_q, begin_k:end_k]
    
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=5000): # , dropout=0.1
		super().__init__()
		# self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(max_len*2) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0) #.transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x, step_pos=None):
		if step_pos:
			return x + self.pe[:, :step_pos, :]
		x = x + self.pe[:, :x.size(1), :]
		return x
        # return self.dropout(x)

def compute_loss(output, targets, probe_mode="number_reg", probe_aggr="mean"): #, targets_base=None):
    probe, logits, outputs_probe = output

    # cross entropy loss again logits and targets
    logits = logits.transpose(1,2)
    output_losses = nn.functional.cross_entropy(logits, targets, ignore_index=-1, reduction="none")
    pred_loss = output_losses.mean(dim=1)

    if probe_mode == "certainity":
        ## certainity probe target is the accuracy of the prediction
        ## cross entropy allow wide range of possibilities
        outputs_probe_target = output_losses.negative().exp()
    if probe_mode == "top_certainity":
        ## looking at the top predcition
        max_indices = torch.argmax(logits, dim=1)
        outputs_probe_target = 1.0*(max_indices == torch.tensor(targets))
    if probe_mode == "number_reg": # default
        ## number prediction probe, just make regression on predicted word
        outputs_probe_target = outputs_probe

    outputs_probe_losses = nn.functional.mse_loss(outputs_probe_target, targets.float()/16, reduction="none")

    if probe_aggr=="mean":
        probe_target = outputs_probe_target.mean() # average of values
    if probe_aggr=="prod":
        probe_target = outputs_probe_target.prod() # strict: product of values [0,1]
    
    probe_loss = (probe_target-probe)**2

    loss = probe_loss + pred_loss + outputs_probe_losses.mean()
    return loss, probe_loss, pred_loss, output_losses, outputs_probe_losses

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def merge_from_dict(self, d):
        self.__dict__.update(d)
    def __call__(self, *args, **kwargs):
        self.__dict__.update(**kwargs)
        args = [item.strip() for items in args for item in items.split(',')]
        self.__dict__.update(**{name: globals()[name] for name in args})
    def __str__(self):
        return self.__dict__.__str__()
    
if __name__ == '__main__':

    config = CfgNode(
        hdim = 32,
        number_of_head = 4,
        head_size = 8,
        bias=False,

        resid_pdrop = 0.1,
        attn_pdrop = 0.1,

        vocab_size = 256,
        
        input_cache_size = 1024,
        mem_cache_size = 1024,

        max_latent_size = 128,
        max_output_len = 256,
    )
    model = Th1nker(config)
    
    B, T, H = 4, 64, config.hdim

    inputs = torch.randint(config.vocab_size, (B, T))
    targets = torch.randint(config.vocab_size, (B, T))
    # targets = torch.randn((B, T, config.vocab_size)).softmax(dim=-1) # with probability distribution
    
    latent = torch.rand(B, T, H)

    model.init(batch_size = B, latent_size = 32)
    model.load_input(inputs)
    model.compute_step()
    model.compute_step(with_output=T)
    model.compute_step(with_output=targets)
    output = model.get_output()
    # loss = compute_loss(output, targets)

    print("Thanks!")
    


    