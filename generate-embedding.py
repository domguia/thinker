# Goal of this file : generate embdding for wiki dataset using CausalLM model "susnato/phi-1_5_dev" from huggingface

import os
import torch
from transformers import PhiForCausalLM, AutoTokenizer, AutoModelForCausalLM
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# define the model
# model = PhiForCausalLM.from_pretrained("susnato/phi-1_5_dev")
model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

'''
Phi Model config:

 Phi 2   : https://huggingface.co/microsoft/phi-2/blob/main/config.json
  "n_embd": 2560,
  "n_head": 32,
  "n_layer": 32,
  "n_positions": 2048,
  "rotary_dim": 32,
  "vocab_size": 51200
  intermediate_size = 4*2560 (should be checked)

  24

  
Phi 1.5 : https://huggingface.co/microsoft/phi-1_5/blob/main/config.json
  "n_embd": 2048,
  "n_layer": 24,
  intermediate_size = 8192
  ... all same with Ph 2

  
For Kmeans, clustering:
https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/kmeans_vector_quantizer.py

all data -> find kmeans cluster for k=10, k=100, k=500, k=1000, k=10_000
  -> get number of element in each kmeans cluster
  -> distribute them according to frequency among total available static_memory eg. cluster 3 --map-> embedding : 2,3,4,5,6,7

at training -> map target_emb -> cluster -> static mem embedding (we should be able to keep undercontrol the number of embedding in context)
at inference -> put all static memory in context and let the model decide

'''

## FP32 / CPU
# model = PhiForCausalLM.from_pretrained("susnato/phi-1_5_dev", torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)

## FP16 / Flash-Attention / CUDA
# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, device_map="cuda", trust_remote_code=True)

## load model fp16 cpu
# model = PhiForCausalLM.from_pretrained("susnato/phi-1_5_dev", from_tf=True)

## define tokenizer
# tokenizer = AutoTokenizer.from_pretrained("susnato/phi-1_5_dev")


if False:
    # Tokenize the input
    input_text = 'How big is London'
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    with torch.no_grad():
        # Get the model's output
        output = model(input_ids)

    print("input_ids:", input_ids.shape)
    # batch_size, sequence_length, hidden_size

    # The last hidden state is the first element of the output tuple
    last_hidden_state = output[0]

    # You can take the mean of the last hidden state to get a sentence embedding
    sentence_embedding = torch.mean(last_hidden_state, dim=1)

    print("sentence_embedding:", sentence_embedding.shape)


import re

def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

def split_file_by_pattern(filepath, pattern):
    with open(filepath, 'r') as f:
        buffer = ''
        for piece in read_in_chunks(f):
            buffer += piece
            while True:
                match = re.search(pattern, buffer)
                if match:
                    yield buffer[:match.start()]
                    buffer = buffer[match.end():]
                else:
                    break
        if buffer:
            yield buffer

pattern = r"\n = [^=]+ = "
file_path = "./data/wikitext-2-raw-v1/wikitext-2-raw/wiki.test.raw"

target_dir = "./data/wikitext-2-raw-v1/embedding/wikitext-2-raw/"
os.makedirs(target_dir, exist_ok=True)

max_tokens = 512
stride = 128

# Set pad_token as eos_token
tokenizer.pad_token = tokenizer.eos_token

from tqdm import tqdm
file_size = os.path.getsize(file_path)
pbar = tqdm(total=file_size)

n = 0
for i, text_article in enumerate(split_file_by_pattern(file_path, pattern)):
    n += 1
    pbar.update(len(text_article))
    # if len(text_article)>10:
    # if len(text_article)<512:
    if i%5==0:
        # print("text_article:",i, "text_article len:", len(text_article))

        # chunk are too large to fit in memory
        # split chunk into sequences of 512 tokens with overlap of 128 tokens
        tokens = tokenizer(text_article, return_tensors='pt', padding=True)

        # make chunks to fit in memory
        # batch_size = 10
        tokens_chuncks = [ {k: v[:,i:i+max_tokens] for k, v in tokens.items()} for i in range(0, tokens['input_ids'].shape[1], max_tokens-stride) ]

        first_chunk = None
        embeds = []
        for j, tokens in enumerate(tokens_chuncks):
            # print("number of sub sub chunk:", len(texts))
            # print("tokenized:", tokens['input_ids'].shape)

            with torch.no_grad():
                output = model(**tokens, output_hidden_states=True)

            last_hidden_state = output.hidden_states[-1].detach().cpu().squeeze(0)

            # print("last_hidden_state:", last_hidden_state.shape)

            if j == 0:  # first element
                embeds.append(last_hidden_state) # the first element
                # if last_hidden_state.size(0)>1: embeds.append(last_hidden_state[128:, :])
            else:
                embeds.append(last_hidden_state[128:, :])
            # embeds.append(last_hidden_state)

        # Concatenate the embeddings along the first dimension
        embeds = torch.cat(embeds, dim=0)

        # Reshape the tensor to (B*T, H)
        # embeds = [embed.view(-1, embed.shape[2]) for embed in embeds]
        # embeds = embeds.view(-1, embeds.shape[2]).shape

        # write embedding to file with numpy
        filename = target_dir + "/wiki.test.raw."+str(i)
        embeds.numpy().tofile(filename+".embedding.bin")
        # write token to file
        tokens['input_ids'].numpy().tofile(filename+".token.bin")
        # write text to file
        with open(filename+".txt", 'w') as f:
            f.write(text_article)

print("total chunk:", n)

# write a pytorch dataset for the text, token and embedding
import numpy as np
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    # dataset for embedding
    # expect directory of embedding files
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # get all embedding files
        self.files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f.endswith(".embedding.bin")]
        self.files.sort()
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # read embedding file
        embedding = np.fromfile(self.root_dir + self.files[idx], dtype=np.float32)
        # read token file
        token = np.fromfile(self.root_dir + self.files[idx].replace(".embedding.bin", ".token.bin"), dtype=np.int64)
        # read text file
        with open(self.root_dir + self.files[idx].replace(".embedding.bin", ".txt"), 'r') as f:
            text = f.read()

        sample = {'embedding': embedding, 'token': token, 'text': text}

        return sample




# # concatenate all the ids in each dataset into one large file we can use for training
# for split, dset in tokenized.items():
#     arr_len = np.sum(dset["len"], dtype=np.uint64)
#     filename = destination_path / f"{split}.bin"
#     dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
#     arr = np.memmap(str(filename), dtype=dtype, mode="w+", shape=(arr_len,))
#     total_batches = 1024

#     idx = 0
#     for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
#         # Batch together samples for faster write
#         batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
#         arr_batch = np.concatenate(batch["ids"])
#         # Write into mmap
#         arr[idx : idx + len(arr_batch)] = arr_batch
#         idx += len(arr_batch)
#     arr.flush()

