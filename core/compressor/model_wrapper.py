import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional

class HFModelWrapper:
    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the continuous embeddings for a given set of input IDs."""
        return self.model.get_input_embeddings()(input_ids)

    def forward_with_embeddings(self, embeddings: torch.Tensor, past_key_values: Optional[tuple] = None) -> torch.Tensor:
        """Perform a forward pass using continuous embeddings."""
        outputs = self.model(inputs_embeds=embeddings, past_key_values=past_key_values)
        return outputs.logits

    def get_token_ranks(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        For each position, find the rank of the target token in the sorted logits.
        Rank 0 means the target token was the top-1 prediction.
        """
        # logits: (batch, seq_len, vocab_size)
        # target_ids: (batch, seq_len)
        
        # Sort logits in descending order
        # Vectorized rank calculation
        # Find position of target_ids in sorted logits
        # Efficiently get ranks for all positions
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)
        # This is a common trick to get the rank of each element
        # But we specifically want the rank of target_ids
        
        # Find which index in the sorted dimension corresponds to our target
        # ranks[b, t] is the index i such that sorted_indices[b, t, i] == target_ids[b, t]
        
        # Another way: 
        # For each (batch, pos), count how many logits are > target_logit
        target_logits = torch.gather(logits, -1, target_ids.unsqueeze(-1)) # (batch, seq_len, 1)
        ranks = torch.sum(logits > target_logits, dim=-1) # (batch, seq_len)
                
        return ranks

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
