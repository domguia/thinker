import torch
from torch import nn, optim
from core.compressor.model_wrapper import HFModelWrapper
from typing import Optional, List

class SoftPromptOptimizer:
    def __init__(self, wrapper: HFModelWrapper):
        self.wrapper = wrapper

    def optimize(self, target_text: str, n_prompt_tokens: int = 10, n_steps: int = 100, lr: float = 0.1):
        target_ids = self.wrapper.encode(target_text) # (1, seq_len)
        target_len = target_ids.shape[1]
        
        # Initialize prompt embeddings randomly (could also use some mean embedding)
        # Using some real token embeddings as starting point might help
        init_ids = torch.randint(0, self.wrapper.model.config.vocab_size, (1, n_prompt_tokens), device=self.wrapper.device)
        prompt_embeddings = self.wrapper.get_embeddings(init_ids).detach().clone()
        prompt_embeddings.requires_grad = True
        
        optimizer = optim.Adam([prompt_embeddings], lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        
        target_embeddings = self.wrapper.get_embeddings(target_ids) # (1, seq_len, d_model)
        
        for i in range(n_steps):
            optimizer.zero_grad()
            
            # Concatenate prompt and target embeddings for causal prediction
            # The model will predict the next token at each position
            full_embeddings = torch.cat([prompt_embeddings, target_embeddings[:, :-1]], dim=1) # (1, prompt_len + seq_len - 1, d_model)
            
            logits = self.wrapper.forward_with_embeddings(full_embeddings) # (1, total_len, vocab_size)
            
            # We care about the predictions for the target tokens
            # The prediction at the end of the prompt is for the first target token
            # The prediction at target_ids[t] is for target_ids[t+1]
            target_logits = logits[:, n_prompt_tokens-1:, :] # (1, seq_len, vocab_size)
            
            loss = loss_fn(target_logits.view(-1, target_logits.size(-1)), target_ids.view(-1))
            
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Step {i}: Loss = {loss.item():.4f}")
                
        return prompt_embeddings.detach(), loss.item()

    def get_discrete_prompt(self, soft_prompt: torch.Tensor) -> List[int]:
        """Find the nearest discrete tokens for a soft prompt embedding."""
        # soft_prompt: (1, n_prompt_tokens, d_model)
        all_embeddings = self.wrapper.model.get_input_embeddings().weight # (vocab_size, d_model)
        
        # Find nearest neighbor for each embedding in the soft prompt
        # Use cosine similarity or MSE
        discrete_ids = []
        for i in range(soft_prompt.shape[1]):
            emb = soft_prompt[0, i]
            # Euclidean distance
            distances = torch.norm(all_embeddings - emb, dim=1)
            token_id = torch.argmin(distances).item()
            discrete_ids.append(token_id)
            
        return discrete_ids
