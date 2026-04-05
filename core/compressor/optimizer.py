import torch
from torch import nn, optim
from core.compressor.model_wrapper import HFModelWrapper
from typing import Optional, List

class SoftPromptOptimizer:
    def __init__(self, wrapper: HFModelWrapper):
        self.wrapper = wrapper

    def optimize(self, target_text: str, n_prompt_tokens: int = 10, n_steps: int = 100, lr: float = 0.1, 
                 loss_threshold: float = 0.001, patience: int = 20):
        target_ids = self.wrapper.encode(target_text) # (1, seq_len)
        target_len = target_ids.shape[1]
        
        # Initialize prompt embeddings
        init_ids = torch.randint(0, self.wrapper.model.config.vocab_size, (1, n_prompt_tokens), device=self.wrapper.device)
        prompt_embeddings = self.wrapper.get_embeddings(init_ids).detach().clone()
        prompt_embeddings.requires_grad = True
        
        optimizer = optim.Adam([prompt_embeddings], lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        
        target_embeddings = self.wrapper.get_embeddings(target_ids) # (1, seq_len, d_model)
        
        best_loss = float('inf')
        steps_without_improvement = 0
        
        for i in range(n_steps):
            optimizer.zero_grad()
            
            full_embeddings = torch.cat([prompt_embeddings, target_embeddings[:, :-1]], dim=1)
            logits = self.wrapper.forward_with_embeddings(full_embeddings)
            target_logits = logits[:, n_prompt_tokens-1:, :] # (1, seq_len, vocab_size)
            
            loss = loss_fn(target_logits.view(-1, target_logits.size(-1)), target_ids.view(-1))
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            
            # Check for early stopping
            if current_loss < best_loss - 1e-5:
                best_loss = current_loss
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                
            if current_loss < loss_threshold:
                print(f"Step {i}: Converged (Loss {current_loss:.4f} < {loss_threshold})")
                break
                
            if steps_without_improvement >= patience:
                print(f"Step {i}: Early stopping (No improvement for {patience} steps)")
                break

            if i % 20 == 0:
                print(f"Step {i}: Loss = {current_loss:.4f}")
                
        return prompt_embeddings.detach(), best_loss

    def hierarchical_search(self, target_text: str, max_prompt_tokens: int = 100, 
                            loss_threshold: float = 0.01, steps_per_increment: int = 50):
        """
        Incrementally add prompt tokens until the loss threshold is reached.
        This follows the 'hierarchical latent' idea.
        """
        n_prompt = 1
        current_embeddings = None
        
        while n_prompt <= max_prompt_tokens:
            print(f"\n--- Testing n_prompt = {n_prompt} ---")
            # If we have previous embeddings, use them as starting point
            if current_embeddings is not None:
                new_token_id = torch.randint(0, self.wrapper.model.config.vocab_size, (1, 1), device=self.wrapper.device)
                new_embedding = self.wrapper.get_embeddings(new_token_id).detach()
                # We could try to freeze previous embeddings here, but let's start with full optimization
                # for simplicity unless specific 'progressive' behavior is needed.
            
            emb, loss = self.optimize(target_text, n_prompt_tokens=n_prompt, n_steps=steps_per_increment, 
                                     loss_threshold=loss_threshold)
            
            if loss < loss_threshold:
                print(f"Goal reached with {n_prompt} tokens (Loss: {loss:.4f})")
                return emb, n_prompt
            
            # Heuristic for next size: double or +5
            if n_prompt < 10: n_prompt += 2
            elif n_prompt < 50: n_prompt += 10
            else: n_prompt += 25
            
        return emb, n_prompt

    def optimize_batch(self, target_texts: List[str], n_prompt_tokens: int = 10, n_steps: int = 100, lr: float = 0.1,
                       loss_threshold: float = 0.001, patience: int = 20):
        """Optimize prompts for multiple texts in parallel."""
        # Encode all texts
        encoded = [self.wrapper.tokenizer.encode(t, return_tensors="pt")[0] for t in target_texts]
        max_len = max(len(e) for e in encoded)
        
        # Pad target IDs
        target_ids = torch.full((len(target_texts), max_len), -100, device=self.wrapper.device)
        for i, e in enumerate(encoded):
            target_ids[i, :len(e)] = e.to(self.wrapper.device)
            
        # Initialize prompt embeddings (shared prompt length for the batch)
        init_ids = torch.randint(0, self.wrapper.model.config.vocab_size, (len(target_texts), n_prompt_tokens), device=self.wrapper.device)
        prompt_embeddings = self.wrapper.get_embeddings(init_ids).detach().clone()
        prompt_embeddings.requires_grad = True
        
        optimizer = optim.Adam([prompt_embeddings], lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Target embeddings for teacher forcing
        # We need to handle padding here too
        target_embeddings = self.wrapper.model.get_input_embeddings()(torch.clamp(target_ids, min=0))
        
        best_loss = float('inf')
        steps_without_improvement = 0
        
        for i in range(n_steps):
            optimizer.zero_grad()
            
            # Concatenate prompt and target (shifted for next-token prediction)
            full_embeddings = torch.cat([prompt_embeddings, target_embeddings[:, :-1]], dim=1)
            
            logits = self.wrapper.forward_with_embeddings(full_embeddings)
            target_logits = logits[:, n_prompt_tokens-1:, :]
            
            loss = loss_fn(target_logits.reshape(-1, target_logits.size(-1)), target_ids.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            if current_loss < best_loss - 1e-5:
                best_loss = current_loss
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                
            if current_loss < loss_threshold:
                print(f"Step {i}: Batch converged (Loss {current_loss:.4f})")
                break
            if steps_without_improvement >= patience:
                break
                
        return prompt_embeddings.detach(), best_loss

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
