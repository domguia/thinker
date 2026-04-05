import torch
import math
from typing import List

class CompressionMetrics:
    @staticmethod
    def calculate_bit_cost_discrete(token_ids: List[int], vocab_size: int) -> float:
        """Cost of storing discrete token indices."""
        return len(token_ids) * math.log2(vocab_size)

    @staticmethod
    def calculate_correction_bit_cost(ranks: torch.Tensor) -> float:
        """
        Estimate the cost of storing correction ranks in bits.
        Uses Shannon entropy of the rank distribution.
        """
        ranks_flat = ranks.view(-1).cpu().float()
        total_tokens = len(ranks_flat)
        
        if total_tokens == 0:
            return 0.0

        # Calculate frequency of each rank
        unique, counts = torch.unique(ranks, return_counts=True)
        probs = counts.float() / total_tokens
        
        # Entropy H = -sum(p * log2(p))
        entropy = -torch.sum(probs * torch.log2(probs)).item()
        
        return entropy * total_tokens

    @staticmethod
    def summary(prompt_tokens: List[int], ranks: torch.Tensor, vocab_size: int, target_tokens: List[int]) -> dict:
        prompt_bits = CompressionMetrics.calculate_bit_cost_discrete(prompt_tokens, vocab_size)
        correction_bits = CompressionMetrics.calculate_correction_bit_cost(ranks)
        total_compressed_bits = prompt_bits + correction_bits
        
        original_bits = len(target_tokens) * math.log2(vocab_size)
        
        ratio = original_bits / total_compressed_bits if total_compressed_bits > 0 else 0
        
        return {
            "prompt_bits": prompt_bits,
            "correction_bits": correction_bits,
            "total_bits": total_compressed_bits,
            "original_bits": original_bits,
            "compression_ratio": ratio,
            "accuracy": (ranks == 0).float().mean().item()
        }
