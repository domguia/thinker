import torch
import unittest
import math
from core.compressor.metrics import CompressionMetrics
from core.compressor.model_wrapper import HFModelWrapper

class TestCompressor(unittest.TestCase):
    def test_correction_bit_cost(self):
        # Case 1: All ranks are 0 (perfect prediction)
        # Entropy of a single-value distribution is 0
        ranks = torch.zeros((1, 10))
        cost = CompressionMetrics.calculate_correction_bit_cost(ranks)
        self.assertEqual(cost, 0.0)

        # Case 2: Uniform distribution over 2 values (0 and 1)
        # Entropy should be 1 bit per token
        ranks = torch.tensor([0, 1, 0, 1]).float()
        cost = CompressionMetrics.calculate_correction_bit_cost(ranks)
        # Total bits = entropy (1.0) * n_tokens (4) = 4.0
        self.assertAlmostEqual(cost, 4.0)

    def test_token_ranks_logic(self):
        # Manually verify vectorized rank calculation
        # logits: (batch=1, seq=1, vocab=4)
        logits = torch.tensor([[[10.0, 5.0, 20.0, 0.0]]]) 
        # Sorted indices (desc): [2, 0, 1, 3] (values: 20, 10, 5, 0)
        # Ranks:
        # Token 2 -> Rank 0
        # Token 0 -> Rank 1
        # Token 1 -> Rank 2
        # Token 3 -> Rank 3
        
        wrapper = HFModelWrapper(model_name="gpt2", device="cpu") # We just need the method, not the model for this part
        
        target_ids = torch.tensor([[2]])
        ranks = wrapper.get_token_ranks(logits, target_ids)
        self.assertEqual(ranks.item(), 0)
        
        target_ids = torch.tensor([[0]])
        ranks = wrapper.get_token_ranks(logits, target_ids)
        self.assertEqual(ranks.item(), 1)
        
        target_ids = torch.tensor([[1]])
        ranks = wrapper.get_token_ranks(logits, target_ids)
        self.assertEqual(ranks.item(), 2)

    def test_compression_summary(self):
        prompt_tokens = [1, 2, 3] # 3 tokens
        target_tokens = [1, 2, 3, 4, 5, 6] # 6 tokens
        vocab_size = 50257
        ranks = torch.zeros((1, 6)) # Perfect prediction
        
        summary = CompressionMetrics.summary(prompt_tokens, ranks, vocab_size, target_tokens)
        
        # original_bits = 6 * log2(50257)
        # prompt_bits = 3 * log2(50257)
        # correction_bits = 0
        # ratio = (6 * log2) / (3 * log2 + 0) = 2.0
        self.assertAlmostEqual(summary['compression_ratio'], 2.0)
        self.assertEqual(summary['accuracy'], 1.0)

if __name__ == '__main__':
    unittest.main()
