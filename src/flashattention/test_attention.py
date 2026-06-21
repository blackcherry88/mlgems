import unittest

import torch

from attention import basic_attention, einsum_attention, stream_attention


class AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_basic_attention(self):
        Q = torch.zeros(2, 3, dtype=torch.float64)
        K = torch.randn(4, 3, dtype=torch.float64)
        V = torch.arange(8, dtype=torch.float64).reshape(4, 2)

        expected = V.mean(dim=0).expand(2, -1)

        torch.testing.assert_close(basic_attention(Q, K, V), expected)

    def test_einsum_attention_matches_basic_attention(self):
        Q = torch.randn(2, 3, 5, 8, dtype=torch.float64)
        K = torch.randn(2, 3, 7, 8, dtype=torch.float64)
        V = torch.randn(2, 3, 7, 4, dtype=torch.float64)

        torch.testing.assert_close(
            einsum_attention(Q, K, V),
            basic_attention(Q, K, V),
        )

    def test_stream_attention_2d(self):
        Q = torch.randn(5, 8, dtype=torch.float64)
        K = torch.randn(7, 8, dtype=torch.float64)
        V = torch.randn(7, 4, dtype=torch.float64)

        torch.testing.assert_close(
            stream_attention(Q, K, V, q_block_size=3, kv_block_size=2),
            basic_attention(Q, K, V),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_stream_attention_with_batch_and_heads(self):
        Q = torch.randn(2, 3, 5, 8, dtype=torch.float64)
        K = torch.randn(2, 3, 7, 8, dtype=torch.float64)
        V = torch.randn(2, 3, 7, 4, dtype=torch.float64)

        torch.testing.assert_close(
            stream_attention(Q, K, V, q_block_size=2, kv_block_size=3),
            basic_attention(Q, K, V),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_stream_attention_broadcasts_leading_dimensions(self):
        Q = torch.randn(2, 1, 5, 8, dtype=torch.float64)
        K = torch.randn(1, 3, 7, 8, dtype=torch.float64)
        V = torch.randn(1, 3, 7, 4, dtype=torch.float64)

        torch.testing.assert_close(
            stream_attention(Q, K, V, q_block_size=4, kv_block_size=5),
            basic_attention(Q, K, V),
            rtol=1e-10,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
