import unittest

import torch

# from flash_attn import flash_attention
import torch.nn.functional as F


def attention_inner_heads_flash(qkv, num_heads):
    """Computes attention with heads inside of qkv in the channel dimension using FlashAttention.

    Args:
        qkv: Tensor of shape (B, 3*H*C, T) with Qs, Ks, and Vs, where:
            H = number of heads,
            C = number of channels per head.
        num_heads: number of heads.

    Returns:
        Attention output of shape (B, H*C, T).
    """

    # bs, width, length = qkv.shape
    # ch = width // (3 * num_heads)

    # # Split into (q, k, v) of shape (B, H*C, T).
    # q, k, v = qkv.chunk(3, dim=1)

    # # Rescale q and k. This makes them contiguous in memory.
    # scale = ch ** (-1 / 4)  # scale with 4th root = scaling output by sqrt
    # q = q * scale
    # k = k * scale

    # # Reshape q, k, v to (B, H, T, C) for FlashAttention
    # q = q.view(bs, num_heads, ch, length).permute(0, 1, 3, 2)  # (B, H, T, C)
    # k = k.view(bs, num_heads, ch, length).permute(0, 1, 3, 2)  # (B, H, T, C)
    # v = v.view(bs, num_heads, ch, length).permute(0, 1, 3, 2)  # (B, H, T, C)

    # # Compute attention using FlashAttention
    # out = flash_attention(q, k, v)  # (B, H, T, C)

    # # Reshape back to (B, H*C, T)
    # out = out.permute(0, 1, 3, 2).reshape(bs, num_heads * ch, length)
    # return out
    bs, width, length = qkv.shape
    ch = width // (3 * num_heads)

    # Split into (q, k, v) and reshape directly to (B, H, T, C)
    q, k, v = qkv.chunk(3, dim=1)
    q = q.view(bs, num_heads, ch, length).transpose(2, 3)  # (B, H, T, C)
    k = k.view(bs, num_heads, ch, length).transpose(2, 3)  # (B, H, T, C)
    v = v.view(bs, num_heads, ch, length).transpose(2, 3)  # (B, H, T, C)

    # Compute scaled dot-product attention
    out = F.scaled_dot_product_attention(q, k, v)  # (B, H, T, C)

    # Reshape back to (B, H*C, T) in one step
    out = out.transpose(2, 3).reshape(bs, num_heads * ch, length)
    return out



class TestAttentionInnerHeadsFlash(unittest.TestCase):
    def setUp(self):
        # Set up common test variables
        self.batch_size = 2
        self.num_heads = 4
        self.seq_len = 8
        self.channels_per_head = 16
        self.qkv_dim = 3 * self.num_heads * self.channels_per_head

        # Create a random qkv tensor
        self.qkv = torch.randn(self.batch_size, self.qkv_dim, self.seq_len, dtype=torch.float32)

    def attention_inner_heads_old(self, qkv, num_heads):
        """Original implementation of attention for reference."""
        bs, width, length = qkv.shape
        ch = width // (3 * num_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = ch ** (-1 / 4)
        q = q * scale
        k = k * scale
        q = q.view(bs * num_heads, ch, length)
        k = k.view(bs * num_heads, ch, length)
        v = v.reshape(bs * num_heads, ch, length)
        weight = torch.einsum("bct,bcs->bts", q, k)
        weight = torch.softmax(weight.float(), dim=-1).to(weight.dtype)
        out = torch.einsum("bts,bcs->bct", weight, v)
        return out.reshape(bs, num_heads * ch, length)

    def test_attention_inner_heads_flash(self):
        # Compute the output using the old and flash attention implementations
        output_old = self.attention_inner_heads_old(self.qkv, self.num_heads)
        output_flash = attention_inner_heads_flash(self.qkv, self.num_heads)

        # Verify the shapes are the same
        self.assertEqual(output_old.shape, output_flash.shape)

        # Verify that the outputs are close (numerically similar)
        self.assertTrue(torch.allclose(output_old, output_flash, atol=1e-5))

if __name__ == "__main__":
    unittest.main()