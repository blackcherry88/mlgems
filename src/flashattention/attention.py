import torch


def basic_attention(Q, K, V):
    # Compute the attention scores
    scale = Q.shape[-1] ** -0.5
    scores = Q @ K.transpose(-2, -1) * scale
    # Apply softmax to get the attention weights
    weights = torch.softmax(scores, axis=-1)
    # Compute the output as a weighted sum of the values
    output = weights @ V
    return output


def einsum_attention(Q, K, V):
    # Compute the attention scores
    scale = Q.shape[-1] ** -0.5
    scores = torch.einsum('...qd,...kd->...qk', Q, K) * scale
    # Apply softmax to get the attention weights
    weights = torch.softmax(scores, axis=-1)
    # Compute the output as a weighted sum of the values
    output = torch.einsum('...qk,...kd->...qd', weights, V)
    return output


def stream_attention(Q, K, V, q_block_size=64, kv_block_size=64):
    seq_len_q, d_k = Q.shape[-2:]
    seq_len_k = K.shape[-2]
    d_v = V.shape[-1]
    batch_shape = torch.broadcast_shapes(
        Q.shape[:-2],
        K.shape[:-2],
        V.shape[:-2],
    )
    scale = d_k ** -0.5
    output_blocks = []

    for q_start in range(0, seq_len_q, q_block_size):
        q_end = min(q_start + q_block_size, seq_len_q)
        q_block = Q[..., q_start:q_end, :]
        block_len_q = q_end - q_start

        max_score = Q.new_full(
            (*batch_shape, block_len_q),
            float("-inf"),
        )
        denominator = Q.new_zeros((*batch_shape, block_len_q))
        accumulator = V.new_zeros((*batch_shape, block_len_q, d_v))

        for kv_start in range(0, seq_len_k, kv_block_size):
            kv_end = min(kv_start + kv_block_size, seq_len_k)
            k_block = K[..., kv_start:kv_end, :]
            v_block = V[..., kv_start:kv_end, :]

            scores = q_block @ k_block.transpose(-2, -1) * scale
            block_max = scores.max(dim=-1).values
            new_max = torch.maximum(max_score, block_max)
            correction = torch.exp(max_score - new_max)
            weights = torch.exp(scores - new_max[..., None])

            denominator = denominator * correction + weights.sum(dim=-1)
            accumulator = (
                accumulator * correction[..., None] + weights @ v_block
            )
            max_score = new_max

        output_blocks.append(accumulator / denominator[..., None])

    return torch.cat(output_blocks, dim=-2)
