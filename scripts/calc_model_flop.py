#!/usr/bin/env python3
"""
Calculate total FLOPs for a Transformer Language Model architecture.

This script computes the floating point operations (FLOPs) for a forward pass
through a transformer-based language model with the following architecture:
- Token embedding layer
- Multiple transformer blocks with:
  - RMSNorm
  - Multi-head attention with RoPE
  - SwiGLU feed-forward network
- Final linear projection to vocabulary
"""


def calculate_transformer_flops(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
) -> dict:
    """
    Calculate FLOPs for each component of a transformer language model.
    
    Args:
        batch_size: Batch size (B)
        context_length: Sequence length (T)
        vocab_size: Vocabulary size
        num_layers: Number of transformer layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
    
    Returns:
        Dictionary containing FLOPs for each component and total FLOPs
    """
    B = batch_size
    T = context_length
    
    # Calculate d_k and d_v (typically d_model / num_heads)
    d_k = d_model // num_heads
    d_v = d_model // num_heads
    
    flops = {}
    
    # 1. Embedding layer: 0 FLOPs (lookup operation, no multiplication)
    flops['embedding'] = 0
    
    # 2. Transformer blocks (per layer)
    # 2.1 RMSNorm: 0 FLOPs (element-wise operations only)
    flops['rmsnorm_per_layer'] = 0
    
    # 2.2 Multi-head attention
    # Down projection: x W_proj^T for Q, K, V
    # x: (B, T, d_model), W_proj^T: (d_model, 3 * num_heads * d_k)
    flops['attn_down_proj_per_layer'] = 2 * B * T * d_model * (3 * num_heads * d_k)
    
    # Scaled dot product attention
    # QK^T: Q: (B, num_heads, T, d_k), K^T: (B, num_heads, d_k, T)
    flops['attn_qk_per_layer'] = 2 * B * num_heads * T * d_k * T
    
    # weight * V: weight: (B, num_heads, T, T), V: (B, num_heads, T, d_v)
    flops['attn_weighted_v_per_layer'] = 2 * B * num_heads * T * T * d_v
    
    # Up projection: attn * W_O^T
    # attn: (B, T, num_heads * d_v), W_O^T: (num_heads * d_v, d_model)
    flops['attn_up_proj_per_layer'] = 2 * B * T * (num_heads * d_v) * d_model
    
    # Total attention FLOPs per layer
    flops['attention_per_layer'] = (
        flops['attn_down_proj_per_layer'] +
        flops['attn_qk_per_layer'] +
        flops['attn_weighted_v_per_layer'] +
        flops['attn_up_proj_per_layer']
    )
    
    # 2.3 Feed-forward network (SwiGLU)
    # xW_1^T: x: (B, T, d_model), W_1^T: (d_model, d_ff)
    flops['ffn_w1_per_layer'] = 2 * B * T * d_model * d_ff
    
    # xW_3^T: x: (B, T, d_model), W_3^T: (d_model, d_ff)
    flops['ffn_w3_per_layer'] = 2 * B * T * d_model * d_ff
    
    # result * W_2^T: result: (B, T, d_ff), W_2^T: (d_ff, d_model)
    flops['ffn_w2_per_layer'] = 2 * B * T * d_ff * d_model
    
    # Total FFN FLOPs per layer
    flops['ffn_per_layer'] = (
        flops['ffn_w1_per_layer'] +
        flops['ffn_w3_per_layer'] +
        flops['ffn_w2_per_layer']
    )
    
    # Total FLOPs per transformer layer
    flops['transformer_block_per_layer'] = (
        flops['rmsnorm_per_layer'] +
        flops['attention_per_layer'] +
        flops['ffn_per_layer']
    )
    
    # Total FLOPs for all transformer layers
    flops['all_transformer_blocks'] = flops['transformer_block_per_layer'] * num_layers
    
    # 3. Final linear layer: x W^T
    # x: (B, T, d_model), W^T: (d_model, vocab_size)
    flops['final_linear'] = 2 * B * T * d_model * vocab_size
    
    # 4. Total FLOPs
    flops['total'] = (
        flops['embedding'] +
        flops['all_transformer_blocks'] +
        flops['final_linear']
    )
    
    return flops


def format_number(num: int) -> str:
    """Format large numbers with commas for readability."""
    return f"{num:,}"


def main():
    # GPT-2 XL parameters
    params = {
        'model_name': 'GPT-2 XL (long context)',
        'vocab_size': 50257,
        'context_length': 1024,
        'num_layers': 48,
        'd_model': 1600,
        'num_heads': 25,
        'd_ff': 6400,
        'batch_size': 1024,  # Batch size per forward pass
    }
    
    print("=" * 80)
    print(f"Transformer LM FLOPs Calculation - {params['model_name']}")
    print("=" * 80)
    print("\nModel Parameters:")
    print(f"  Model Name: {params['model_name']}")
    print(f"  Vocabulary Size: {format_number(params['vocab_size'])}")
    print(f"  Context Length: {format_number(params['context_length'])}")
    print(f"  Number of Layers: {params['num_layers']}")
    print(f"  Model Dimension: {params['d_model']}")
    print(f"  Number of Heads: {params['num_heads']}")
    print(f"  FFN Hidden Dimension: {params['d_ff']}")
    print(f"  Batch Size: {format_number(params['batch_size'])}")
    print(f"  d_k = d_v: {params['d_model'] // params['num_heads']}")
    
    flops = calculate_transformer_flops(
        batch_size=params['batch_size'],
        context_length=params['context_length'],
        vocab_size=params['vocab_size'],
        num_layers=params['num_layers'],
        d_model=params['d_model'],
        num_heads=params['num_heads'],
        d_ff=params['d_ff'],
    )
    
    print("\n" + "=" * 80)
    print("FLOPs Breakdown")
    print("=" * 80)
    
    print("\nPer Transformer Layer:")
    print(f"  RMSNorm: {format_number(flops['rmsnorm_per_layer'])}")
    print(f"  Attention:")
    print(f"    - Down projection (QKV): {format_number(flops['attn_down_proj_per_layer'])}")
    print(f"    - QK^T: {format_number(flops['attn_qk_per_layer'])}")
    print(f"    - Weighted V: {format_number(flops['attn_weighted_v_per_layer'])}")
    print(f"    - Up projection: {format_number(flops['attn_up_proj_per_layer'])}")
    print(f"    - Total Attention: {format_number(flops['attention_per_layer'])}")
    print(f"  FFN (SwiGLU):")
    print(f"    - xW_1^T: {format_number(flops['ffn_w1_per_layer'])}")
    print(f"    - xW_3^T: {format_number(flops['ffn_w3_per_layer'])}")
    print(f"    - result*W_2^T: {format_number(flops['ffn_w2_per_layer'])}")
    print(f"    - Total FFN: {format_number(flops['ffn_per_layer'])}")
    print(f"  Total per layer: {format_number(flops['transformer_block_per_layer'])}")
    
    print(f"\nAll Transformer Blocks ({params['num_layers']} layers):")
    print(f"  {format_number(flops['all_transformer_blocks'])}")
    
    print(f"\nFinal Linear Layer:")
    print(f"  {format_number(flops['final_linear'])}")
    
    print()
    print(f"TOTAL FLOPs: {format_number(flops['total'])}")
    print()

if __name__ == "__main__":
    main()
