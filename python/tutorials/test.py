

import triton
import triton.language as tl
import torch
import torch.nn.functional as F
import math 

@triton.jit
def _fwd_kernel_flash_decode_stage1(
    Q, K, V, seq_len, sm_scale,
    Mid_O, Mid_O_LogExpSum,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_lbs, stride_lh,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    seq_start_block = tl.program_id(1)

    seq_start_index = seq_start_block * BLOCK_SEQ
    seq_index = seq_start_block * BLOCK_DMODEL

    offset_d = tl.arange(0, BLOCK_DMODEL)

    offset_q = offset_d

    offs_n = tl.arange(0, BLOCK_N)

    block_n_size = BLOCK_SEQ // BLOCK_N

    q = tl.load(Q + offset_q + cur_batch * stride_qbs)
    
    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    k_tile_ptr = tl.make_block_ptr(base=K + cur_batch * stride_kbs, shape=(seq_len, BLOCK_DMODEL), strides=(stride_kh, stride_kd),
                                   offsets=(seq_start_index, 0), block_shape=(BLOCK_N, BLOCK_DMODEL),
                                   order=(1, 0))
    v_tile_ptr = tl.make_block_ptr(base=V + cur_batch * stride_vbs, shape=(seq_len, BLOCK_DMODEL), strides=(stride_vh, stride_vd),
                                   offsets=(seq_start_index, 0), block_shape=(BLOCK_N, BLOCK_DMODEL),
                                   order=(1, 0))

    for start_n in range(0, block_n_size, 1):
        k = tl.load(k_tile_ptr)

        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        
        v = tl.load(v_tile_ptr)
    
        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v, axis=0)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic

    
        k_tile_ptr = tl.advance(k_tile_ptr, (BLOCK_N, 0))
        v_tile_ptr = tl.advance(v_tile_ptr, (BLOCK_N, 0))

    off_o = offset_d
    tl.store(Mid_O  + off_o + seq_index + cur_batch * stride_obs, acc / sum_exp)
    tl.store(Mid_O_LogExpSum + seq_start_block + cur_batch * stride_lbs, max_logic + tl.log(sum_exp))
        

@torch.no_grad()
def flash_decoding_stage1(q, k, v):
    BLOCK_SEQ = 256
    BLOCK_N = 16
    BLOCK_DMODEL = 2048
    assert BLOCK_SEQ % BLOCK_N == 0
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    seq_len = k.shape[1]
    sm_scale = 1.0 / (Lk ** 0.5)
    batch = q.shape[0]
    
    grid = (batch, triton.cdiv(seq_len, BLOCK_SEQ))

    O = torch.zeros((batch, seq_len // BLOCK_SEQ, BLOCK_DMODEL), device=q.device)
    LogExpSum = torch.zeros((batch, seq_len // BLOCK_SEQ), device=q.device)
    
    _fwd_kernel_flash_decode_stage1[grid](q, k, v, seq_len, sm_scale ,
                                          O, LogExpSum,
                                          q.stride(0), q.stride(1), q.stride(2),
                                          k.stride(0), k.stride(1), k.stride(2),
                                          v.stride(0), v.stride(1), v.stride(2),
                                          O.stride(0), O.stride(1), O.stride(2),
                                          LogExpSum.stride(0), LogExpSum.stride(1),
                                          BLOCK_SEQ = BLOCK_SEQ, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N = BLOCK_N,
                                          num_warps = 16)

    return O, LogExpSum

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    Mid_O, Mid_O_LogExpSum,
    O,
    stride_obs, stride_oh, stride_od,
    stride_qbs, stride_qh, stride_qd,
    stride_lbs, stride_lh,
    seq_len,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    
    cur_batch = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    block_n_size = seq_len // BLOCK_SEQ
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_d + block_seq_n * BLOCK_DMODEL + cur_batch * stride_qbs)
        tlogic = tl.load(Mid_O_LogExpSum + block_seq_n + cur_batch * stride_lbs)
        
        new_max_logic = tl.maximum(tlogic, max_logic)
        
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    tl.store(O + offs_d + cur_batch * stride_obs, acc / sum_exp)
    return

@torch.no_grad()
def flash_decoding_stage2(mid_out, mid_out_logexpsum, O):
    BLOCK_SEQ = 256
    BLOCK_DMODEL = 2048

    batch = O.shape[0]
    seq_len = BLOCK_SEQ * mid_out_logexpsum.shape[-1]

    grid = lambda meta: (batch, )
    
    _fwd_kernel_flash_decode_stage2[grid](mid_out, mid_out_logexpsum, O,
                                          O.stride(0), O.stride(1), O.stride(2),
                                          mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
                                          mid_out_logexpsum.stride(0), mid_out_logexpsum.stride(1),
                                          seq_len, 
                                          BLOCK_SEQ = BLOCK_SEQ, BLOCK_DMODEL=BLOCK_DMODEL)

    return O

def flash_decoding(q, cache_k, cache_v):
    batch_size = 1
    
    output_tensor = torch.empty_like(q)
    mid_out, mid_out_logexpsum = flash_decoding_stage1(q, cache_k, cache_v)
    output_tensor = flash_decoding_stage2(mid_out, mid_out_logexpsum, output_tensor)

    return output_tensor

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value):
        # Ensure query, key, value are in the format (batch_size, seq_len, hidden_dim)
        # For the sake of this example, we're assuming they are already correctly formatted
        
        # Step 2: Calculate the dot products of the query with all keys
        # matmul_qk shape: (batch_size, seq_len_query, seq_len_key)
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))
        
        # Step 3: Scale the dot-products
        d_k = query.size(-1)  # dimension of the keys
        scaled_attention_logits = matmul_qk / math.sqrt(d_k)

        # Step 4: Apply softmax to get the weights
        # attention_weights shape: (batch_size, seq_len_query, seq_len_key)
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # Step 5: Multiply by the values to get the final output
        # output shape: (batch_size, seq_len_query, hidden_dim)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

torch.manual_seed(0)
sequence = 2048
size = 2048
q = torch.rand((2, 1, size), device='cuda')
k = torch.rand((2, sequence, size), device='cuda')
v = torch.rand((2, sequence, size), device='cuda')
attention = ScaledDotProductAttention()
output_torch, attention_weights = attention(q, k, v)
output_triton = flash_decoding(q,k,v)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 14, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    q = torch.rand((128, 1, 2048), device='cuda')
    k = torch.rand((128, size, 2048), device='cuda')
    v = torch.rand((128, size, 2048), device='cuda')
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: attention(q, k, v), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash_decoding(q,k,v), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)