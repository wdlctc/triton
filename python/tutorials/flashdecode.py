import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage1(
    Q, K, V, sm_scale, Req_to_tokens, B_req_idx, B_Seqlen,
    Mid_O, # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum, #[batch, head, seq_block_num]
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    gqa_group_size,
    BLOCK_SEQ: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)
    cur_kv_head = cur_head // gqa_group_size

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    
    block_n_size = tl.where(cur_batch_end_index - cur_batch_start_index <= 0, 0, cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1) // BLOCK_N
    
    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
    
    q = tl.load(Q + off_q)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx +  offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :]
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        att_value = tl.where(offs_n_new < cur_batch_end_index, att_value, float("-inf"))
        v = tl.load(V + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        
        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v, axis=0)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic
    
    need_store = tl.where(block_n_size == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid_o = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + offs_d
        off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block
        tl.store(Mid_O + off_mid_o, acc / sum_exp)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))
    return


@torch.no_grad()
def flash_decode_stage1(q, k, v, Req_to_tokens, B_req_idx, B_Seqlen, max_len_in_batch, mid_out, mid_out_logsumexp, block_seq):
    BLOCK_SEQ = block_seq
    BLOCK_N = 16
    assert BLOCK_SEQ % BLOCK_N == 0
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)
    batch, head_num = B_req_idx.shape[0], q.shape[1]
    grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK_SEQ))
    gqa_group_size = q.shape[1] // k.shape[1]
    
    _fwd_kernel_flash_decode_stage1[grid](
        q, k, v, sm_scale, Req_to_tokens, B_req_idx, B_Seqlen,
        mid_out,
        mid_out_logsumexp,
        Req_to_tokens.stride(0), Req_to_tokens.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logsumexp.stride(0), mid_out_logsumexp.stride(1), mid_out_logsumexp.stride(2),
        gqa_group_size,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK_N,
        num_warps=1,
        num_stages=2,
    )
    return



@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O, # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum, #[batch, head, seq_block_num]
    O, #[batch, head, head_dim]
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    stride_obs, stride_oh, stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    
    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return


@torch.no_grad()
def flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, O, block_seq):
    Lk = mid_out.shape[-1]
    assert Lk in {16, 32, 64, 128}
    batch, head_num = mid_out.shape[0], mid_out.shape[1]
    grid = (batch, head_num)
    
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen, mid_out, mid_out_logexpsum, O,
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logexpsum.stride(0), mid_out_logexpsum.stride(1), mid_out_logexpsum.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=Lk,
        num_warps=4,
        num_stages=2,
    )
    return


def token_decode_attention_flash_decoding(q, infer_state, q_head_num, head_dim, cache_k, cache_v, out=None):
    BLOCK_SEQ = 256
    batch_size = infer_state.batch_size
    max_len_in_batch = infer_state.max_len_in_batch
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from .flash_decoding_stage1 import flash_decode_stage1
    from .flash_decoding_stage2 import flash_decode_stage2

    o_tensor = torch.empty_like(q) if out is None else out
    
    if getattr(infer_state, 'mid_o', None) is None:
        infer_state.mid_o = torch.empty([batch_size, 
                                        q_head_num, 
                                        max_len_in_batch // BLOCK_SEQ + 1, 
                                        head_dim], 
                                        dtype=torch.float32, 
                                        device="cuda")
        infer_state.mid_o_logexpsum = torch.empty([batch_size, 
                                        q_head_num,
                                        max_len_in_batch // BLOCK_SEQ + 1], 
                                        dtype=torch.float32, 
                                        device="cuda")
        
    mid_o = infer_state.mid_o
    mid_o_logexpsum = infer_state.mid_o_logexpsum

    flash_decode_stage1(q.view(calcu_shape1),
                                cache_k,
                                cache_v,
                                infer_state.req_manager.req_to_token_indexs,
                                infer_state.b_req_idx,
                                infer_state.b_seq_len,
                                infer_state.max_len_in_batch,
                                mid_o,
                                mid_o_logexpsum,
                                BLOCK_SEQ)
    flash_decode_stage2(mid_o,
                        mid_o_logexpsum, 
                        infer_state.b_seq_len, 
                        o_tensor.view(calcu_shape1), 
                        BLOCK_SEQ)
    return o_tensor


torch.manual_seed(0)
sequence = 2048
size = 2048
q = torch.rand((1, 64, 1, size), device='cuda')
k = torch.rand((1, 64, sequence, size), device='cuda')
v = torch.rand((1, 64, sequence, size), device='cuda')
output_torch, attention_weights = attention(q, k, v)
output_triton = token_decode_attention_flash_decoding(q,k,v)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')