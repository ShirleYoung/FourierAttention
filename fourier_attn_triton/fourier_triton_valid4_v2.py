
import torch
import torch.nn as nn
import triton
import triton.language as tl

import numpy as np

import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0")  # triton.runtime.driver.active.get_active_torch_device()

from flash_attn.flash_attn_interface import flash_attn_func

torch.manual_seed(20)


################################################
######### Translated Fourier Transform #########
################################################
@triton.jit
def mat_fourier_kernel(
        a_ptr, c_ptr,
        B, M, N, K, T, K_start, 
        stride_ab, stride_ak, stride_am, 
        stride_cb, stride_cn, stride_cm, 
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    pid_b = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + stride_ab * pid_b + (offs_k[:, None] * stride_ak + offs_am[None, :] * stride_am)

    # -----------------------------------------------------------
    k_range = offs_k.to(tl.float32)
    offs_bn = offs_bn.to(tl.float32)
    c_cos = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    c_sin = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        a = tl.load(a_ptrs, mask=k_range[:, None] < K, other=0.0).to(tl.float32)

        F = (2. * 3.14159265359 * (k_range[None, :] + K_start) * offs_bn[:, None] / T).to(tl.float32)
        Fc = tl.cos(F) * (k_range[None, :] < K)
        Fs = tl.sin(F) * (k_range[None, :] < K)

        c_cos = tl.dot(Fc, a, c_cos)
        c_sin = tl.dot(Fs, a, c_sin)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        k_range = k_range + BLOCK_SIZE_K 

    # -----------------------------------------------------------
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cb * pid_b + stride_cn * offs_cn[:, None] + stride_cm * offs_cm[None, :]
    c_mask = (offs_cm[None, :] < M) & (offs_cn[:, None] < N)
    tl.store(c_ptrs, c_cos, mask=c_mask)

    offs_cn = N + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cb * pid_b + stride_cn * offs_cn[:, None] + stride_cm * offs_cm[None, :]
    c_mask = (offs_cm[None, :] < M) & (offs_cn[:, None] < 2 * N)
    tl.store(c_ptrs, c_sin, mask=c_mask)


def mat_fourier_triton(a, N, T, K_start=0):
    assert a.is_contiguous(), "Matrix A must be contiguous"
    B, K, M = a.shape
    c = torch.empty((B, 2 * N, M), device=a.device, dtype=torch.float32)
    grid = lambda META: (B, triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    if K > 32:
        mat_fourier_kernel[grid](
            a, c,  #
            B, M, N, K, T, K_start, #
            a.stride(0), a.stride(1), a.stride(2),  #
            c.stride(0), c.stride(1), c.stride(2),  #
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, 
        )
    else: 
        mat_fourier_kernel[grid](
            a, c,  #
            B, M, N, K, T, K_start, #
            a.stride(0), a.stride(1), a.stride(2),  #
            c.stride(0), c.stride(1), c.stride(2),  #
            BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=16, GROUP_SIZE_M=8, 
        ) #newly added
    return c #compressed, no batch size


@triton.jit
def mat_fourier_inv_kernel(
        c_ptr, a_ptr, 
        M, K, N, T, 
        stride_cm, stride_cn, 
        stride_am, stride_ak, 
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """
    BLOCK_SIZE_N is for K, BLOCK_SIZE_K is for N
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    offs_cm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bk = (pid_k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % K
    offs_n = tl.arange(0, BLOCK_SIZE_K)
    cc_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    cs_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + (offs_n[None, :] + N) * stride_cn)
    # b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    n_range = offs_n.to(tl.float32)
    offs_bk = offs_bk.to(tl.float32)
    a = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for n in range(0, tl.cdiv(N, BLOCK_SIZE_K)):

        cc = tl.load(cc_ptrs, mask=offs_n[None, :] < N - n * BLOCK_SIZE_K, other=0.0)
        cs = tl.load(cs_ptrs, mask=offs_n[None, :] < 2 * N - n * BLOCK_SIZE_K, other=0.0)

        F = 2. * 3.14159265359 * n_range[:, None] * offs_bk[None, :] / T
        Fc = tl.cos(F) * (n_range[:, None] < N)
        Fs = tl.sin(F) * (n_range[:, None] < N)

        a = tl.dot(cc, Fc, a)
        a = tl.dot(cs, Fs, a)

        cc_ptrs += BLOCK_SIZE_K * stride_cn
        cs_ptrs += BLOCK_SIZE_K * stride_cn
        n_range = n_range + BLOCK_SIZE_K

    a = a / N 

    # -----------------------------------------------------------
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ak = pid_k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + stride_am * offs_am[:, None] + stride_ak * offs_ak[None, :]
    a_mask = (offs_am[:, None] < M) & (offs_ak[None, :] < K)
    tl.store(a_ptrs, a, mask=a_mask)


def mat_fourier_inv_triton(c, K, T):
    assert c.is_contiguous(), "Matrix C must be contiguous"
    M, N = c.shape
    N = N // 2
    a = torch.empty((M, K), device=c.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_N']), )
    mat_fourier_inv_kernel[grid](
        c, a,  #
        M, K, N, T, #
        c.stride(0), c.stride(1),  #
        a.stride(0), a.stride(1),  #
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32, GROUP_SIZE_M=8
    )
    return a


################################################
######## Avg & Std' for Inversed Signal ########
################################################
@triton.jit
def mat_fourier_avg_std_triton_kernel(
    # Pointers
    c_ptr, sum_ptr, max_ptr, min_ptr,
    # Dimensions
    M, K, N, T,
    # Strides
    stride_cb, stride_cn, stride_cm,
    stride_sum_b, stride_sum_k, stride_sum_m,
    stride_max_b, stride_max_k, stride_max_m,
    stride_min_b, stride_min_k, stride_min_m,
    # Constants
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, # Renamed for clarity: size of blocks in the K dimension
    BLOCK_SIZE_N: tl.constexpr, # Renamed for clarity: size of blocks in the N dimension
    GROUP_SIZE_M: tl.constexpr,
    INFINITY: tl.constexpr,
):
    """
    This kernel computes partial sums, maximums, and minimums for sub-blocks of the K dimension.
    It uses a grouped scheduling strategy to increase parallelism and data locality.
    The final reduction is performed on the host.
    """
    # --- 1. Calculate Program IDs and Grouping ---
    pid_b = tl.program_id(axis=0)
    pid_flat = tl.program_id(axis=1)

    # Deconstruct the flat PID to get M and K block indices
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pids_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid_flat // num_pids_in_group
    
    # Calculate the starting M block index for this group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m_actual = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # Calculate the specific M and K block indices for this instance
    pid_m = first_pid_m + (pid_flat % group_size_m_actual)
    pid_k = (pid_flat % num_pids_in_group) // group_size_m_actual

    # --- Compute Pointers and Offsets ---
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = offs_m < M
    
    # Offsets for the K dimension block
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    k_mask = offs_k < K
    
    # --- 3. Reconstruct Signal and Find Partials ---
    # Accumulator for the reconstructed signal `a`. Shape: (BLOCK_SIZE_K, BLOCK_SIZE_M)
    a = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), dtype=tl.float32)

    # Loop over the N dimension in blocks (inner reduction loop)
    for n_start in range(0, N, BLOCK_SIZE_N):
        offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = offs_n < N

        # Pointers to coefficient blocks
        c_cos_ptrs = c_ptr + pid_b * stride_cb + offs_n[:, None] * stride_cn + offs_m[None, :] * stride_cm
        c_sin_ptrs = c_ptr + pid_b * stride_cb + (offs_n[:, None] + N) * stride_cn + offs_m[None, :] * stride_cm
        
        # Load coefficient blocks
        c_cos = tl.load(c_cos_ptrs, mask=n_mask[:, None] & m_mask[None, :], other=0.0)
        c_sin = tl.load(c_sin_ptrs, mask=n_mask[:, None] & m_mask[None, :], other=0.0)

        # Reconstruct: element-wise multiply and reduce
        F_arg = (2. * np.pi / T) * offs_k[:, None] * offs_n[None, :]
        Fc = tl.cos(F_arg)
        Fs = tl.sin(F_arg)

        # Accumulate the result of the dot product (manual implementation)
        a += tl.dot(Fc, c_cos, allow_tf32=True)
        a += tl.dot(Fs, c_sin, allow_tf32=True)

    # Correct FIX: Normalize by N *after* summing over the full N dimension
    a /= N
    
    # --- 4. Calculate and Store Partial Results ---
    # Correct FIX: Calculate partial sum, not partial average. Host will divide by K.
    partial_sum = tl.sum(tl.where(k_mask[:, None], a, 0.0), axis=0)
    
    # Correct FIX: Use safe masking for min/max
    partial_max = tl.max(tl.where(k_mask[:, None], a, -INFINITY), axis=0)
    partial_min = tl.min(tl.where(k_mask[:, None], a, INFINITY), axis=0)

    # Pointers to output tensors for this specific K-block
    sum_ptrs = sum_ptr + pid_b * stride_sum_b + pid_k * stride_sum_k + offs_m * stride_sum_m
    max_ptrs = max_ptr + pid_b * stride_max_b + pid_k * stride_max_k + offs_m * stride_max_m
    min_ptrs = min_ptr + pid_b * stride_min_b + pid_k * stride_min_k + offs_m * stride_min_m
    
    tl.store(sum_ptrs, partial_sum, mask=m_mask)
    tl.store(max_ptrs, partial_max, mask=m_mask)
    tl.store(min_ptrs, partial_min, mask=m_mask)


def mat_fourier_avg_std_triton(c, K, T):
    assert c.is_contiguous(), "Matrix C must be contiguous"
    B, N_2, M = c.shape
    N = N_2 // 2

    # --- Kernel Configuration ---
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = 256 # Block size for the K dimension
    BLOCK_SIZE_N = 16  # Block size for the N dimension (inner reduction)
    GROUP_SIZE_M = 8   # Group M-blocks for better scheduling

    # --- Host-side Setup ---
    num_k_blocks = triton.cdiv(K, BLOCK_SIZE_K)
    partial_sum = torch.empty((B, num_k_blocks, M), device=c.device, dtype=torch.float32)
    partial_max = torch.empty((B, num_k_blocks, M), device=c.device, dtype=torch.float32)
    partial_min = torch.empty((B, num_k_blocks, M), device=c.device, dtype=torch.float32)

    num_m_blocks = triton.cdiv(M, BLOCK_SIZE_M)
    grid = (B, num_m_blocks * num_k_blocks)

    # --- Launch Kernel ---
    mat_fourier_avg_std_triton_kernel[grid](
        c, partial_sum, partial_max, partial_min,
        M, K, N, T,
        c.stride(0), c.stride(1), c.stride(2),
        partial_sum.stride(0), partial_sum.stride(1), partial_sum.stride(2),
        partial_max.stride(0), partial_max.stride(1), partial_max.stride(2),
        partial_min.stride(0), partial_min.stride(1), partial_min.stride(2),
        BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, GROUP_SIZE_M,
        INFINITY=float('inf'),
        num_warps=16,
        num_stages=3
    )

    total_sum = partial_sum.sum(dim=1)
    final_max = partial_max.max(dim=1)[0]
    final_min = partial_min.min(dim=1)[0]
    
    avg: torch.Tensor = total_sum / K
    
    std = torch.maximum(final_max - avg, avg - final_min)
    return avg.to(torch.float16), std.to(torch.float16)


################################################
######## Avg & Std' for Original Signal ########
################################################
@triton.autotune(
    configs=[
        # Larger block sizes for K
        triton.Config({'BLOCK_SIZE_K': 2048, 'BLOCK_SIZE_M': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE_K': 2048, 'BLOCK_SIZE_M': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_K': 4096, 'BLOCK_SIZE_M': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE_K': 4096, 'BLOCK_SIZE_M': 4}, num_warps=8),
    ],
    key=['K', 'M'],
)
@triton.jit
def _mean_max_dev_one_pass_kernel(
    A_ptr,      # Pointer to input tensor
    Avg_ptr,    # Pointer to output mean tensor
    Max_dev_ptr,# Pointer to output max deviation tensor
    stride_ab,  # Stride for batch dimension of A
    stride_ak,  # Stride for K dimension of A
    stride_am,  # Stride for M dimension of A
    stride_avgb,# Stride for batch dimension of Avg
    stride_avgm,# Stride for M dimension of Avg
    stride_maxb,# Stride for batch dimension of Max_dev
    stride_maxm,# Stride for M dimension of Max_dev
    B: tl.int32, # Batch size
    K: tl.int32, # K dimension size
    M: tl.int32, # M dimension size
    BLOCK_SIZE_K: tl.constexpr, # Block size for K dimension, tuned by autotuner
    BLOCK_SIZE_M: tl.constexpr, # Block size for M dimension, tuned by autotuner
):
    """
    Triton kernel to compute mean and max absolute deviation in a true single pass.
    It processes a 2D tile (BLOCK_SIZE_K x BLOCK_SIZE_M) at a time.
    """
    # 1. Get program IDs
    pid_b = tl.program_id(0)
    pid_m_block = tl.program_id(1) # This is now a block index

    # =============================================
    #  SETUP: Pointers and accumulators
    # =============================================
    # Pointers to the first element of the input row
    a_row_start_ptr = A_ptr + pid_b * stride_ab

    # Column indices for this program instance to handle
    m_offsets = pid_m_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < M

    # Vector accumulators for the M-dimension block
    sum_accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    min_val_accumulator = tl.full((BLOCK_SIZE_M,), float('inf'), dtype=tl.float32)
    max_val_accumulator = tl.full((BLOCK_SIZE_M,), float('-inf'), dtype=tl.float32)

    # =============================================
    #  SINGLE PASS: Calculate sum, min, and max
    # =============================================
    for k_offset in range(0, K, BLOCK_SIZE_K):
        # Row indices for the current tile
        k_offsets = k_offset + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K

        # Create pointers for the 2D tile
        # Ptrs shape: [BLOCK_SIZE_K, BLOCK_SIZE_M]
        a_ptrs = a_row_start_ptr + \
                 k_offsets[:, None] * stride_ak + \
                 m_offsets[None, :] * stride_am
        
        # Combined mask for the 2D tile
        tile_mask = k_mask[:, None] & m_mask[None, :]

        # Load the 2D tile of data
        data_tile = tl.load(a_ptrs, mask=tile_mask, other=0.0).to(tl.float32)

        # Update sum accumulator (reduce along K dimension)
        sum_accumulator += tl.sum(data_tile, axis=0)

        # Update min and max accumulators
        # Replace masked-out values with inf/-inf to not affect reduction
        current_min = tl.min(tl.where(tile_mask, data_tile, float('inf')), axis=0)
        min_val_accumulator = tl.minimum(min_val_accumulator, current_min)

        current_max = tl.max(tl.where(tile_mask, data_tile, float('-inf')), axis=0)
        max_val_accumulator = tl.maximum(max_val_accumulator, current_max)
        
    # =============================================
    #  Post-Pass: Final Calculations
    # =============================================
    avg_val = sum_accumulator / K
    dev1 = max_val_accumulator - avg_val
    dev2 = avg_val - min_val_accumulator
    max_dev_val = tl.maximum(dev1, dev2)

    # =============================================
    #  Write final results
    # =============================================
    avg_out_ptrs = Avg_ptr + pid_b * stride_avgb + m_offsets * stride_avgm
    max_dev_out_ptrs = Max_dev_ptr + pid_b * stride_maxb + m_offsets * stride_maxm

    # Store the block of results, masking for M dimension
    tl.store(avg_out_ptrs, avg_val, mask=m_mask)
    tl.store(max_dev_out_ptrs, max_dev_val, mask=m_mask)


def mat_avg_std_triton(a: torch.Tensor):
    """
    Python wrapper for the fused, tiled, one-pass Triton kernel.
    """
    # Input validation
    assert a.is_cuda and a.is_contiguous(), "Input tensor must be a contiguous CUDA tensor"
    assert a.dim() == 3, "Input tensor must be 3-dimensional"

    B, K, M = a.shape

    # Create output tensors
    avg = torch.empty((B, M), device=a.device, dtype=torch.float32)
    max_dev = torch.empty((B, M), device=a.device, dtype=torch.float32)

    # The grid for the kernel launch.
    # We now have fewer programs in the M dimension as each handles a block.
    grid = lambda META: (B, triton.cdiv(M, META['BLOCK_SIZE_M']))
    
    # Launch the kernel
    _mean_max_dev_one_pass_kernel[grid](
        a, avg, max_dev,
        a.stride(0), a.stride(1), a.stride(2),
        avg.stride(0), avg.stride(1),
        max_dev.stride(0), max_dev.stride(1),
        B, K, M
    )
    
    # Cast back to original dtype if necessary
    return avg.to(a.dtype), max_dev.to(a.dtype)


@triton.heuristics(
    {
        "num_warps": lambda args: 4,  # default 4
        "num_stages": lambda args: 1,  # default 3 for faster forward pass
    }
)
@triton.jit
def fourier_attn_kernel(
    q_ptr, k_il_ptr, v_il_ptr, k_mc_ptr, v_mc_ptr, k_mn_ptr, v_mn_ptr, o_up_ptr, o_dn_ptr, qk_max_ptr, 
    avg_k_u_ptr, avg_k_d_ptr, scale_k_ptr, 
    avg_v_u_ptr, avg_v_d_ptr, scale_v_ptr, 
    kc_list_ptr, kc_h_list_ptr, kn_list_ptr, kn_h_list_ptr, 
    vc_list_ptr, vc_h_list_ptr, vn_list_ptr, vn_h_list_ptr, 
    B: tl.constexpr, H: tl.constexpr, num_rep, N_il: tl.constexpr, N_m: tl.constexpr, 
    N: tl.constexpr, D: tl.constexpr, T_scale, SM_scale, 
    stride_qb, stride_qn, stride_qd, 
    stride_kb_il, stride_kn_il, stride_kd_il,
    stride_vb_il, stride_vn_il, stride_vd_il,
    stride_kb_mc, stride_kn_mc, stride_kd_mc,
    stride_vb_mc, stride_vn_mc, stride_vd_mc,
    stride_kb_mn, stride_kn_mn, stride_kd_mn,
    stride_vb_mn, stride_vn_mn, stride_vd_mn,
    stride_avg_kb_u, stride_avg_kd_u, 
    stride_avg_kb_d, stride_avg_kd_d, 
    stride_scale_kb, stride_scale_kd, 
    stride_avg_vb_u, stride_avg_vd_u, 
    stride_avg_vb_d, stride_avg_vd_d, 
    stride_scale_vb, stride_scale_vd, 
    stride_kc_list, stride_kc_h_list, 
    stride_kn_list, stride_kn_h_list, 
    stride_vc_list, stride_vc_h_list, 
    stride_vn_list, stride_vn_h_list, 
    stride_ob_up, stride_oh_up, stride_on_up, stride_ok_up, stride_od_up,
    stride_ob_dn, stride_oh_dn, stride_on_dn, stride_ok_dn, 
    stride_mb_dn, stride_mh_dn, stride_mn_dn, stride_mk_dn, num_block_il, 
    BLOCK_SIZE_IL_O: tl.constexpr, BLOCK_SIZE_IL: tl.constexpr, 
    BLOCK_SIZE_M_O: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_h_q = tl.program_id(axis=1)
    pid_h_kv = tl.program_id(axis=1) // num_rep
    pid_n_kv = tl.program_id(axis=2)

    offs_q = tl.arange(0, 16)
    offs_d_q = tl.arange(0, D) + pid_h_q * D
    offs_d_kv = tl.arange(0, D) + pid_h_kv * D
    offs_d_ = tl.arange(0, D)

    mask_q = (offs_q[:, None] < 1)

    if pid_n_kv < num_block_il:
 
        q_ptrs = q_ptr + stride_qb * pid_m + stride_qn * offs_q[:, None] + stride_qd * offs_d_q[None, :]
        q = tl.load(q_ptrs, mask=mask_q, other=0.0).to(tl.float32)
        o_up = tl.zeros((16, D), dtype=tl.float32)
        o_dn = tl.zeros((16, ), dtype=tl.float32)
        qk_max = o_dn - 1e6 
         
        offs_k = tl.arange(0, BLOCK_SIZE_IL) + pid_n_kv * BLOCK_SIZE_IL_O
        upper_l = min((pid_n_kv + 1) * BLOCK_SIZE_IL_O, N_il)

        for _ in range(0, tl.cdiv(BLOCK_SIZE_IL_O, BLOCK_SIZE_IL)):

            kv_ptrs = k_il_ptr + stride_kb_il * pid_m + stride_kn_il * offs_k[:, None] + stride_kd_il * offs_d_kv[None, :]
            kv = tl.load(kv_ptrs, mask=(offs_k[:, None] < upper_l), other=0.0).to(tl.float32)
            qk_il = tl.dot(q, kv.trans()) * SM_scale
            qk_max_new = tl.maximum(qk_max, tl.max(qk_il, axis=-1))
            qk_il = tl.exp2(qk_il - qk_max_new[:, None]) * (offs_k[None, :] < upper_l)

            kv_ptrs = v_il_ptr + stride_vb_il * pid_m + stride_vn_il * offs_k[:, None] + stride_vd_il * offs_d_kv[None, :]
            kv = tl.load(kv_ptrs, mask=(offs_k[:, None] < upper_l), other=0.0).to(tl.float32)
            o_up = o_up * tl.exp2(qk_max - qk_max_new)[:, None] + tl.dot(qk_il, kv)  #  
            o_dn = o_dn * tl.exp2(qk_max - qk_max_new) + tl.sum(qk_il, axis=-1)  # 
            qk_max = qk_max_new

            offs_k += BLOCK_SIZE_IL

        o_up_ptrs = o_up_ptr + stride_ob_up * pid_m + stride_oh_up * pid_h_q + \
            stride_on_up * offs_q[:, None] + stride_ok_up * pid_n_kv + stride_od_up * offs_d_[None, :]
        tl.store(o_up_ptrs, o_up, mask=(offs_q[:, None] < 1) & (offs_d_[None, :] < D))
        o_dn_ptrs = o_dn_ptr + stride_ob_dn * pid_m + stride_oh_dn * pid_h_q + \
            stride_on_dn * offs_q + stride_ok_dn * pid_n_kv
        tl.store(o_dn_ptrs, o_dn, mask=(offs_q < 1))
        qk_max_ptrs = qk_max_ptr + stride_mb_dn * pid_m + stride_mh_dn * pid_h_q + \
            stride_mn_dn * offs_q + stride_mk_dn * pid_n_kv
        tl.store(qk_max_ptrs, qk_max, mask=(offs_q < 1))
 
    else:
    
        l = pid_n_kv - num_block_il
        offs_l = tl.arange(0, BLOCK_SIZE_M) + l * BLOCK_SIZE_M_O
        upper_l = min((l + 1) * BLOCK_SIZE_M_O, N_m)
        l_range = offs_l.to(tl.float32)

        kc_start = tl.load(kc_h_list_ptr + pid_h_kv * stride_kc_h_list).to(tl.int16)
        kc_len = tl.load(kc_h_list_ptr + (pid_h_kv + 1) * stride_kc_h_list).to(tl.int16) - kc_start
        kn_start = tl.load(kn_h_list_ptr + pid_h_kv * stride_kn_h_list).to(tl.int16)
        kn_len = tl.load(kn_h_list_ptr + (pid_h_kv + 1) * stride_kn_h_list).to(tl.int16) - kn_start
        vc_start = tl.load(vc_h_list_ptr + pid_h_kv * stride_vc_h_list).to(tl.int16)
        vc_len = tl.load(vc_h_list_ptr + (pid_h_kv + 1) * stride_vc_h_list).to(tl.int16) - vc_start
        vn_start = tl.load(vn_h_list_ptr + pid_h_kv * stride_vn_h_list).to(tl.int16)
        vn_len = tl.load(vn_h_list_ptr + (pid_h_kv + 1) * stride_vn_h_list).to(tl.int16) - vn_start

        kc_range = tl.load(kc_list_ptr + (kc_start + offs_d_) * stride_kc_list, mask=(offs_d_ < kc_len), other=kc_start + kc_len).to(tl.int16)  # * (offs_d_ < kc_len)
        kc_range = kc_range - pid_h_kv * D + pid_h_q * D
        kn_range = tl.load(kn_list_ptr + (kn_start + offs_d_) * stride_kn_list, mask=(offs_d_ < kn_len), other=kn_start + kn_len).to(tl.int16)  # * (offs_d_ < kn_len)
        kn_range = kn_range - pid_h_kv * D + pid_h_q * D
        vc_range = tl.load(vc_list_ptr + (vc_start + offs_d_) * stride_vc_list, mask=(offs_d_ < vc_len), other=vc_start + vc_len).to(tl.int16)  # * (offs_d_ < vc_len)
        vc_range = vc_range - pid_h_kv * D + pid_h_q * D
        vn_range = tl.load(vn_list_ptr + (vn_start + offs_d_) * stride_vn_list, mask=(offs_d_ < vn_len), other=vn_start + vn_len).to(tl.int16)  # * (offs_d_ < vn_len)
        vn_range = vn_range - pid_h_kv * D + pid_h_q * D

        q_ptrs = q_ptr + stride_qb * pid_m + stride_qn * offs_q[:, None] + stride_qd * kc_range[None, :]
        q_c = tl.load(q_ptrs, mask=(offs_q[:, None] < 1) & (offs_d_[None, :] < kc_len), other=0.0).to(tl.float32)
        q_ptrs = q_ptr + stride_qb * pid_m + stride_qn * offs_q[:, None] + stride_qd * kn_range[None, :]
        q_n = tl.load(q_ptrs, mask=(offs_q[:, None] < 1) & (offs_d_[None, :] < kn_len), other=0.0).to(tl.float32)

        o_up_c = tl.zeros((16, D), dtype=tl.float32)
        o_up_n = tl.zeros((16, D), dtype=tl.float32)
        o_dn = tl.zeros((16, ), dtype=tl.float32)   
        qk_max = o_dn - 1e6 
    
        avg_k_u_ptrs = avg_k_u_ptr + stride_avg_kb_u * pid_m + stride_avg_kd_u * (offs_d_ + kc_start)
        avg_k_u = tl.load(avg_k_u_ptrs, mask=(offs_d_ < kc_len), other=0.0).to(tl.float32)
        avg_k_d_ptrs = avg_k_d_ptr + stride_avg_kb_d * pid_m + stride_avg_kd_d * (offs_d_ + kc_start)
        avg_k_d = tl.load(avg_k_d_ptrs, mask=(offs_d_ < kc_len), other=0.0).to(tl.float32)
        scale_k_ptrs = scale_k_ptr + stride_scale_kb * pid_m + stride_scale_kd * (offs_d_ + kc_start)
        scale_k = tl.load(scale_k_ptrs, mask=(offs_d_ < kc_len), other=1.0).to(tl.float32)

        avg_v_u_ptrs = avg_v_u_ptr + stride_avg_vb_u * pid_m + stride_avg_vd_u * (offs_d_ + vc_start)
        avg_v_u = tl.load(avg_v_u_ptrs, mask=(offs_d_ < vc_len), other=0.0).to(tl.float32)
        avg_v_d_ptrs = avg_v_d_ptr + stride_avg_vb_d * pid_m + stride_avg_vd_d * (offs_d_ + vc_start)
        avg_v_d = tl.load(avg_v_d_ptrs, mask=(offs_d_ < vc_len), other=0.0).to(tl.float32)
        scale_v_ptrs = scale_v_ptr + stride_scale_vb * pid_m + stride_scale_vd * (offs_d_ + vc_start)
        scale_v = tl.load(scale_v_ptrs, mask=(offs_d_ < vc_len), other=1.0).to(tl.float32)

        for _ in range(0, tl.cdiv(BLOCK_SIZE_M_O, BLOCK_SIZE_M)):

            offs_n = tl.arange(0, BLOCK_SIZE_N)

            n_range = offs_n.to(tl.float32)
            kv_md = tl.zeros((BLOCK_SIZE_M, D), dtype=tl.float32)
            kv_mc_ptrs = k_mc_ptr + stride_kb_mc * pid_m + stride_kn_mc * offs_n[:, None] + stride_kd_mc * (offs_d_ % kc_len + kc_start)[None, :]

            for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):

                F = T_scale * n_range[None, :] * l_range[:, None]

                Fc = tl.cos(F) * (n_range[None, :] < N) * (l_range[:, None] < upper_l)
                k_c = tl.load(kv_mc_ptrs, mask=(offs_d_[None, :] < kc_len), other=0.0).to(tl.float32)
                kv_md = tl.dot(Fc, k_c, kv_md)
                Fc = tl.sin(F) * (n_range[None, :] < N) * (l_range[:, None] < upper_l)
                k_c = tl.load(kv_mc_ptrs + N * stride_kn_mc, mask=(offs_d_[None, :] < kc_len), other=0.0).to(tl.float32)
                kv_md = tl.dot(Fc, k_c, kv_md)

                kv_mc_ptrs += BLOCK_SIZE_N * stride_kn_mc
                n_range += BLOCK_SIZE_N

            kv_md = (kv_md / N - avg_k_d[None, :]) * scale_k[None, :] + avg_k_u[None, :]  # 
            kv_mn_ptrs = k_mn_ptr + stride_kb_mn * pid_m + stride_kn_mn * offs_l[:, None] + stride_kd_mn * (offs_d_ % kn_len + kn_start)[None, :]
            kv_mn = tl.load(kv_mn_ptrs, mask=(l_range[:, None] < upper_l) & (offs_d_[None, :] < kn_len), other=0.0).to(tl.float32)
            
            qk_m = (tl.dot(q_c, kv_md.trans()) + tl.dot(q_n, kv_mn.trans())) * SM_scale
            # qk_max_new = tl.maximum(qk_max, tl.max(qk_m, axis=-1))
            qk_max_new = tl.maximum(qk_max, tl.max(qk_m, axis=-1))
            qk_m = tl.exp2(qk_m - qk_max_new[:, None]) * (l_range[None, :] < upper_l)
            # qk_m = tl.where(qk_m == qk_m, qk_m, 0)
            # qk_m = tl.maximum(tl.minimum(qk_m, 1e6), -1e6)

            n_range = offs_n.to(tl.float32)
            kv_md = tl.zeros((BLOCK_SIZE_M, D), dtype=tl.float32)
            kv_mc_ptrs = v_mc_ptr + stride_vb_mc * pid_m + stride_vn_mc * offs_n[:, None] + stride_vd_mc * (offs_d_ % vc_len + vc_start)[None, :]

            for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):

                F = T_scale * n_range[None, :] * l_range[:, None]

                Fc = tl.cos(F) * (n_range[None, :] < N) * (l_range[:, None] < upper_l)
                v_c = tl.load(kv_mc_ptrs, mask=(offs_d_[None, :] < vc_len), other=0.0).to(tl.float32)
                kv_md = tl.dot(Fc, v_c, kv_md)
                Fc = tl.sin(F) * (n_range[None, :] < N) * (l_range[:, None] < upper_l)
                v_c = tl.load(kv_mc_ptrs + N * stride_vn_mc, mask=(offs_d_[None, :] < vc_len), other=0.0).to(tl.float32)
                kv_md = tl.dot(Fc, v_c, kv_md)

                kv_mc_ptrs += BLOCK_SIZE_N * stride_vn_mc
                n_range += BLOCK_SIZE_N

            kv_md = (kv_md / N - avg_v_d[None, :]) * scale_v[None, :] + avg_v_u[None, :]  # 
            kv_mn_ptrs = v_mn_ptr + stride_vb_mn * pid_m + stride_vn_mn * offs_l[:, None] + stride_vd_mn * (offs_d_ % vn_len + vn_start)[None, :]
            kv_mn = tl.load(kv_mn_ptrs, mask=(l_range[:, None] < upper_l) & (offs_d_[None, :] < vn_len), other=0.0).to(tl.float32)

            o_up_c = o_up_c * tl.exp2(qk_max - qk_max_new)[:, None] + tl.dot(qk_m, kv_md)  # + 1
            o_up_n = o_up_n * tl.exp2(qk_max - qk_max_new)[:, None] + tl.dot(qk_m, kv_mn)  # + 1
            o_dn = o_dn * tl.exp2(qk_max - qk_max_new) + tl.sum(qk_m, axis=-1)  # + 1
            qk_max = qk_max_new

            offs_l += BLOCK_SIZE_M
            l_range += BLOCK_SIZE_M

        oc_up_ptrs = o_up_ptr + stride_ob_up * pid_m + stride_oh_up * pid_h_q + \
            stride_on_up * offs_q[:, None] + stride_ok_up * pid_n_kv + stride_od_up * (vc_range[None, :] - pid_h_q * D)
        tl.store(oc_up_ptrs, o_up_c, mask=(offs_q[:, None] < 1) & (offs_d_[None, :] < vc_len))
        on_up_ptrs = o_up_ptr + stride_ob_up * pid_m + stride_oh_up * pid_h_q + \
            stride_on_up * offs_q[:, None] + stride_ok_up * pid_n_kv + stride_od_up * (vn_range[None, :] - pid_h_q * D)
        tl.store(on_up_ptrs, o_up_n, mask=(offs_q[:, None] < 1) & (offs_d_[None, :] < vn_len))
        o_dn_ptrs = o_dn_ptr + stride_ob_dn * pid_m + stride_oh_dn * pid_h_q + \
            stride_on_dn * offs_q + stride_ok_dn * pid_n_kv
        tl.store(o_dn_ptrs, o_dn, mask=(offs_q < 1))
        qk_max_ptrs = qk_max_ptr + stride_mb_dn * pid_m + stride_mh_dn * pid_h_q + \
            stride_mn_dn * offs_q + stride_mk_dn * pid_n_kv
        tl.store(qk_max_ptrs, qk_max, mask=(offs_q < 1))

        # o_c_ptrs = o_ptr + stride_ob * pid_m + stride_oh * pid_h_q + \
        #     stride_on * offs_q[:, None] + stride_od * (vc_range[None, :] - pid_h_q * D)
        # tl.store(o_c_ptrs, o_up_c / o_dn[:, None], mask=(offs_q[:, None] < 1) & (offs_d_[None, :] < vc_len))
        # o_n_ptrs = o_ptr + stride_ob * pid_m + stride_oh * pid_h_q + \
        #     stride_on * offs_q[:, None] + stride_od * (vn_range[None, :] - pid_h_q * D)
        # tl.store(o_n_ptrs, o_up_n / o_dn[:, None], mask=(offs_q[:, None] < 1) & (offs_d_[None, :] < vn_len))


def triton_fourier_attn(q, k_il, v_il, k_mc, v_mc, k_mn, v_mn, 
                        avg_k_u, avg_k_d, scale_k, avg_v_u, avg_v_d, scale_v, 
                        kc_list, kc_h_list, kn_list, kn_h_list, 
                        vc_list, vc_h_list, vn_list, vn_h_list, 
                        num_head_q, num_head_kv, seq_len_m, T, sm_scale=-1):  # , print_needed=False

    batch_size, seq_len_q, feature_dim  = q.shape
    head_dim = feature_dim // num_head_q
    assert seq_len_q == 1, f"{q.shape = }"
    seq_len_il = k_il.shape[1]
    num_rep = num_head_q // num_head_kv
    N = k_mc.shape[1] // 2
    sm_scale = sm_scale if sm_scale != -1 else 1 / np.sqrt(head_dim)

    # print(f"{batch_size = }, {seq_len_q = }, {feature_dim = }, {head_dim = }, {num_rep = }", flush=True)

    T_scale = 2.0 * 3.14159265359 / T
    SM_scale = sm_scale * 1.44269504

    BLOCK_SIZE_IL_O = 256
    BLOCK_SIZE_IL = 256
    BLOCK_SIZE_M_O = 128
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32

    num_block_il = int(np.ceil(seq_len_il / BLOCK_SIZE_IL_O))
    num_block_m = int(np.ceil(seq_len_m / BLOCK_SIZE_M_O))

    o_up = torch.empty((batch_size, num_head_q, 16, num_block_il+num_block_m, head_dim), device=q.device, dtype=torch.float)
    o_dn = torch.empty((batch_size, num_head_q, 16, num_block_il+num_block_m), device=q.device, dtype=torch.float)
    qk_max = torch.empty((batch_size, num_head_q, 16, num_block_il+num_block_m), device=q.device, dtype=torch.float)

    grid = lambda META: (batch_size, num_head_q, num_block_il+num_block_m, )
    fourier_attn_kernel[grid](
        q, k_il, v_il, k_mc, v_mc, k_mn, v_mn, o_up, o_dn, qk_max, 
        avg_k_u, avg_k_d, scale_k, avg_v_u, avg_v_d, scale_v, 
        kc_list, kc_h_list, kn_list, kn_h_list, 
        vc_list, vc_h_list, vn_list, vn_h_list, 
        batch_size, num_head_q, num_rep, seq_len_il, seq_len_m, N, head_dim, T_scale, SM_scale, 
        q.stride(0), q.stride(1), q.stride(2), 
        k_il.stride(0), k_il.stride(1), k_il.stride(2),
        v_il.stride(0), v_il.stride(1), v_il.stride(2),
        k_mc.stride(0), k_mc.stride(1), k_mc.stride(2),
        v_mc.stride(0), v_mc.stride(1), v_mc.stride(2), 
        k_mn.stride(0), k_mn.stride(1), k_mn.stride(2),
        v_mn.stride(0), v_mn.stride(1), v_mn.stride(2), 
        avg_k_u.stride(0), avg_k_u.stride(1), 
        avg_k_d.stride(0), avg_k_d.stride(1), 
        scale_k.stride(0), scale_k.stride(1), 
        avg_v_u.stride(0), avg_v_u.stride(1), 
        avg_v_d.stride(0), avg_v_d.stride(1), 
        scale_v.stride(0), scale_v.stride(1), 
        kc_list.stride(0), kc_h_list.stride(0), 
        kn_list.stride(0), kn_h_list.stride(0), 
        vc_list.stride(0), vc_h_list.stride(0), 
        vn_list.stride(0), vn_h_list.stride(0), 
        o_up.stride(0), o_up.stride(1), o_up.stride(2), o_up.stride(3), o_up.stride(4), 
        o_dn.stride(0), o_dn.stride(1), o_dn.stride(2), o_dn.stride(3), 
        qk_max.stride(0), qk_max.stride(1), qk_max.stride(2), qk_max.stride(3), num_block_il, 
        BLOCK_SIZE_IL_O, BLOCK_SIZE_IL, BLOCK_SIZE_M_O, BLOCK_SIZE_M, BLOCK_SIZE_N,
    )

    # o_up = o_up.to(q)
    # o_dn = o_dn.to(q)
    # qk_max = qk_max.to(q)

    # if print_needed:
    #     print(f'{qk_max[:, 8, 0].detach().cpu().numpy().tolist() = }')
    #     print(f'{o_dn[:, 8, 0].detach().cpu().numpy().tolist() = }')
    #     print(f'{o_up.isfinite().all() = }')
    #     # print(f'{torch.max(qk_max, dim=-1).values[:, 8, 0].detach().cpu().numpy().tolist() = }')

    qk_max = torch.where(o_dn > 0, qk_max, -1e6)
    qk_max_ = torch.max(qk_max, dim=-1, keepdim=True).values  # [..., :num_block_il]
    attn_scale = qk_max - qk_max_
    attn_scale = torch.exp2(attn_scale)

    # if print_needed:
    #     print(f'{attn_scale[:, 8, 0].detach().cpu().numpy().tolist() = }')

    o_up = o_up * attn_scale[..., None]
    o_dn = o_dn * attn_scale
    
    # o_up = o_up * torch.exp2(qk_max - qk_max_)[..., None]
    # o_dn = o_dn * torch.exp2(qk_max - qk_max_)
    # o = torch.sum(o_up, dim=-2) / torch.sum(o_dn, dim=-1, keepdim=True)
    # o = (o_up_sum + 1e-18) / (o_dn_sum + 1e-18)
    # o = (torch.sum(o_up, dim=-2) + 1e-18) / (torch.sum(o_dn, dim=-1, keepdim=True) + 1e-18)

    # if print_needed:
    #     print(f'{torch.sum(o_dn, dim=-1)[:, 8, 0].detach().cpu().numpy().tolist() = }')

    o = torch.sum(o_up, dim=-2) / torch.sum(o_dn, dim=-1, keepdim=True)

    # return o[..., :1, :].half()    # Originally returned half 16
    return o[..., :1, :]   # Modified to return 32 but still has nan 32


def mat_fourier_inv_torch(c, K, T):
    N = c.shape[1] // 2
    vals = np.arange(0, K, dtype=np.float32)
    n_range = np.arange(N, dtype=np.float32)

    B = (vals[:, None] * n_range[None, :]) * 2 * np.pi / T
    B = np.concatenate([np.cos(B), np.sin(B)], axis=-1)
    B = torch.from_numpy(B).to(dtype=c.dtype, device=DEVICE)

    return torch.einsum('mnd,kn->mkd', c, B) / N


def torch_fourier_attn(q, k_il, v_il, k_mc, v_mc, k_mn, v_mn, 
                       avg_k_u, avg_k_d, scale_k, avg_v_u, avg_v_d, scale_v, 
                       kc_list, kc_h_list, kn_list, kn_h_list, 
                       vc_list, vc_h_list, vn_list, vn_h_list, 
                       num_head_q, num_head_kv, seq_len_m, T, sm_scale=-1):

    batch_size, seq_len_q, feature_dim  = q.shape
    head_dim = feature_dim // num_head_q
    assert seq_len_q == 1
    seq_len_kv = k_il.shape[1] + seq_len_m
    num_rep = num_head_q // num_head_kv
    sm_scale = sm_scale if sm_scale != -1 else 1 / np.sqrt(head_dim)

    q_ = q.reshape(batch_size, 1, num_head_q, head_dim).contiguous()
    # k_ilm = k_il.reshape(batch_size, k_il.shape[1], num_head_kv, head_dim).contiguous()
    # v_ilm = v_il.reshape(batch_size, k_il.shape[1], num_head_kv, head_dim).contiguous()
    # std_out = flash_attn_func(q_, k_ilm, v_ilm, softmax_scale=sm_scale, causal=True)  # , _, _, return_attn_probs=True

    k_md = mat_fourier_inv_torch(k_mc, K=seq_len_m, T=T)
    k_md = (k_md - avg_k_d[:, None, :]) * scale_k[:, None, :] + avg_k_u[:, None, :]
    v_md = mat_fourier_inv_torch(v_mc, K=seq_len_m, T=T)
    v_md = (v_md - avg_v_d[:, None, :]) * scale_v[:, None, :] + avg_v_u[:, None, :]

    k_m = torch.zeros((batch_size, seq_len_m, feature_dim // num_rep)).to(device=q.device, dtype=q.dtype)
    k_m[..., kc_list] = k_md.to(k_m)
    k_m[..., kn_list] = k_mn.to(k_m)
    v_m = torch.zeros((batch_size, seq_len_m, feature_dim // num_rep)).to(device=q.device, dtype=q.dtype)
    v_m[..., vc_list] = v_md.to(v_m)
    v_m[..., vn_list] = v_mn.to(v_m)

    k_ilm = torch.cat([k_il, k_m], dim=1).contiguous()
    v_ilm = torch.cat([v_il, v_m], dim=1).contiguous()

    q_ = q.reshape(batch_size, 1, num_head_q, head_dim).contiguous()
    k_ilm = k_ilm.reshape(batch_size, seq_len_kv, num_head_kv, head_dim).contiguous()
    # k_ilm = k_ilm[:, :, :, None, :].expand(batch_size, seq_len_kv, num_head_kv, num_rep, head_dim)
    # k_ilm = k_ilm.reshape(batch_size, seq_len_kv, num_head_kv * num_rep, head_dim)
    v_ilm = v_ilm.reshape(batch_size, seq_len_kv, num_head_kv, head_dim).contiguous()
    # v_ilm = v_ilm[:, :, :, None, :].expand(batch_size, seq_len_kv, num_head_kv, num_rep, head_dim)
    # v_ilm = v_ilm.reshape(batch_size, seq_len_kv, num_head_kv * num_rep, head_dim)

    std_out = flash_attn_func(q_.half(), k_ilm.half(), v_ilm.half(), softmax_scale=sm_scale, causal=True)  # , _, _, return_attn_probs=True

    return std_out.transpose(1, 2)


class MultiDimHiPPO2(nn.Module):
    """Multi-dimensional Linear time invariant x' = Ax + Bu"""
    def __init__(self, N, input_dim, method='legt', dt=1.0, T=1.0, discretization='bilinear', scale=False, c=0.0):
        super().__init__()
        self.method = method #Use Legendre measure
        self.N = N
        self.input_dim = input_dim
        self.dt = dt
        self.T = T
        self.c = c
        
        self.base_N = N // input_dim  # Number of states per feature dimension

    def forward(self, base_input, token_num, slice_input=None, fast=False):
        base_N = self.N // self.input_dim // 2 # Number of states per dimension
        if slice_input is not None:            
            return base_input + mat_fourier_triton(slice_input, base_N, int(1/self.dt), K_start=token_num)
        else:
            base_input_shifted = base_input.to(torch.complex64)
            fft_result = torch.fft.fft(base_input_shifted, n=int(1/self.dt), dim=1)
            fft_result_truncated = fft_result[:, :base_N, :]
            c_cos = fft_result_truncated.real
            c_sin = -fft_result_truncated.imag # Negate to match the kernel's +sin output

            # Concatenate along the channel dimension to get shape (B, 2 * N, M)
            return torch.cat([c_cos, c_sin], dim=1)

            # return mat_fourier_triton(base_input, base_N, int(1/self.dt))  
