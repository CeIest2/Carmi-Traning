import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99

@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

_XX_CONFIGS = [
    # les 2 configs H100 d'origine, gardées comme candidates
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 8}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=8),
    # candidates pour Ada (MMAv2, 99 Kio smem/bloc)
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,  "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 8}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 8}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32,  "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
]

@triton.autotune(configs=_XX_CONFIGS, key=["M", "K"])
@triton.jit
def XXT_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Load A blocks for C[m,n] = A[m,:] @ A[n,:].T
    # Load A[m, k] -> shape (BM, BK)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    # Load A[n, k] -> shape (BN, BK). Transpose to get (BK, BN) for accumulation.
    # Loading (BN, BK) is coalesced because stride_c is 1 (contiguous dim is k).
    at_ptrs = A_ptr + (offs_n[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at_temp = tl.load(at_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at = tl.trans(at_temp)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)



def XXT(A: torch.Tensor, out: torch.Tensor):
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M and out.size(-1) == M
    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda META: (batch_size * triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(M, META["BLOCK_SIZE_N"]),)
    XXT_kernel[grid](
        A_ptr=A, C_ptr=out, M=M, K=K,
        a_stride_b=input_batch_stride, a_stride_r=A.stride(-2), a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride, c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
        LOWER_UPPER=1,
    )
    return out

# -----------------------------------------------------------------------------
# Triton kernel for X.T @ X (tall matrices)
# Computes C = A.T @ A where A is (M, K) and output C is (K, K)

@triton.jit
def XTX_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    """
    Compute C = A.T @ A where A is (M, K) and C is (K, K).
    This is the transpose variant of XXT for tall matrices.
    
    The output matrix C is symmetric, so we compute upper triangle and mirror.
    We iterate over blocks of M (the reduction dimension after transpose).
    """
    pid = tl.program_id(axis=0)
    # Note: Output is (K, K), so we use K for the output grid
    batch_idx, k_idx, n_idx = _pid_to_block(
        pid, K, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed (symmetry optimization)
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= k_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (k_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # For A.T @ A:
    # - A.T has shape (K, M), so A.T[k, m] = A[m, k]
    # - We load blocks from columns k_idx and n_idx of A (which are rows of A.T)
    # - We reduce over M (the shared dimension)
    offs_k = (k_idx + tl.arange(0, BLOCK_SIZE_M)) % K  # Output row indices (columns of A)
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % K  # Output col indices (columns of A)
    offs_m = tl.arange(0, BLOCK_SIZE_K)  # Reduction dimension (rows of A)

    # Pointers for loading A[:, k_idx:k_idx+BLOCK] (transposed view is A.T[k_idx:, :])
    # at_ptrs loads A.T block: A.T[offs_k, offs_m] = A[offs_m, offs_k]
    at_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    # a_ptrs loads A block for the other factor: A.T[offs_m, offs_n].T = A[offs_m, offs_n]
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_n[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of M (the reduction dimension)
    for m in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        m_remaining = M - m * BLOCK_SIZE_K
        # Load A.T[offs_k, offs_m] = A[offs_m, offs_k] -> shape (BLOCK_K, BLOCK_M)
        at = tl.load(at_ptrs, mask=offs_m[:, None] < m_remaining, other=0.0)
        # Load A[offs_m, offs_n] -> shape (BLOCK_K, BLOCK_N)
        a = tl.load(a_ptrs, mask=offs_m[:, None] < m_remaining, other=0.0)
        # C[k, n] = sum_m A.T[k, m] * A[m, n] = sum_m A[m, k] * A[m, n]
        # at.T @ a: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) = (BLOCK_M, BLOCK_N)
        accumulator = tl.dot(at.T, a, accumulator)
        at_ptrs += BLOCK_SIZE_K * a_stride_r
        a_ptrs += BLOCK_SIZE_K * a_stride_r

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_ck = k_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_ck[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_ck[:, None] < K) & (offs_cn[None, :] < K)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal (symmetry)
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_ck[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < K) & (offs_ck[None, :] < K)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def XTX(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A.T @ A
    
    For tall matrices (M > K), this is more efficient than transposing
    and using XXT because the intermediate products are smaller (K x K vs M x M).
    
    Args:
        A: Input tensor of shape (M, K) or (batch, M, K)
        out: Output tensor of shape (K, K) or (batch, K, K)
    
    Returns:
        out: The same output tensor, filled with A.T @ A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == K, f"Output matrix has incorrect shape: expected ({K}, {K}), got {tuple(out.shape[-2:])}"
    assert out.size(-1) == K, f"Output matrix has incorrect shape: expected ({K}, {K}), got {tuple(out.shape[-2:])}"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    # Hardcoded configs based on H100 autotuning
    if K == 768:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
        num_stages, num_warps = 4, 8
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 128
        num_stages, num_warps = 4, 8

    grid = (batch_size * triton.cdiv(K, BLOCK_SIZE_M) * triton.cdiv(K, BLOCK_SIZE_N),)
    XTX_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        LOWER_UPPER=1,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out


@triton.jit
def ba_plus_cAA_kernel(
    A_ptr, C_ptr,
    M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from XXT_kernel, but also loads and adds a block of A
    # Performance is slightly slower than XXT_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Coalesced loads similar to XXT_kernel
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_n[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        k_remaining = M - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at_temp = tl.load(at_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at = tl.trans(at_temp)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    # Hardcoded config based on H100 autotuning (M=768)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
    num_stages, num_warps = 4, 8

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)
    ba_plus_cAA_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        LOWER_UPPER=1,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out

# -----------------------------------------------------------------------------
# Triton kernel for MLP: relu(x @ W1.T)^2, by @andrewbriand, @jrauvola

@triton.jit
def linear_relu_square_kernel(a_ptr, b_ptr, c_ptr, aux_ptr,
                                 dequant_scale_ptr,
                                 M, N, K,
                                 stride_am, stride_bn, stride_cm,
                                 BLOCK_SIZE_M: tl.constexpr,
                                 BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr,
                                 NUM_SMS: tl.constexpr,
                                 FORWARD: tl.constexpr,
                                 USE_FP8: tl.constexpr):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        offs_m = (offs_am + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (offs_bn + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :])
        b_ptrs = b_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :])

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            k_remaining = K - ki * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            acc = tl.dot(a, b.T, acc)
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K

        if USE_FP8:
            acc *= tl.load(dequant_scale_ptr)

        c = acc.to(tl.bfloat16)

        offs_cm = offs_am + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = offs_bn + tl.arange(0, BLOCK_SIZE_N)
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :]
        aux_ptrs = aux_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :]

        if not FORWARD:
            pre = tl.load(aux_ptrs, mask=mask_c, other=0.0)
            c = 2 * c * tl.where(pre > 0, pre, 0)

        tl.store(c_ptrs, c, mask=mask_c)

        if FORWARD:
            post = tl.maximum(c, 0)
            post = post * post
            tl.store(aux_ptrs, post, mask=mask_c)


def _lrs_launch(a, b, aux, a_f8, b_f8, dequant_scale_ptr):
    """Lance linear_relu_square_kernel. Retourne (c, aux)."""
    M, K = a.shape
    N, K = b.shape
    use_fp8 = b_f8 is not None

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    FORWARD = aux is None
    if FORWARD:
        aux = torch.empty((M, N), device=a.device, dtype=a.dtype)

    a_kernel = a_f8 if use_fp8 else a
    b_kernel = b_f8 if use_fp8 else b
    # les loads pointeurs supposent la dimension intérieure contiguë
    assert a_kernel.stride(-1) == 1 and b_kernel.stride(-1) == 1
    assert c.stride(-1) == 1 and aux.stride(-1) == 1

    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count

    BLOCK_SIZE_M = LRS_BM
    BLOCK_SIZE_N = LRS_BN
    BLOCK_SIZE_K = LRS_BK_FP8 if use_fp8 else LRS_BK_BF16
    num_stages = LRS_STAGES_FWD if FORWARD else LRS_STAGES_BWD
    num_warps = LRS_WARPS

    if use_fp8:
        assert dequant_scale_ptr is not None
    else:
        dequant_scale_ptr = _get_dummy_f32(a.device)

    def grid(META):
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        ), )

    linear_relu_square_kernel[grid](
        a_kernel, b_kernel, c, aux,
        dequant_scale_ptr,
        M, N, K,
        a_kernel.stride(0), b_kernel.stride(0), c.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_SMS=NUM_SMS,
        FORWARD=FORWARD,
        USE_FP8=use_fp8,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return c, aux


_dummy_f32 = None  # lazily initialized 1-element tensor for unused pointer args

def _get_dummy_f32(device):
    global _dummy_f32
    if _dummy_f32 is None or _dummy_f32.device != device:
        _dummy_f32 = torch.zeros(1, dtype=torch.float32, device=device)
    return _dummy_f32



from typing import Optional

# Config MLP a tuner sur 4060 Ti (voir etape 3 pour le benchmark)
LRS_BM, LRS_BN = 64, 128
LRS_BK_BF16, LRS_BK_FP8 = 64, 128
LRS_STAGES_FWD, LRS_STAGES_BWD, LRS_WARPS = 4, 3, 8


# Opérateur forward : crée aux, retourne (pre, post)
@torch.library.custom_op("nanogpt::lrs_fwd", mutates_args=())
def _lrs_fwd_op(a: torch.Tensor, b: torch.Tensor,
                a_f8: Optional[torch.Tensor] = None,
                b_f8: Optional[torch.Tensor] = None,
                dequant_scale_ptr: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    return _lrs_launch(a, b, None, a_f8, b_f8, dequant_scale_ptr)

@_lrs_fwd_op.register_fake
def _(a, b, a_f8=None, b_f8=None, dequant_scale_ptr=None):
    M, K = a.shape
    N, K = b.shape
    return a.new_empty((M, N)), a.new_empty((M, N))

# Opérateur backward : lit aux (pre), retourne uniquement c
@torch.library.custom_op("nanogpt::lrs_bwd", mutates_args=())
def _lrs_bwd_op(a: torch.Tensor, b: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
    c, _ = _lrs_launch(a, b, aux, None, None, None)
    return c

@_lrs_bwd_op.register_fake
def _(a, b, aux):
    M, K = a.shape
    N, K = b.shape
    return a.new_empty((M, N))


class FusedLinearReLUSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2, W1_f8=None, dequant_scale=None, x_f8=None):
        x_flat = x.view((-1, x.shape[-1]))
        if W1_f8 is not None:
            assert x_f8 is not None and dequant_scale is not None
            x_f8 = x_f8.view((-1, x_f8.shape[-1]))
            pre, post = torch.ops.nanogpt.lrs_fwd(
                x_flat, W1, a_f8=x_f8, b_f8=W1_f8, dequant_scale_ptr=dequant_scale)
        else:
            pre, post = torch.ops.nanogpt.lrs_fwd(x_flat, W1)
        x3 = post @ W2
        ctx.save_for_backward(x, W1, W2, pre, post)
        return x3.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, pre, post = ctx.saved_tensors
        dW2 = post.T @ grad_output
        dpre = torch.ops.nanogpt.lrs_bwd(
            grad_output.view((-1, grad_output.shape[-1])), W2, pre)
        dW1 = dpre.T @ x
        dx = dpre @ W1
        return dx.view(x.shape), dW1, dW2, None, None, None


# -----------------------------------------------------------------------------
# Tiled transpose copy kernel: dst (N, M) = src (M, N).T
# Uses coalesced reads from src and coalesced writes to dst via tl.trans().
# Replaces PyTorch's elementwise copy_ which uses a naive 75k-block kernel
# with non-coalesced writes, saturating all SMs and blocking NCCL.

@triton.jit
def _transpose_copy_kernel(
    src_ptr, dst_ptr,
    M, N,
    src_stride_m, src_stride_n,
    dst_stride_0, dst_stride_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Coalesced read from src (M, N)
    tile = tl.load(
        src_ptr + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask, other=0.0,
    )

    # Coalesced write to dst (N, M): dst[n, m] = src[m, n]
    mask_T = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(
        dst_ptr + offs_n[:, None] * dst_stride_0 + offs_m[None, :] * dst_stride_1,
        tl.trans(tile), mask=mask_T,
    )


def transpose_copy(src: torch.Tensor, dst: torch.Tensor):
    """Tiled transpose copy: dst = src.T where src is (M, N) and dst is (N, M).

    Uses a 64x128 tiled Triton kernel with coalesced reads AND writes,
    achieving near memory-bandwidth-limited performance.
    """
    assert src.ndim == 2 and dst.ndim == 2
    M, N = src.shape
    assert dst.shape == (N, M), f"Expected dst shape ({N}, {M}), got {dst.shape}"

    BLOCK_M, BLOCK_N = 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _transpose_copy_kernel[grid](
        src, dst,
        M, N,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )


# -----------------------------------------------------------------------------
# Tiled transpose-add kernel: dst (M, N) += src (N, M).T
# Same tiling strategy as transpose_copy but with a fused read-add-write.
# Replaces PyTorch's .add_(src.T) which uses the same 75k-block elementwise
# kernel with non-coalesced reads from the transposed operand.

@triton.jit
def _transpose_add_kernel(
    src_ptr, dst_ptr,
    M, N,
    src_stride_m, src_stride_n,
    dst_stride_0, dst_stride_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Coalesced read from src (M, N)
    src_tile = tl.load(
        src_ptr + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask, other=0.0,
    )

    # Coalesced read-add-write on dst (N, M): dst[n, m] += src[m, n]
    mask_T = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    dst_ptrs = dst_ptr + offs_n[:, None] * dst_stride_0 + offs_m[None, :] * dst_stride_1
    dst_tile = tl.load(dst_ptrs, mask=mask_T, other=0.0)
    tl.store(dst_ptrs, dst_tile + tl.trans(src_tile), mask=mask_T)


def transpose_add(src: torch.Tensor, dst: torch.Tensor):
    """Tiled transpose-add: dst += src.T where src is (M, N) and dst is (N, M).

    Uses a 32x32 tiled Triton kernel with coalesced access on both src and dst,
    replacing PyTorch's .add_(src.T) which has non-coalesced reads from the
    transposed operand.
    """
    assert src.ndim == 2 and dst.ndim == 2
    M, N = src.shape
    assert dst.shape == (N, M), f"Expected dst shape ({N}, {M}), got {dst.shape}"

    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _transpose_add_kernel[grid](
        src, dst,
        M, N,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

# 512 threads -> NUM_LOADS = ceil(50304 / (512*8)) = 13 (au lieu de 25 pour 256)
# 1 bloc/SM sur SM89 (smem 100 608 o > 100 Kio/2), 16 warps résidents au lieu de 8
CE_KERNEL_BLOCK_SIZE = 512
CE_KERNEL_VOCAB_SIZE = 50304

CE_KERNEL_DECLS = f"""
constexpr int VOCAB_SIZE = {CE_KERNEL_VOCAB_SIZE};
constexpr int BLOCK_SIZE = {CE_KERNEL_BLOCK_SIZE};
"""

CE_KERNEL_SOURCE = """
#include <cuda_bf16.h>
#include <math_constants.h>

#define __nv_fp8_e5m2 char
#define uint16_t unsigned short
#define uint8_t unsigned char
#define int64_t long long

__device__ __forceinline__ __nv_fp8_e5m2 f32_to_fp8_e5m2(float x) {
    uint16_t packed;
    asm volatile(
        "cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
        : "=h"(packed)
        : "f"(x), "f"(0.0f)
    );
    __nv_fp8_e5m2 result;
    *reinterpret_cast<uint8_t*>(&result) = (packed & (0xFF << 8)) >> 8;
    return result;
}

struct __align__(16) __nv_bfloat168 {
    __nv_bfloat16 data[8];
    __device__ __nv_bfloat16& operator[](int i) { return data[i]; }
    __device__ const __nv_bfloat16& operator[](int i) const { return data[i]; }
};

struct __align__(8) __nv_fp8_e5m28 {
    __nv_fp8_e5m2 data[8];
    __device__ __nv_fp8_e5m2& operator[](int i) { return data[i]; }
    __device__ const __nv_fp8_e5m2& operator[](int i) const { return data[i]; }
};

template<typename T> __device__ constexpr T CEIL_DIV(T a, T b) { return (a + b - 1) / b; }

__device__ float sigmoid(float x) {
  return 0.5f + __tanhf(x * 0.5f) * 0.5f;
}

extern "C"
__launch_bounds__(BLOCK_SIZE, 1)  // minBlocks=1 : sur SM89 un seul bloc tient par SM (100 608 o smem).
                                  // Remettre 2 uniquement si portage Hopper (228 Kio/SM).
__global__ void ce_fwd_bwd_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ mtp_weights,
    float* __restrict__ losses,
    __nv_fp8_e5m2* grad_input,
    int batch_size,
    int n_predict,
    double A_param,
    double B_param,
    double C_param,
    double grad_s_param,
    double grad_scale_param)
{
  constexpr int VEC_WIDTH = 8;
  constexpr int NUM_FULL_LOADS = VOCAB_SIZE / (BLOCK_SIZE * VEC_WIDTH);
  constexpr int NUM_LOADS = CEIL_DIV(VOCAB_SIZE, BLOCK_SIZE * VEC_WIDTH);

  float A = (float)A_param;
  float B = (float)B_param;
  float C = (float)C_param;
  float grad_s = (float)grad_s_param;
  float grad_scale = (float)grad_scale_param;

  extern __shared__ __nv_bfloat16 smem[];

  static_assert(VEC_WIDTH == 8);

  const __nv_bfloat16 *block_logit_ptr = logits + (int64_t)VOCAB_SIZE * blockIdx.x;

  float inv_C = 1 / C;
  float B_div_C = B * inv_C;
  float thread_max = -CUDART_INF_F;

  #pragma unroll
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      __nv_bfloat168 result = *(__nv_bfloat168*)(&block_logit_ptr[idx]);
      __nv_bfloat168 result_sigmoid;
      #pragma unroll
      for (int k = 0; k < VEC_WIDTH; k++) {
        float tmp = __bfloat162float(result[k]);
        tmp = sigmoid(tmp * inv_C + B_div_C);
        result_sigmoid[k] = __float2bfloat16(tmp);
        tmp = A * tmp;
        thread_max = max(tmp, thread_max);
      }
      *(__nv_bfloat168*)(&smem[idx]) = result_sigmoid;
    }
  }

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  int warp_id = threadIdx.x / 32;
  __shared__ float block_maxs[NUM_WARPS];
  __shared__ float block_sums[NUM_WARPS];

  for (int offset = 16; offset > 0; offset >>= 1)
    thread_max = fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));

  if (threadIdx.x % 32 == 0) {
    block_maxs[warp_id] = thread_max;
  }

  __syncthreads();

  float block_max = -CUDART_INF_F;
  for (int i = 0; i < NUM_WARPS; i++) {
    block_max = fmaxf(block_max, block_maxs[i]);
  }

  float thread_sum = 0.0f;
  #pragma unroll 2
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    __nv_bfloat168 l;
    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      l = *(__nv_bfloat168*)(&smem[idx]);
    }
    #pragma unroll
    for (int k = 0; k < VEC_WIDTH; k++) {
      float tmp = A * __bfloat162float(l[k]);
      tmp = __expf(tmp - block_max);
      if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
        thread_sum += tmp;
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1)
    thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

  if (threadIdx.x % 32 == 0) {
    block_sums[warp_id] = thread_sum;
  }

  __syncthreads();

  float block_sum = 0.0f;
  for (int i = 0; i < NUM_WARPS; i++) {
    block_sum += block_sums[i];
  }

  float lse = block_max + __logf(block_sum);

  if (threadIdx.x == 0) {
    float total_loss = 0.0f;
    for (int k = 0; k < n_predict; k++) {
      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size) {
        float weight = mtp_weights[k];
        int64_t target = targets[target_idx];
        if (target >= 0 && target < VOCAB_SIZE) {
          float z_target = A * __bfloat162float(smem[target]);
          total_loss += weight * (lse - z_target);
        }
      }
    }
    losses[blockIdx.x] = total_loss;
  }

  float S_w = 0.0f;

  for (int i = 0; i < n_predict; i++) {
    S_w += mtp_weights[i];
  }

  #pragma unroll 4
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    __nv_fp8_e5m28 result;

    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      __nv_bfloat168 sigmoid_us = *(__nv_bfloat168*)(&smem[idx]);
      #pragma unroll
      for (int j = 0; j < VEC_WIDTH; j++) {
        float sigmoid_u = __bfloat162float(sigmoid_us[j]);
        float z = A * sigmoid_u;
        float p = __expf(z - lse);

        float term1 = S_w * p;
        float term2 = 0.0f;

        float grad_z = term1 - term2;
        float grad_x = grad_scale * (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);
        auto result_tmp = f32_to_fp8_e5m2(grad_x);
        result[j] = *reinterpret_cast<__nv_fp8_e5m2*>(&result_tmp);
      }
      *(__nv_fp8_e5m28*)(&grad_input[(int64_t)blockIdx.x * VOCAB_SIZE + idx]) = result;
    }
  }

  __syncthreads();

  if (threadIdx.x < n_predict && blockIdx.x + threadIdx.x < batch_size) {
    int i = threadIdx.x;
    int64_t target = targets[blockIdx.x + i];

    float sigmoid_u = __bfloat162float(smem[target]);
    float z = A * sigmoid_u;
    float p = __expf(z - lse);

    float term1 = S_w * p;
    float term2 = 0.0f;

    #pragma unroll
    for (int k = 0; k < 3; k++) {
      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size && k < n_predict) {
        if (targets[target_idx] == target) {
          term2 += mtp_weights[k];
        }
      }
    }

    float grad_z = term1 - term2;
    float grad_x = grad_scale * (1.0f / C * A) * (1.0f / grad_s) * grad_z * sigmoid_u * (1.0f - sigmoid_u);
    auto result_tmp = f32_to_fp8_e5m2(grad_x);
    auto result = *reinterpret_cast<__nv_fp8_e5m2*>(&result_tmp);
    grad_input[(int64_t)blockIdx.x * VOCAB_SIZE + target] = result;
  }
}
"""

# PATCH (RTX 4060 Ti / Ada Lovelace) : compute_capability="90" ciblait Hopper (H100) en dur,
# ce qui produit un PTX incompatible avec Ada (SM89) -> "PTX JIT compilation failed".
# On détecte la capacité de calcul réelle de la carte installée à la place.
_cc_major, _cc_minor = torch.cuda.get_device_capability()
ce_fwd_bwd_kernel = torch.cuda._compile_kernel(
    CE_KERNEL_DECLS + CE_KERNEL_SOURCE,
    "ce_fwd_bwd_kernel",
    compute_capability=f"{_cc_major}{_cc_minor}",
    cuda_include_dirs=["/usr/local/cuda/include/"],
    nvcc_options=["-lineinfo", "--use_fast_math"],
)
ce_fwd_bwd_kernel.set_shared_memory_config(CE_KERNEL_VOCAB_SIZE * 2)

@torch.library.custom_op("nanogpt::ce_fwd_bwd", mutates_args={"losses", "grad_input"})
def ce_fwd_bwd(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mtp_weights: torch.Tensor,
    losses: torch.Tensor,
    grad_input: torch.Tensor,
    n_rows: int,
    n_predict: int,
    A: float,
    B: float,
    C: float,
    grad_s: float,
    grad_scale: float,
) -> None:
    grid = (n_rows, 1, 1)
    ce_fwd_bwd_kernel(
        grid,
        (CE_KERNEL_BLOCK_SIZE, 1, 1),
        (logits, targets, mtp_weights, losses, grad_input,
         n_rows, n_predict, A, B, C, grad_s, grad_scale),
        shared_mem=CE_KERNEL_VOCAB_SIZE * 2,
    )

class FusedSoftcappedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, targets, mtp_weights, lm_head_weight, x_s, w_s, grad_s, grad_scale, A=23.0, B=5.0, C=7.5):

        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = lm_head_weight.div(w_s).to(torch.float8_e4m3fn)

        w_f8_col_major = w_f8.T.contiguous().T

        logits = torch._scaled_mm(
            x_f8,
            w_f8_col_major,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )

        n_rows, n_cols = logits.shape
        if mtp_weights is None:
             mtp_weights = torch.tensor([1.0], device=logits.device, dtype=torch.float32)
        n_predict = mtp_weights.shape[0]

        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        logits = logits.contiguous()
        targets = targets.contiguous()
        mtp_weights = mtp_weights.contiguous()

        grad_input = torch.empty((n_rows, n_cols), dtype=torch.float8_e5m2, device=logits.device)

        ce_fwd_bwd(logits, targets, mtp_weights, losses, grad_input,
             n_rows, n_predict, A, B, C, grad_s, grad_scale)

        ctx.save_for_backward(x_f8, w_f8, grad_input)
        ctx.params = (A, B, C, x_s, w_s, grad_s)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        x_f8, w_f8, grad_input = ctx.saved_tensors
        A, B, C, x_s, w_s, grad_s = ctx.params
        n_rows, n_cols = grad_input.shape   # remplace logits.shape

        grad_output = grad_output.contiguous()

        x_scale = grad_input.new_tensor(x_s, dtype=torch.float32)
        w_scale = grad_input.new_tensor(w_s, dtype=torch.float32)
        grad_scale = grad_input.new_tensor(grad_s, dtype=torch.float32)

        grad_x = torch._scaled_mm(
            grad_input, w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=grad_scale, scale_b=w_scale,
            use_fast_accum=False,
        )

        x_f8_T = torch.empty((x_f8.shape[1], x_f8.shape[0]), dtype=x_f8.dtype, device=x_f8.device)
        transpose_copy(x_f8, x_f8_T)

        grad_input_T = torch.empty((n_cols, n_rows), dtype=grad_input.dtype, device=grad_input.device)
        transpose_copy(grad_input, grad_input_T)

        grad_w = torch._scaled_mm(
            x_f8_T, grad_input_T.T,
            out_dtype=torch.float32,
            scale_a=x_scale, scale_b=grad_scale,
            use_fast_accum=False,
        )

        # 11 entrées -> 11 sorties (None pour tout ce qui n'est pas différentiable)
        return grad_x, None, None, grad_w, None, None, None, None, None, None, None