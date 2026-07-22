import torch
from torch import Tensor

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng
# Transposed layout by @ChrisJMcCormick allows for faster gradient accumulation.

@torch.library.custom_op("nanogpt::mm_t", mutates_args=())
def mm_t_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    """Computes y = x @ w with F8 weights stored as (in_features, out_features)."""
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        assert x.shape[1] == w.shape[0]  # x: (batch, in), w: (in, out)

        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)

        # _scaled_mm requires column-major B. w_f8 is row-major (in, out).
        # .T.contiguous().T creates a column-major view without changing logical shape.
        w_f8_col_major = w_f8.T.contiguous().T

        out = torch._scaled_mm(
            x_f8,
            w_f8_col_major,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_t_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[0]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_t_backward", mutates_args=())
def mm_t_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()

        x_scale = grad.new_tensor(x_s, dtype=torch.float32)
        w_scale = grad.new_tensor(w_s, dtype=torch.float32)
        grad_scale = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)

        # grad_x = grad @ w.T
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=grad_scale,
            scale_b=w_scale,
            use_fast_accum=False,
        )

        # grad_w = x.T @ grad
        # Result is (in, out), naturally matching weight storage. No final .T needed.
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_scale,
            scale_b=grad_scale,
            use_fast_accum=False,
        )

        return grad_x, grad_w

    grad_x, grad_w = impl(g, x_f8, w_f8)

    return grad_x, grad_w

@mm_t_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.to(torch.float32)

def backward_t(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_t_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context_t(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_t_op.register_autograd(backward_t, setup_context=setup_context_t)
