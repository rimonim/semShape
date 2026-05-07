"""
Muon + AdamW optimizer, adapted from karpathy/nanochat.

Muon is applied to 2D matrix parameters in the transformer blocks (attention and MLP weights).
AdamW is applied to everything else (embeddings, unembedding, biases, scalars).

Reference: https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py
"""

import torch
import torch.distributed as dist
from torch import Tensor

# Detect compute dtype based on hardware
def _get_compute_dtype():
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:  # Ampere+
            return torch.bfloat16
    return torch.float32

COMPUTE_DTYPE = _get_compute_dtype()

# -----------------------------------------------------------------------------
# Fused optimizer kernels
# -----------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step_t: Tensor,
    lr_t: Tensor,
    beta1_t: Tensor,
    beta2_t: Tensor,
    eps_t: Tensor,
    wd_t: Tensor,
) -> None:
    # weight_decay -> momentum_update -> bias_correction -> param_update
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

# Polynomial coefficients for the Polar Express orthogonalization
# (Newton-Schulz iteration for approximate matrix polar decomposition)
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    red_dim: int,
) -> None:
    # momentum -> polar_express -> variance_reduction -> cautious_update
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Orthogonalize via Newton-Schulz (Polar Express)
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious update: only update in directions where gradient and parameter agree
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------
# Single-GPU MuonAdamW optimizer
# -----------------------------------------------------------------------------

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined Muon + AdamW optimizer for single-GPU training.

    Each param group must have a 'kind' key: either 'adamw' or 'muon'.
    AdamW groups need: lr, betas, eps, weight_decay
    Muon groups need: lr, momentum, ns_steps, beta2, weight_decay
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # Pre-allocate 0-D CPU tensors for hyperparameters (avoids repeated allocation)
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            adamw_step_fused(
                p, grad, exp_avg, exp_avg_sq,
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        params: list[Tensor] = group['params']
        if not params:
            return

        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        muon_step_fused(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )

        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")
