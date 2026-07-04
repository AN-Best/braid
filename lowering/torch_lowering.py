"""
lowering/torch_lowering.py
===========================
Lowers a Braid IR dict to a pure PyTorch ODE function.

The resulting function:
- Uses only native torch operations (no CasADi at runtime)
- Supports arbitrary batch dimensions: x.shape = (..., n_states)
- Is fully differentiable via PyTorch autograd
- Is compatible with torch.vmap for massively parallel batching
- Works directly with torchdiffeq.odeint / odeint_adjoint

GPU usage
---------
Move tensors to device before calling:
    y0 = y0.to('cuda')
    p  = p.to('cuda')
The lowered function automatically inherits the device from its inputs.
"""

import torch


def _eval_ast(node: dict, x: torch.Tensor, p: torch.Tensor,
              state_idx: dict, param_idx: dict) -> torch.Tensor:
    """
    Recursively evaluate an IR AST node as a PyTorch expression.

    x: (..., n_states)  — state tensor with arbitrary leading batch dims
    p: (..., n_params) or (n_params,)  — parameter tensor
    """
    op = node['op']

    # ── Leaves ───────────────────────────────────────────────────────────────
    if op == 'var':
        i = state_idx[node['name']]
        return x[..., i]

    if op == 'param':
        i = param_idx[node['name']]
        if p.dim() == 1:
            return p[i]
        return p[..., i]

    if op == 'const':
        val = float(node['value'])
        return torch.tensor(val, dtype=x.dtype, device=x.device)

    # ── Composite ─────────────────────────────────────────────────────────────
    args = [_eval_ast(a, x, p, state_idx, param_idx) for a in node['args']]

    dispatch = {
        'add':  lambda a: a[0] + a[1],
        'sub':  lambda a: a[0] - a[1],
        'mul':  lambda a: a[0] * a[1],
        'div':  lambda a: a[0] / a[1],
        'neg':  lambda a: -a[0],
        'pow':  lambda a: torch.pow(a[0], a[1]),
        'sin':  lambda a: torch.sin(a[0]),
        'cos':  lambda a: torch.cos(a[0]),
        'tan':  lambda a: torch.tan(a[0]),
        'exp':  lambda a: torch.exp(a[0]),
        'log':  lambda a: torch.log(a[0]),
        'sqrt': lambda a: torch.sqrt(a[0]),
        'abs':  lambda a: torch.abs(a[0]),
        'min':  lambda a: torch.minimum(a[0], a[1]),
        'max':  lambda a: torch.maximum(a[0], a[1]),
        'ite':  lambda a: torch.where(a[0] != 0.0, a[1], a[2]),
    }

    if op not in dispatch:
        raise NotImplementedError(f"Torch lowering: unsupported op '{op}'")

    return dispatch[op](args)


def make_torch_ode(ir: dict):
    """
    Returns a PyTorch ODE function  f(t, x, p) → dxdt.

    The returned function handles:
    - Unbatched: x.shape = (n_states,)
    - Batched:   x.shape = (batch, n_states)  or  (..., n_states)

    Args:
        ir: Braid IR dict

    Returns:
        Callable: (t: Tensor, x: Tensor, p: Tensor) → Tensor same shape as x

    Example (GPU + torchdiffeq)::

        from lowering.torch_lowering import make_torch_ode
        from torchdiffeq import odeint_adjoint

        ode_fn = make_torch_ode(ir)
        p = torch.tensor([1.0, 10.0, 0.5], device='cuda')

        def func(t, y):
            return ode_fn(t, y, p)

        y0 = torch.zeros(1024, len(ir['states']), device='cuda')
        t  = torch.linspace(0, 5.0, 200, device='cuda')
        sol = odeint_adjoint(func, y0, t, method='dopri5')
        # sol.shape: (200, 1024, n_states)
    """
    state_idx = {name: i for i, name in enumerate(ir['states'])}
    param_idx = {name: i for i, name in enumerate(ir['params'])}
    rhs_asts  = [entry['expr'] for entry in ir['ode_rhs']]

    def ode_func(t: torch.Tensor, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        t: scalar or 0-d tensor (time)
        x: (..., n_states)
        p: (n_params,) or (..., n_params)
        Returns: (..., n_states)
        """
        components = [
            _eval_ast(ast, x, p, state_idx, param_idx)
            for ast in rhs_asts
        ]
        return torch.stack(components, dim=-1)

    return ode_func


def make_torch_jacobian(ir: dict):
    """
    Returns a function  J(t, x, p) → Tensor of shape (..., n_states, n_states)
    using torch.func.jacrev (vmap-compatible analytical Jacobian).

    Requires PyTorch >= 2.0.
    """
    ode_fn = make_torch_ode(ir)
    n = len(ir['states'])

    def jac_func(t: torch.Tensor, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # x: (..., n_states)
        # Returns: (..., n_states, n_states)
        def f_flat(x_flat):
            return ode_fn(t, x_flat, p)

        try:
            # PyTorch 2.0+ functional API
            from torch.func import jacrev
            J = jacrev(f_flat)(x)
        except ImportError:
            # Fallback: finite differences
            eps = 1e-6
            f0 = ode_fn(t, x, p)
            batch_shape = x.shape[:-1]
            J = torch.zeros(*batch_shape, n, n, dtype=x.dtype, device=x.device)
            for j in range(n):
                xp = x.clone()
                xp[..., j] = xp[..., j] + eps
                J[..., :, j] = (ode_fn(t, xp, p) - f0) / eps

        return J

    return jac_func


def make_torch_residuals(ir: dict):
    """
    Returns a callable  F(t, x, yp, p) → residuals  using PyTorch.

    Args:
        ir: Braid IR dict with 'residuals' and 'xdots'.

    Returns:
        Callable with signature  F(t: Tensor, x: Tensor, yp: Tensor, p: Tensor) → Tensor same shape as x
    """
    state_idx = {name: i for i, name in enumerate(ir['states'])}
    xdot_idx  = {name: i for i, name in enumerate(ir.get('xdots', []))}
    param_idx = {name: i for i, name in enumerate(ir['params'])}
    res_asts  = ir['residuals']

    def residual_func(t: torch.Tensor, x: torch.Tensor, yp: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        components = [
            _eval_ast_dae(ast, x, yp, p, state_idx, xdot_idx, param_idx)
            for ast in res_asts
        ]
        return torch.stack(components, dim=-1)

    return residual_func


def _eval_ast_dae(node: dict, x: torch.Tensor, yp: torch.Tensor, p: torch.Tensor,
                  state_idx: dict, xdot_idx: dict, param_idx: dict) -> torch.Tensor:
    op = node['op']

    if op == 'var':
        name = node['name']
        if name in state_idx:
            i = state_idx[name]
            return x[..., i]
        elif name in xdot_idx:
            i = xdot_idx[name]
            return yp[..., i]
        else:
            raise KeyError(f"Variable '{name}' not found in states or xdots.")

    if op == 'param':
        i = param_idx[node['name']]
        if p.dim() == 1:
            return p[i]
        return p[..., i]

    if op == 'const':
        val = float(node['value'])
        return torch.tensor(val, dtype=x.dtype, device=x.device)

    args = [_eval_ast_dae(a, x, yp, p, state_idx, xdot_idx, param_idx) for a in node['args']]

    dispatch = {
        'add':  lambda a: a[0] + a[1],
        'sub':  lambda a: a[0] - a[1],
        'mul':  lambda a: a[0] * a[1],
        'div':  lambda a: a[0] / a[1],
        'neg':  lambda a: -a[0],
        'pow':  lambda a: torch.pow(a[0], a[1]),
        'sin':  lambda a: torch.sin(a[0]),
        'cos':  lambda a: torch.cos(a[0]),
        'tan':  lambda a: torch.tan(a[0]),
        'exp':  lambda a: torch.exp(a[0]),
        'log':  lambda a: torch.log(a[0]),
        'sqrt': lambda a: torch.sqrt(a[0]),
        'abs':  lambda a: torch.abs(a[0]),
        'min':  lambda a: torch.minimum(a[0], a[1]),
        'max':  lambda a: torch.maximum(a[0], a[1]),
        'ite':  lambda a: torch.where(a[0] != 0.0, a[1], a[2]),
    }

    if op not in dispatch:
        raise NotImplementedError(f"Torch lowering: unsupported op '{op}'")

    return dispatch[op](args)

