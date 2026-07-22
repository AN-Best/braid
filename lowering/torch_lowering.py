"""
lowering/torch_lowering.py
===========================
Lowers a Braid IR dict to a compiled PyTorch ODE/DAE function.

The resulting function:
- Uses only native torch operations (no CasADi at runtime)
- Supports arbitrary batch dimensions: x.shape = (..., n_states)
- Is fully differentiable via PyTorch autograd
- Is compatible with torch.vmap for massively parallel batching
- Works directly with torchdiffeq.odeint / odeint_adjoint
"""

import torch

def _ast_to_str(node: dict, state_idx: dict, param_idx: dict) -> str:
    op = node['op']
    if op == 'var':
        i = state_idx[node['name']]
        return f"x[..., {i}]"
    if op == 'param':
        i = param_idx[node['name']]
        return f"p[..., {i}]"
    if op == 'const':
        return str(float(node['value']))
    
    args = [_ast_to_str(a, state_idx, param_idx) for a in node['args']]
    
    dispatch = {
        'add':  lambda a: f"({a[0]} + {a[1]})",
        'sub':  lambda a: f"({a[0]} - {a[1]})",
        'mul':  lambda a: f"({a[0]} * {a[1]})",
        'div':  lambda a: f"({a[0]} / {a[1]})",
        'neg':  lambda a: f"(-{a[0]})",
        'pow':  lambda a: f"torch.pow({a[0]}, {a[1]})",
        'sin':  lambda a: f"torch.sin({a[0]})",
        'cos':  lambda a: f"torch.cos({a[0]})",
        'tan':  lambda a: f"torch.tan({a[0]})",
        'exp':  lambda a: f"torch.exp({a[0]})",
        'log':  lambda a: f"torch.log({a[0]})",
        'sqrt': lambda a: f"torch.sqrt({a[0]})",
        'abs':  lambda a: f"torch.abs({a[0]})",
        'min':  lambda a: f"torch.minimum({a[0]}, {a[1]})",
        'max':  lambda a: f"torch.maximum({a[0]}, {a[1]})",
        'ite':  lambda a: f"torch.where({a[0]} != 0.0, {a[1]}, {a[2]})",
        'atan2': lambda a: f"torch.atan2({a[0]}, {a[1]})",
    }

    
    if op not in dispatch:
        raise NotImplementedError(f"Torch lowering: unsupported op '{op}'")
    return dispatch[op](args)

def make_torch_ode(ir: dict):
    state_idx = {name: i for i, name in enumerate(ir['states'])}
    param_idx = {name: i for i, name in enumerate(ir['params'])}
    rhs_strs = []
    for entry in ir['ode_rhs']:
        rhs_strs.append(_ast_to_str(entry['expr'], state_idx, param_idx))
    
    stack_str = ", ".join([f"torch.as_tensor({s}, device=x.device, dtype=x.dtype)" for s in rhs_strs])
    code = f"""
def ode_func_compiled(t, x, p):
    return torch.stack([{stack_str}], dim=-1)
"""
    local_vars = {}
    exec(code, {'torch': torch}, local_vars)
    return local_vars['ode_func_compiled']

def make_torch_jacobian(ir: dict):
    ode_fn = make_torch_ode(ir)
    n = len(ir['states'])

    def jac_func(t: torch.Tensor, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        def f_flat(x_flat):
            return ode_fn(t, x_flat, p)

        try:
            from torch.func import jacrev
            J = jacrev(f_flat)(x)
        except ImportError:
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

def _ast_to_str_dae(node: dict, state_idx: dict, xdot_idx: dict, param_idx: dict) -> str:
    op = node['op']
    if op == 'var':
        name = node['name']
        if name in state_idx:
            i = state_idx[name]
            return f"x[..., {i}]"
        elif name in xdot_idx:
            i = xdot_idx[name]
            return f"yp[..., {i}]"
        else:
            raise KeyError(f"Variable '{name}' not found in states or xdots.")
    if op == 'param':
        i = param_idx[node['name']]
        return f"p[..., {i}]"
    if op == 'const':
        return str(float(node['value']))
    
    args = [_ast_to_str_dae(a, state_idx, xdot_idx, param_idx) for a in node['args']]
    
    dispatch = {
        'add':  lambda a: f"({a[0]} + {a[1]})",
        'sub':  lambda a: f"({a[0]} - {a[1]})",
        'mul':  lambda a: f"({a[0]} * {a[1]})",
        'div':  lambda a: f"({a[0]} / {a[1]})",
        'neg':  lambda a: f"(-{a[0]})",
        'pow':  lambda a: f"torch.pow({a[0]}, {a[1]})",
        'sin':  lambda a: f"torch.sin({a[0]})",
        'cos':  lambda a: f"torch.cos({a[0]})",
        'tan':  lambda a: f"torch.tan({a[0]})",
        'exp':  lambda a: f"torch.exp({a[0]})",
        'log':  lambda a: f"torch.log({a[0]})",
        'sqrt': lambda a: f"torch.sqrt({a[0]})",
        'abs':  lambda a: f"torch.abs({a[0]})",
        'min':  lambda a: f"torch.minimum({a[0]}, {a[1]})",
        'max':  lambda a: f"torch.maximum({a[0]}, {a[1]})",
        'ite':  lambda a: f"torch.where({a[0]} != 0.0, {a[1]}, {a[2]})",
        'atan2': lambda a: f"torch.atan2({a[0]}, {a[1]})",
    }

    
    if op not in dispatch:
        raise NotImplementedError(f"Torch lowering: unsupported op '{op}'")
    return dispatch[op](args)

def make_torch_residuals(ir: dict):
    state_idx = {name: i for i, name in enumerate(ir['states'])}
    xdot_idx  = {name: i for i, name in enumerate(ir.get('xdots', []))}
    param_idx = {name: i for i, name in enumerate(ir['params'])}
    
    res_strs = []
    for ast in ir['residuals']:
        res_strs.append(_ast_to_str_dae(ast, state_idx, xdot_idx, param_idx))
    
    stack_str = ", ".join([f"torch.as_tensor({s}, device=x.device, dtype=x.dtype)" for s in res_strs])
    code = f"""
def residual_func_compiled(t, x, yp, p):
    return torch.stack([{stack_str}], dim=-1)
"""
    local_vars = {}
    exec(code, {'torch': torch}, local_vars)
    return local_vars['residual_func_compiled']

