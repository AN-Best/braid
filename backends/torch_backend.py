import numpy as np
import torch

import torchdiffeq

class PyTorchSimulationResult:
    def __init__(self, t_arr, y_arr):
        self.t = t_arr
        self.y = y_arr
        self.success = True

def simulate_torch(ode_func_raw, jac_func_raw, t_span, y0, params, method, device=None, **kwargs):
    if method is None:
        method = 'dopri5'
        
    method_lower = method.lower()
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # y0 and params can be batched or single
    y = torch.as_tensor(y0, dtype=torch.float64, device=device)
    params_tensor = torch.as_tensor(params, dtype=torch.float64, device=device)
    
    is_batched = (y.dim() > 1)
    
    # RHS wrapper for torchdiffeq
    # torchdiffeq expects f(t, y) where t is a scalar tensor and y is [states] or [batch, states]
    def f(t_val, y_vec):
        # t_val from torchdiffeq is a scalar tensor
        t_scalar = float(t_val.cpu().item()) if t_val.dim() == 0 else float(t_val[0].cpu().item())

        # ode_func_raw (from torch_lowering.make_torch_ode) expects:
        #   t: scalar, x: (..., n_states) tensor, p: (n_params,) or (..., n_params) tensor
        # Pass tensors directly — no list unpacking needed.
        result = ode_func_raw(t_scalar, y_vec, params_tensor)
        return result

    # Resolve evaluation times
    t_eval = kwargs.get('t_eval', None)
    if t_eval is None:
        num_steps = kwargs.get('num_steps', 100)
        t_eval = torch.linspace(t_span[0], t_span[1], num_steps, dtype=torch.float64, device=device)
    else:
        t_eval = torch.as_tensor(t_eval, dtype=torch.float64, device=device)

    # Use torchdiffeq adjoint or standard integrator
    use_adjoint = kwargs.get('use_adjoint', False)
    integrate_fn = torchdiffeq.odeint_adjoint if use_adjoint else torchdiffeq.odeint

    # Dopri5 is default, but others like 'rk4', 'euler', 'adams' are supported
    rtol = kwargs.get('rtol', 1e-7)
    atol = kwargs.get('atol', 1e-9)

    # Perform integration
    options = kwargs.get('options', None)
    y_solved = integrate_fn(
        f, 
        y, 
        t_eval, 
        rtol=rtol, 
        atol=atol, 
        method=method,
        options=options
    )

    # torchdiffeq outputs y_solved of shape [len(t_eval), batch_size, n_states] (if batched)
    # or [len(t_eval), n_states] (if single)
    if is_batched:
        # Permute to [batch_size, n_states, len(t_eval)] for Braid API compatibility
        y_out = y_solved.permute(1, 2, 0)
    else:
        # Permute to [n_states, len(t_eval)] for Braid API compatibility
        y_out = y_solved.permute(1, 0)

    # Return as numpy array for simulation outputs, keeping cpu compatibility
    return PyTorchSimulationResult(t_eval.cpu().numpy(), y_out.cpu().numpy())


def simulate_torch_dae(residual_func, t_span, y0, yp0, params, method, device=None, **kwargs):
    import torchdae
    if method is None:
        method = 'radau'

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    y = torch.as_tensor(y0, dtype=torch.float64, device=device)
    params_tensor = torch.as_tensor(params, dtype=torch.float64, device=device)

    is_batched = (y.dim() > 1)
    if not is_batched:
        # torchdae requires batch dim
        y_input = y.unsqueeze(0)
    else:
        y_input = y

    if yp0 is not None:
        yp = torch.as_tensor(yp0, dtype=torch.float64, device=device)
        if not is_batched:
            yp_input = yp.unsqueeze(0)
        else:
            yp_input = yp
    else:
        yp_input = torch.zeros_like(y_input)

    def F_wrapper(t_val, y_val, yp_val):
        t_scalar = float(t_val)
        return residual_func(t_scalar, y_val, yp_val, params_tensor)

    num_steps = kwargs.get('num_steps', 100)

    method_lower = method.lower()
    if method_lower in ('radau', 'solve_radau_iia5'):
        sol = torchdae.solve_radau_iia5(F_wrapper, t_span, y_input, yp0=yp_input, n_steps=num_steps)
    elif method_lower in ('bdf1', 'solve_bdf1'):
        sol = torchdae.solve_bdf1(F_wrapper, t_span, y_input, yp0=yp_input, n_steps=num_steps)
    elif method_lower in ('bdf2', 'solve_bdf2'):
        sol = torchdae.solve_bdf2(F_wrapper, t_span, y_input, yp0=yp_input, n_steps=num_steps)
    elif method_lower in ('generalized_alpha', 'solve_generalized_alpha'):
        sol = torchdae.solve_generalized_alpha(F_wrapper, t_span, y_input, yp0=yp_input, n_steps=num_steps)
    else:
        raise ValueError(
            f"Unknown PyTorch DAE method '{method}'. "
            "Supported: 'radau', 'bdf1', 'bdf2', 'generalized_alpha'"
        )

    ts = sol.ts.cpu().numpy()
    ys = sol.ys

    if is_batched:
        # Permute from [time, batch, states] to [batch, states, time]
        y_out = ys.permute(1, 2, 0).cpu().numpy()
    else:
        # Permute from [time, 1, states] to [states, time]
        y_out = ys.squeeze(1).permute(1, 0).cpu().numpy()

    return PyTorchSimulationResult(ts, y_out)

