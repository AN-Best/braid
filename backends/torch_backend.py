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
        
        if is_batched:
            y_list_input = [y_vec[:, i] for i in range(y_vec.shape[1])]
        else:
            y_list_input = [y_vec[i] for i in range(len(y_vec))]
            
        if params_tensor.dim() > 1:
            params_list_input = [params_tensor[:, i] for i in range(params_tensor.shape[1])]
        else:
            params_list_input = [params_tensor[i] for i in range(len(params_tensor))]
            
        res = ode_func_raw(t_scalar, y_list_input, params_list_input)
        
        stack_dim = -1 if is_batched else 0
        
        tensors = []
        for r in res:
            t_r = torch.as_tensor(r, dtype=torch.float64, device=device)
            if is_batched and t_r.dim() == 0:
                t_r = t_r.expand(y_vec.shape[0])
            tensors.append(t_r)
            
        return torch.stack(tensors, dim=stack_dim)

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
