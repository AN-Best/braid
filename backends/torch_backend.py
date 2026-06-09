import numpy as np
import torch

def rk4_step_torch(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2.0, y + h/2.0 * k1)
    k3 = f(t + h/2.0, y + h/2.0 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

def backward_euler_step_torch(f, jac_fn, t, y, h, tol=1e-8, max_iter=50):
    # Solve y_next = y + h * f(t + h, y_next) using Newton-Raphson
    y_next = y.clone().detach().requires_grad_(False)
    
    for _ in range(max_iter):
        k = f(t + h, y_next)
        g = y_next - y - h * k
        if torch.norm(g) < tol:
            return y_next.clone()
            
        J_f = jac_fn(t + h, y_next)
        J = torch.eye(len(y), dtype=torch.float64, device=y.device) - h * J_f
        try:
            delta = torch.linalg.solve(J, -g)
        except RuntimeError:
            delta = -g
            
        y_next = y_next + delta
        
    return y_next.clone()

STEPPERS_TORCH = {
    'euler': lambda f, t, y, h: y + h * f(t, y),
    'rk4': rk4_step_torch
}

class PyTorchSimulationResult:
    def __init__(self, t_arr, y_arr):
        self.t = t_arr
        self.y = y_arr
        self.success = True

def simulate_torch(ode_func_raw, jac_func_raw, t_span, y0, params, method, device=None, **kwargs):
    if method is None:
        method = 'rk4'
        
    method_lower = method.lower()
    if method_lower not in STEPPERS_TORCH and method_lower != 'backward_euler':
        raise ValueError(f"Unknown PyTorch method '{method}'. Supported: {list(STEPPERS_TORCH.keys()) + ['backward_euler']}")
        
    num_steps = kwargs.get('num_steps', 1000)
    t0, tf = t_span
    h = (tf - t0) / num_steps
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # y0 and params can be batched or single
    y = torch.as_tensor(y0, dtype=torch.float64, device=device)
    params_tensor = torch.as_tensor(params, dtype=torch.float64, device=device)
    
    is_batched = (y.dim() > 1)
    
    t_list = [t0]
    y_list = [y.clone()]
    
    # RHS wrapper
    def f(t_val, y_vec):
        if is_batched:
            y_list_input = [y_vec[:, i] for i in range(y_vec.shape[1])]
        else:
            y_list_input = [y_vec[i] for i in range(len(y_vec))]
            
        if params_tensor.dim() > 1:
            params_list_input = [params_tensor[:, i] for i in range(params_tensor.shape[1])]
        else:
            params_list_input = [params_tensor[i] for i in range(len(params_tensor))]
            
        res = ode_func_raw(t_val, y_list_input, params_list_input)
        
        stack_dim = -1 if is_batched else 0
        
        tensors = []
        for r in res:
            t_r = torch.as_tensor(r, dtype=torch.float64, device=device)
            if is_batched and t_r.dim() == 0:
                t_r = t_r.expand(y_vec.shape[0])
            tensors.append(t_r)
            
        return torch.stack(tensors, dim=stack_dim)
        
    def jac_fn(t_val, y_vec):
        if is_batched:
            y_list_input = [y_vec[:, i] for i in range(y_vec.shape[1])]
        else:
            y_list_input = [y_vec[i] for i in range(len(y_vec))]
            
        if params_tensor.dim() > 1:
            params_list_input = [params_tensor[:, i] for i in range(params_tensor.shape[1])]
        else:
            params_list_input = [params_tensor[i] for i in range(len(params_tensor))]
            
        res_jac = jac_func_raw(t_val, y_list_input, params_list_input)
        
        rows = []
        for row in res_jac:
            row_tensors = []
            for val in row:
                val_tensor = torch.as_tensor(val, dtype=torch.float64, device=device)
                if is_batched and val_tensor.dim() == 0:
                    val_tensor = val_tensor.expand(y_vec.shape[0])
                row_tensors.append(val_tensor)
            rows.append(torch.stack(row_tensors, dim=-1))
            
        stack_dim = 1 if is_batched else 0
        return torch.stack(rows, dim=stack_dim)

    curr_t = t0
    curr_y = y
    
    if method_lower == 'backward_euler' and is_batched:
        # Batched backward Euler Newton-Raphson solver loops over batch elements
        for _ in range(num_steps):
            y_next = curr_y.clone()
            for b in range(y.shape[0]):
                def f_single(t_val, y_single):
                    y_list_input_single = [y_single[i] for i in range(len(y_single))]
                    if params_tensor.dim() > 1:
                        params_list_input_single = [params_tensor[b, i] for i in range(params_tensor.shape[1])]
                    else:
                        params_list_input_single = [params_tensor[i] for i in range(len(params_tensor))]
                    res_single = ode_func_raw(t_val, y_list_input_single, params_list_input_single)
                    return torch.stack([torch.as_tensor(r, dtype=torch.float64, device=device) for r in res_single])
                
                def jac_single(t_val, y_single):
                    y_list_input_single = [y_single[i] for i in range(len(y_single))]
                    if params_tensor.dim() > 1:
                        params_list_input_single = [params_tensor[b, i] for i in range(params_tensor.shape[1])]
                    else:
                        params_list_input_single = [params_tensor[i] for i in range(len(params_tensor))]
                    res_jac_single = jac_func_raw(t_val, y_list_input_single, params_list_input_single)
                    rows_single = []
                    for row in res_jac_single:
                        rows_single.append(torch.stack([torch.as_tensor(val, dtype=torch.float64, device=device) for val in row]))
                    return torch.stack(rows_single, dim=0)
                
                y_next[b] = backward_euler_step_torch(f_single, jac_single, curr_t, curr_y[b], h)
            curr_y = y_next
            curr_t += h
            t_list.append(curr_t)
            y_list.append(curr_y.clone())
            
    elif method_lower == 'backward_euler':
        for _ in range(num_steps):
            curr_y = backward_euler_step_torch(f, jac_fn, curr_t, curr_y, h)
            curr_t += h
            t_list.append(curr_t)
            y_list.append(curr_y.clone())
            
    else:
        step_fn = STEPPERS_TORCH[method_lower]
        for _ in range(num_steps):
            curr_y = step_fn(f, curr_t, curr_y, h)
            curr_t += h
            t_list.append(curr_t)
            y_list.append(curr_y.clone())
            
    y_stacked = torch.stack(y_list)
    if is_batched:
        y_out = y_stacked.permute(1, 2, 0).cpu().numpy()
    else:
        y_out = y_stacked.t().cpu().numpy()
        
    return PyTorchSimulationResult(np.array(t_list), y_out)
