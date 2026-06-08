import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sympy as sp
import numpy as np

# --- NumPy Steppers ---

def rk4_step_numpy(f, t, y, h):
    k1 = np.array(f(t, y))
    k2 = np.array(f(t + h/2.0, y + h/2.0 * k1))
    k3 = np.array(f(t + h/2.0, y + h/2.0 * k2))
    k4 = np.array(f(t + h, y + h * k3))
    return y + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

def backward_euler_step_numpy(f, t, y, h):
    from scipy.optimize import root
    def g(y_next):
        return y_next - y - h * np.array(f(t + h, y_next))
    sol = root(g, y, method='hybr')
    return sol.x

STEPPERS_NUMPY = {
    'euler': lambda f, t, y, h: y + h * np.array(f(t, y)),
    'rk4': rk4_step_numpy,
    'backward_euler': backward_euler_step_numpy
}

# --- PyTorch Steppers ---

def rk4_step_torch(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2.0, y + h/2.0 * k1)
    k3 = f(t + h/2.0, y + h/2.0 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

def backward_euler_step_torch(f, t, y, h, tol=1e-8, max_iter=50):
    import torch
    # Solve y_next = y + h * f(t + h, y_next) using Newton-Raphson
    y_next = y.clone().detach().requires_grad_(True)
    
    for _ in range(max_iter):
        k = f(t + h, y_next)
        g = y_next - y - h * k
        if torch.norm(g) < tol:
            return y_next.detach()
            
        def g_func(y_val):
            return y_val - y - h * f(t + h, y_val)
            
        # Compute Jacobian J = I - h * J_f
        J = torch.autograd.functional.jacobian(g_func, y_next)
        try:
            delta = torch.linalg.solve(J, -g)
        except RuntimeError:
            delta = -g
            
        y_next = (y_next + delta).detach().requires_grad_(True)
        
    return y_next.detach()

STEPPERS_TORCH = {
    'euler': lambda f, t, y, h: y + h * f(t, y),
    'rk4': rk4_step_torch,
    'backward_euler': backward_euler_step_torch
}

# --- Core Compilation and Integration ---

def lambdify_system(dae, backend: str):
    """Compiles the ODE assignments of a SystemDAE object into a target backend function.
    
    The resulting function has the signature:
        f(t, states, params)
    """
    t = dae.t
    states = dae.states
    params = dae.params
    
    exprs = []
    for state in states:
        state_deriv = sp.Derivative(state, t)
        if state_deriv in dae.ode_assignments:
            exprs.append(dae.ode_assignments[state_deriv])
        else:
            raise ValueError(f"Derivative of state {state} ({state_deriv}) is not defined in ode_assignments.")
            
    # Map friendly names to SymPy's internal lambdify backend names
    backend_map = {
        'numpy': 'numpy',
        'pytorch': 'torch',
        'torch': 'torch',
        'jax': 'jax'
    }
    target_backend = backend_map.get(backend.lower(), backend)
    
    return sp.lambdify((t, states, params), exprs, target_backend)

def simulate_system(dae, t_span, y0, params, backend='numpy', method=None, device=None, **kwargs):
    """Integrates the ODE system defined by the SystemDAE object using the selected backend and method.
    
    Parameters:
        dae: A SystemDAE instance.
        t_span: Tuple of (t0, tf).
        y0: List/array of initial state values.
        params: List/array of parameter values.
        backend: String ('numpy', 'pytorch', 'jax').
        method: Solver method:
            - For numpy: any scipy method ('RK45', 'BDF', etc.) or custom ('euler', 'rk4', 'backward_euler').
            - For pytorch: custom ('euler', 'rk4', 'backward_euler').
        kwargs: Extra config (e.g. num_steps=1000).
    """
    backend_lower = backend.lower()
    
    if backend_lower in ('numpy', 'scipy'):
        # Default numpy method to SciPy's RK45
        if method is None:
            method = 'RK45'
            
        ode_func_raw = lambdify_system(dae, 'numpy')
        
        # Check if we should use SciPy's built-in solver or our custom loop
        if method.lower() in STEPPERS_NUMPY:
            num_steps = kwargs.get('num_steps', 1000)
            t0, tf = t_span
            h = (tf - t0) / num_steps
            
            step_fn = STEPPERS_NUMPY[method.lower()]
            
            t_list = [t0]
            y_list = [np.array(y0, dtype=np.float64)]
            
            curr_t = t0
            curr_y = y_list[0]
            
            # Wrapper fun for step functions
            def f(t_val, y_val):
                return ode_func_raw(t_val, y_val, params)
                
            for _ in range(num_steps):
                curr_y = step_fn(f, curr_t, curr_y, h)
                curr_t += h
                t_list.append(curr_t)
                y_list.append(curr_y.copy())
                
            class NumpyCustomResult:
                def __init__(self, t_arr, y_arr):
                    self.t = t_arr
                    self.y = y_arr.T
                    self.success = True
                    
            return NumpyCustomResult(np.array(t_list), np.array(y_list))
        else:
            # SciPy solve_ivp
            import scipy.integrate
            def fun(t_val, y_val):
                return ode_func_raw(t_val, y_val, params)
            sol = scipy.integrate.solve_ivp(fun, t_span, y0, method=method, **kwargs)
            return sol
            
    elif backend_lower in ('pytorch', 'torch'):
        import torch
        
        if method is None:
            method = 'rk4'
            
        method_lower = method.lower()
        if method_lower not in STEPPERS_TORCH:
            raise ValueError(f"Unknown PyTorch method '{method}'. Supported: {list(STEPPERS_TORCH.keys())}")
            
        num_steps = kwargs.get('num_steps', 1000)
        t0, tf = t_span
        h = (tf - t0) / num_steps
        
        ode_func_raw = lambdify_system(dae, 'torch')
        
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
            return torch.stack([torch.as_tensor(r, dtype=torch.float64, device=device) for r in res], dim=stack_dim)
            
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
                    
                    y_next[b] = backward_euler_step_torch(f_single, curr_t, curr_y[b], h)
                curr_y = y_next
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
            
        class PyTorchSimulationResult:
            def __init__(self, t_arr, y_arr):
                self.t = t_arr
                self.y = y_arr
                self.success = True
                
        return PyTorchSimulationResult(np.array(t_list), y_out)
        
    elif backend_lower == 'jax':
        raise ImportError(
            "JAX is not installed in the active environment. "
            "Please install JAX to use the 'jax' simulation backend."
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Supported backends: 'numpy', 'pytorch'.")
