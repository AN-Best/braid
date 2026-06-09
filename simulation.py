import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sympy as sp
import numpy as np
from json_ir import from_json

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

def lambdify_jacobian(dae, backend: str):
    """Compiles the analytical Jacobian of the ODE assignments into a target backend function.
    
    The resulting function has the signature:
        J(t, states, params)
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
            
    # Compute symbolic Jacobian with respect to states
    J_sym = sp.Matrix(exprs).jacobian(states)
    
    backend_map = {
        'numpy': 'numpy',
        'pytorch': 'torch',
        'torch': 'torch',
        'jax': 'jax'
    }
    target_backend = backend_map.get(backend.lower(), backend)
    
    return sp.lambdify((t, states, params), J_sym, target_backend)

def simulate_system(dae, t_span, y0, params, backend='numpy', method=None, device=None, **kwargs):
    """Integrates the ODE system defined by the SystemDAE object using the selected backend and method.
    
    Parameters:
        dae: A SystemDAE instance, a JSON file path, or a JSON string.
        t_span: Tuple of (t0, tf).
        y0: List/array of initial state values.
        params: List/array of parameter values.
        backend: String ('numpy', 'pytorch', 'jax').
        method: Solver method.
        device: The PyTorch device (for 'pytorch' backend).
        kwargs: Extra config (e.g. num_steps=1000).
    """
    if isinstance(dae, (str, os.PathLike)):
        path_str = os.fspath(dae)
        if os.path.isfile(path_str):
            with open(path_str, 'r', encoding='utf-8') as f:
                json_str = f.read()
        else:
            json_str = path_str
        dae = from_json(json_str)

    backend_lower = backend.lower()
    
    if backend_lower in ('numpy', 'scipy'):
        from backends.numpy_backend import simulate_numpy
        ode_func_raw = lambdify_system(dae, 'numpy')
        jac_func_raw = lambdify_jacobian(dae, 'numpy')
        return simulate_numpy(ode_func_raw, jac_func_raw, t_span, y0, params, method, **kwargs)
        
    elif backend_lower in ('pytorch', 'torch'):
        from backends.torch_backend import simulate_torch
        ode_func_raw = lambdify_system(dae, 'torch')
        jac_func_raw = lambdify_jacobian(dae, 'torch')
        return simulate_torch(ode_func_raw, jac_func_raw, t_span, y0, params, method, device, **kwargs)
        
    elif backend_lower == 'jax':
        raise ImportError(
            "JAX is not installed in the active environment. "
            "Please install JAX to use the 'jax' simulation backend."
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Supported backends: 'numpy', 'pytorch'.")
