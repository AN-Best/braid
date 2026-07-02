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

    # Resolve params if passed as a dictionary or None (to use defaults)
    if isinstance(params, dict) or params is None:
        param_dict = params or {}
        flat_params = []
        param_meta = getattr(dae, "param_meta", {})
        
        for p in dae.params:
            val = None
            found = False
            
            # 1. Match by Symbol object key
            if p in param_dict:
                val = param_dict[p]
                found = True
            # 2. Match by Symbol name string key
            elif p.name in param_dict:
                val = param_dict[p.name]
                found = True
            # 3. Match by component-dot-parameter key if param_meta is available
            elif param_meta:
                sym_repr = sp.srepr(p)
                if sym_repr in param_meta:
                    meta = param_meta[sym_repr]
                    comp_dot_name = f"{meta['component']}.{meta['name']}"
                    if comp_dot_name in param_dict:
                        val = param_dict[comp_dot_name]
                        found = True
                    elif 'default' in meta:
                        val = meta['default']
                        found = True
            
            if not found:
                raise ValueError(
                    f"Parameter '{p.name}' is missing and has no default value. "
                    f"Please specify it in your params dictionary."
                )
            flat_params.append(val)
            
        # Check if there is any batched parameter (i.e. if any value is an array/list of size > 1)
        def is_seq(v):
            if isinstance(v, (str, bytes)):
                return False
            try:
                len(v)
                return True
            except TypeError:
                return False
                
        is_batched = False
        batch_size = None
        for val in flat_params:
            if is_seq(val):
                is_batched = True
                batch_size = len(val)
                break
        
        # Detect if any parameter is a PyTorch tensor to preserve gradients/autograd graph
        has_torch_tensor = False
        for val in flat_params:
            if hasattr(val, 'requires_grad') or type(val).__name__ == 'Tensor':
                has_torch_tensor = True
                break
                
        if has_torch_tensor:
            import torch
            target_device = None
            target_dtype = torch.float64
            for val in flat_params:
                if isinstance(val, torch.Tensor):
                    target_device = val.device
                    target_dtype = val.dtype
                    break
            
            torch_params = []
            for val in flat_params:
                if isinstance(val, torch.Tensor):
                    torch_params.append(val.to(device=target_device, dtype=target_dtype))
                else:
                    torch_params.append(torch.tensor(val, device=target_device, dtype=target_dtype))
                    
            if is_batched:
                batched_tensors = []
                for val in torch_params:
                    if val.dim() > 0:
                        batched_tensors.append(val)
                    else:
                        batched_tensors.append(val.expand(batch_size))
                params = torch.stack(batched_tensors, dim=-1)
            else:
                params = torch.stack(torch_params)
        else:
            if is_batched:
                batched_list = []
                for val in flat_params:
                    if is_seq(val):
                        batched_list.append(np.array(val))
                    else:
                        batched_list.append(np.full(batch_size, val))
                params = np.column_stack(batched_list)
            else:
                params = np.array(flat_params)

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
        
    elif backend_lower == 'julia':
        from backends.julia_backend import simulate_julia
        return simulate_julia(dae, t_span, y0, params, method, device, **kwargs)
        
    elif backend_lower in ('c', 'sundials'):
        from backends.c_backend import simulate_c
        compile_c = kwargs.pop('compile_c', (backend_lower == 'c'))
        compiler_required = (backend_lower == 'c')
        return simulate_c(dae, t_span, y0, params, method, compile_c=compile_c, compiler_required=compiler_required, **kwargs)
        
    elif backend_lower == 'jax':
        raise ImportError(
            "JAX is not installed in the active environment. "
            "Please install JAX to use the 'jax' simulation backend."
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Supported backends: 'numpy', 'pytorch', 'julia', 'c', 'sundials'.")
