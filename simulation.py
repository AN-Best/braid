"""
simulation.py
=============
CasADi-based simulation backend interface for Braid.
Provides `simulate_system` which compiles a DAE system into IR,
lowers it to a NumPy/PyTorch backend, and solves it.
"""

import os
import numpy as np
import ir
from lowering import (
    make_numpy_ode,
    make_numpy_jacobian,
    make_torch_ode,
    make_torch_jacobian,
)


def simulate_system(dae, t_span, y0, params, backend='numpy', method=None, device=None, **kwargs):
    """
    Integrates the DAE system using the selected backend and method.

    Parameters:
        dae: A CasadiDAE instance, a Braid IR dict, a JSON file path, or a JSON string.
        t_span: Tuple of (t0, tf).
        y0: List/array of initial state values.
        params: Dict of {name: value} or ordered list/array.
        backend: String ('numpy' or 'pytorch' [alias: 'torch']).
        method: Solver method.
        device: The PyTorch device (for 'pytorch' backend).
        kwargs: Extra config (e.g. num_steps=1000).
    """
    # ── 1. Resolve IR ────────────────────────────────────────────────────────
    if isinstance(dae, (str, os.PathLike)):
        dae_resolved = ir.from_json(dae)
    elif isinstance(dae, dict):
        dae_resolved = dae
    else:
        # It's a CasadiDAE or System
        from base import System
        if isinstance(dae, System):
            dae_resolved = ir.compile_to_ir(dae)
        else:
            # Assume it's a CasadiDAE
            from index_reduction import pantelides_pass, tearing_pass
            dae = pantelides_pass(dae)
            dae = tearing_pass(dae)
            dae_resolved = ir.from_json(ir.to_json(dae))

    # ── 2. Resolve Parameters ────────────────────────────────────────────────
    # Resolve parameters if passed as a dictionary or None (to use defaults)
    if isinstance(params, dict) or params is None:
        param_dict = params or {}
        flat_params = []
        param_meta = dae_resolved.get("param_meta", {})
        
        for p_name in dae_resolved['params']:
            val = None
            found = False
            
            # Match options:
            # 1. Match by name string key
            if p_name in param_dict:
                val = param_dict[p_name]
                found = True
            # 2. Match by component-dot-parameter key if param_meta is available
            elif param_meta:
                if p_name in param_meta:
                    meta = param_meta[p_name]
                    comp_dot_name = f"{meta['component']}.{meta['param']}"
                    if comp_dot_name in param_dict:
                        val = param_dict[comp_dot_name]
                        found = True
                    elif 'default' in meta:
                        val = meta['default']
                        found = True
            
            if not found:
                raise ValueError(
                    f"Parameter '{p_name}' is missing and has no default value. "
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

    # ── 3. Validate Model Type ────────────────────────────────────────────────
    # These backends only support pure explicit ODE models.
    model_type = dae_resolved.get('model_type', 'ODE')
    if model_type != 'ODE':
        raise ValueError(
            f"The resolved model has model_type='{model_type}' (DAE). "
            f"These simulation solvers only support pure explicit ODE models. "
            f"Please compile the system to an explicit ODE first."
        )

    backend_lower = backend.lower()
    
    if backend_lower in ('numpy', 'scipy'):
        from backends.numpy_backend import simulate_numpy
        ode_func = make_numpy_ode(dae_resolved)
        jac_func = make_numpy_jacobian(dae_resolved)
        return simulate_numpy(ode_func, jac_func, t_span, y0, params, method, **kwargs)
        
    elif backend_lower in ('pytorch', 'torch'):
        from backends.torch_backend import simulate_torch
        ode_func = make_torch_ode(dae_resolved)
        jac_func = make_torch_jacobian(dae_resolved)
        return simulate_torch(ode_func, jac_func, t_span, y0, params, method, device, **kwargs)
        
    elif backend_lower == 'julia':
        from backends.julia_backend import simulate_julia
        return simulate_julia(dae_resolved, t_span, y0, params, method, **kwargs)
        
    else:
        raise ValueError(f"Unknown backend '{backend}'. Supported backends: 'numpy', 'pytorch' (alias: 'torch'), 'julia'.")
