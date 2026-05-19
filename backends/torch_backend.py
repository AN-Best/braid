import sympy as sp
import torch

class TorchSystem(torch.nn.Module):
    """A PyTorch module representing an assembled physical system DAE.
    
    Computes the residual of the system equations:
        residual(t, x, x_dot, z, p) = 0
    
    Attributes:
        state_names (list of str): Names of the differential states (x)
        state_dot_names (list of str): Names of the state derivatives (x_dot)
        algebraic_names (list of str): Names of the algebraic variables (z)
        param_names (list of str): Names of the parameters (p)
    """
    def __init__(self, dae, state_symbols, state_dot_symbols, algebraic_symbols, param_symbols, equations, jac_func, b_func):
        super().__init__()
        self.state_names = [s.name for s in state_symbols]
        self.state_dot_names = [s.name for s in state_dot_symbols]
        self.algebraic_names = [s.name for s in algebraic_symbols]
        self.param_names = [p.name for p in param_symbols]
        
        # Build the ordered list of all input symbols for lambdify
        self.all_symbols = [dae.t] + state_symbols + state_dot_symbols + algebraic_symbols + param_symbols
        self.t_sym = dae.t
        
        # Lambdify the equations with PyTorch backend
        self.func = sp.lambdify(self.all_symbols, equations, modules="torch")
        
        # Save Jacobian and constant term evaluators for numerical solving
        self.jac_func = jac_func
        self.b_func = b_func
        
        self.num_equations = len(equations)
        self.num_variables = len(state_dot_symbols) + len(algebraic_symbols)

    def forward(self, t, x, x_dot, z, p):
        """Evaluate the DAE residuals.
        
        Args:
            t (Tensor or float): Time.
            x (Tensor): Differential states of shape (..., num_states).
            x_dot (Tensor): State derivatives of shape (..., num_states).
            z (Tensor): Algebraic states of shape (..., num_algebraics).
            p (Tensor): Parameters of shape (..., num_params).
            
        Returns:
            Tensor of shape (..., num_equations) containing the residuals.
        """
        # Determine device and dtype from torch inputs to ensure consistency
        ref_tensor = None
        for val in (x, x_dot, z, p):
            if isinstance(val, torch.Tensor):
                ref_tensor = val
                break
        
        device = ref_tensor.device if ref_tensor is not None else None
        dtype = ref_tensor.dtype if ref_tensor is not None else torch.float32
        
        # Convert all inputs to tensors
        t_t = self._ensure_tensor(t, device, dtype)
        x_t = self._ensure_tensor(x, device, dtype)
        x_dot_t = self._ensure_tensor(x_dot, device, dtype)
        z_t = self._ensure_tensor(z, device, dtype)
        p_t = self._ensure_tensor(p, device, dtype)
        
        # Determine the target batch shape (all dimensions except the last one)
        batch_shape = torch.Size([])
        for val in (x_t, x_dot_t, z_t, p_t):
            if val.dim() >= 1:
                batch_shape = val.shape[:-1]
                break
                
        # Unpack variables for lambdify
        args = [t_t]
        
        # Unpack x
        for i in range(len(self.state_names)):
            args.append(x_t[..., i])
            
        # Unpack x_dot
        for i in range(len(self.state_dot_names)):
            args.append(x_dot_t[..., i])
            
        # Unpack z
        for i in range(len(self.algebraic_names)):
            args.append(z_t[..., i])
            
        # Unpack p
        for i in range(len(self.param_names)):
            args.append(p_t[..., i])
            
        # Call the lambdified function
        res = self.func(*args)
        
        # Ensure we always work with a tuple or list of residuals
        if not isinstance(res, (list, tuple)):
            res = [res]
            
        # Standardize and broadcast residuals to the target batch shape
        tensor_res = []
        for r in res:
            if not isinstance(r, torch.Tensor):
                r = torch.tensor(r, dtype=dtype, device=device)
            else:
                r = r.to(dtype=dtype, device=device)
            
            # Broadcast to batch shape if needed
            if r.shape != batch_shape:
                try:
                    r = r.expand(batch_shape)
                except RuntimeError:
                    r = torch.broadcast_to(r, batch_shape)
            tensor_res.append(r)
            
        # Stack along the last dimension to get (..., num_equations)
        return torch.stack(tensor_res, dim=-1)

    def _ensure_tensor(self, val, device, dtype):
        if val is None:
            return torch.empty(0, device=device, dtype=dtype)
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val, device=device, dtype=dtype)
        return val

def from_dae(dae) -> TorchSystem:
    """Compile a SystemDAE into a runnable TorchSystem.
    
    Args:
        dae (SystemDAE): The system DAE to compile.
        
    Returns:
        TorchSystem: A PyTorch module for the DAE.
    """
    # 1. Find all Derivative expressions in the equations
    all_derivs = set()
    for eq in dae.equations:
        all_derivs.update(eq.find(sp.Derivative))

    # 2. Map state derivatives and algebraic derivatives to simple symbols
    state_dot_symbols = []
    state_dot_subs = {}
    
    other_deriv_symbols = []
    other_deriv_subs = {}
    
    for s in dae.states:
        name = s.func.__name__
        deriv_sym = sp.Symbol(f"d_{name}")
        state_dot_symbols.append(deriv_sym)
        state_dot_subs[sp.Derivative(s, dae.t)] = deriv_sym
        
    for deriv in all_derivs:
        # Check if this derivative is already mapped as a state derivative
        if deriv in state_dot_subs:
            continue
            
        # It's a derivative of an algebraic function! E.g. Derivative(x1_spring(t), t)
        func = deriv.expr
        if isinstance(func, sp.core.function.AppliedUndef):
            name = func.func.__name__
            deriv_sym = sp.Symbol(f"d_{name}")
            other_deriv_symbols.append(deriv_sym)
            other_deriv_subs[deriv] = deriv_sym
        else:
            # Fallback for generic expressions inside Derivative
            clean_func_str = str(func).replace('(', '_').replace(')', '_').replace(' ', '_').replace(',', '_')
            deriv_sym = sp.Symbol(f"d_{clean_func_str}")
            other_deriv_symbols.append(deriv_sym)
            other_deriv_subs[deriv] = deriv_sym

    # 3. States
    state_symbols = []
    state_subs = {}
    for s in dae.states:
        name = s.func.__name__
        state_sym = sp.Symbol(name)
        state_symbols.append(state_sym)
        state_subs[s] = state_sym

    # 4. Find all other functions (algebraics)
    all_funcs = set()
    for eq in dae.equations:
        all_funcs.update(eq.find(sp.core.function.AppliedUndef))

    algebraic_funcs = [f for f in all_funcs if f not in dae.states]
    
    algebraic_symbols = []
    algebraic_subs = {}
    for f in algebraic_funcs:
        name = f.func.__name__
        alg_sym = sp.Symbol(name)
        algebraic_symbols.append(alg_sym)
        algebraic_subs[f] = alg_sym

    # Add the other derivatives as algebraic symbols too!
    algebraic_symbols.extend(other_deriv_symbols)
    
    # Sort algebraic symbols alphabetically by name for deterministic ordering
    algebraic_symbols.sort(key=lambda s: s.name)

    # 5. Substitute equations
    replaced_equations = []
    for eq in dae.equations:
        # First substitute state derivatives and algebraic derivatives
        eq_replaced = eq.subs(state_dot_subs)
        eq_replaced = eq_replaced.subs(other_deriv_subs)
        
        # Then substitute states and algebraic functions
        eq_replaced = eq_replaced.subs(state_subs)
        eq_replaced = eq_replaced.subs(algebraic_subs)
        
        replaced_equations.append(eq_replaced)

    # 6. Parameters
    param_symbols = list(dae.params)

    # 7. Extract flat Jacobian and constant term symbolically
    V_symbols = state_dot_symbols + algebraic_symbols
    
    eq_matrix = sp.Matrix(replaced_equations)
    J_matrix = eq_matrix.jacobian(V_symbols)
    b_matrix = eq_matrix.subs({v: 0 for v in V_symbols})
    
    flat_J = list(J_matrix)
    flat_b = list(b_matrix)
    
    jac_b_inputs = [dae.t] + state_symbols + param_symbols
    
    jac_func = sp.lambdify(jac_b_inputs, flat_J, modules="torch")
    b_func = sp.lambdify(jac_b_inputs, flat_b, modules="torch")

    return TorchSystem(
        dae=dae,
        state_symbols=state_symbols,
        state_dot_symbols=state_dot_symbols,
        algebraic_symbols=algebraic_symbols,
        param_symbols=param_symbols,
        equations=replaced_equations,
        jac_func=jac_func,
        b_func=b_func
    )
