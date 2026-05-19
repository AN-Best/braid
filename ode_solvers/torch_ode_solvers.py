import torch

def ensure_tensor(val, device=None, dtype=torch.float32):
    if val is None:
        return torch.empty(0, device=device, dtype=dtype)
    if not isinstance(val, torch.Tensor):
        val = torch.tensor(val, device=device, dtype=dtype)
    return val

def _stack_and_reshape(flat_res, target_shape, batch_shape, dtype, device):
    """Standardizes types, broadcasts constants, stacks, and reshapes symbolic matrices."""
    tensors = []
    for r in flat_res:
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=dtype, device=device)
        else:
            r = r.to(dtype=dtype, device=device)
            
        # Broadcast constant or unbatched inputs to matching batch shape
        if r.shape != batch_shape:
            try:
                r = r.expand(batch_shape)
            except RuntimeError:
                r = torch.broadcast_to(r, batch_shape)
        tensors.append(r)
        
    # Stack along flat dimension
    stacked = torch.stack(tensors, dim=-1)
    
    # Reshape to target shape
    return stacked.reshape(list(batch_shape) + list(target_shape))

def solve_algebraic_step(system, t, x, p):
    """Solve the implicit system algebraic constraints for state derivatives and algebraic variables.
    
    Uses a batched linear least-squares solver (pseudo-inverse via QR/SVD decomposition)
    to solve J * V = -b where V = [x_dot, z].
    
    Args:
        system (TorchSystem): The compiled PyTorch system.
        t (Tensor or float): Time.
        x (Tensor): State vector of shape (..., num_states).
        p (Tensor): Parameter vector of shape (..., num_params).
        
    Returns:
        x_dot (Tensor): Differential state derivatives of shape (..., num_states).
        z (Tensor): Algebraic variables of shape (..., num_algebraics).
    """
    ref_tensor = None
    for val in (x, p):
        if isinstance(val, torch.Tensor):
            ref_tensor = val
            break
            
    device = ref_tensor.device if ref_tensor is not None else None
    dtype = ref_tensor.dtype if ref_tensor is not None else torch.float32
    
    t_t = ensure_tensor(t, device, dtype)
    x_t = ensure_tensor(x, device, dtype)
    p_t = ensure_tensor(p, device, dtype)
    
    # Extract current batch shape
    batch_shape = torch.Size([])
    for val in (x_t, p_t):
        if val.dim() >= 1:
            batch_shape = val.shape[:-1]
            break
            
    # Unpack arguments for Jacobian and constant evaluators
    args = [t_t]
    for i in range(len(system.state_names)):
        args.append(x_t[..., i])
    for i in range(len(system.param_names)):
        args.append(p_t[..., i])
        
    # Evaluate flat symbolic formulas
    J_flat = system.jac_func(*args)
    b_flat = system.b_func(*args)
    
    # Stack into correct batched tensors
    J = _stack_and_reshape(J_flat, (system.num_equations, system.num_variables), batch_shape, dtype, device)
    b = _stack_and_reshape(b_flat, (system.num_equations,), batch_shape, dtype, device)
    
    # Solve batched J * V = -b using least-squares solver (highly robust for rectangular DAEs)
    b_unsqueezed = b.unsqueeze(-1)
    sol = torch.linalg.lstsq(J, -b_unsqueezed).solution
    V = sol.squeeze(-1) # shape: (..., num_variables)
    
    # Unpack V into state derivatives (x_dot) and algebraic variables (z)
    num_states = len(system.state_names)
    x_dot = V[..., :num_states]
    z = V[..., num_states:]
    
    return x_dot, z

def rk4_rollout(system, x0, p, dt, num_steps):
    """Run a batched numerical DAE simulation rollout using Runge-Kutta 4 (RK4) integration.
    
    Args:
        system (TorchSystem): The compiled PyTorch system.
        x0 (Tensor): Initial states of shape (..., num_states).
        p (Tensor): System parameters of shape (..., num_params).
        dt (float): Integrator time step.
        num_steps (int): Number of steps to simulate.
        
    Returns:
        x_traj (Tensor): State trajectory of shape (num_steps + 1, ..., num_states).
        z_traj (Tensor): Algebraic variable trajectory of shape (num_steps, ..., num_algebraics).
    """
    device = x0.device if isinstance(x0, torch.Tensor) else None
    dtype = x0.dtype if isinstance(x0, torch.Tensor) else torch.float32
    
    x = ensure_tensor(x0, device, dtype)
    p = ensure_tensor(p, device, dtype)
    dt_t = ensure_tensor(dt, device, dtype)
    
    x_list = [x]
    z_list = []
    
    t = 0.0
    for _ in range(num_steps):
        # Solve algebraic step to get current derivative and algebraic states
        x_dot, z = solve_algebraic_step(system, t, x, p)
        z_list.append(z)
        
        # RK4 Integration steps
        k1 = x_dot
        
        k2_dot, _ = solve_algebraic_step(system, t + dt_t / 2.0, x + (dt_t / 2.0) * k1, p)
        k2 = k2_dot
        
        k3_dot, _ = solve_algebraic_step(system, t + dt_t / 2.0, x + (dt_t / 2.0) * k2, p)
        k3 = k3_dot
        
        k4_dot, _ = solve_algebraic_step(system, t + dt_t, x + dt_t * k3, p)
        k4 = k4_dot
        
        # Update states and time
        x = x + (dt_t / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + dt
        
        x_list.append(x)
        
    x_traj = torch.stack(x_list, dim=0)
    z_traj = torch.stack(z_list, dim=0)
    
    return x_traj, z_traj

def euler_rollout(system, x0, p, dt, num_steps):
    """Run a batched numerical DAE simulation rollout using explicit Forward Euler integration.
    
    Args:
        system (TorchSystem): The compiled PyTorch system.
        x0 (Tensor): Initial states of shape (..., num_states).
        p (Tensor): System parameters of shape (..., num_params).
        dt (float): Integrator time step.
        num_steps (int): Number of steps to simulate.
        
    Returns:
        x_traj (Tensor): State trajectory of shape (num_steps + 1, ..., num_states).
        z_traj (Tensor): Algebraic variable trajectory of shape (num_steps, ..., num_algebraics).
    """
    device = x0.device if isinstance(x0, torch.Tensor) else None
    dtype = x0.dtype if isinstance(x0, torch.Tensor) else torch.float32
    
    x = ensure_tensor(x0, device, dtype)
    p = ensure_tensor(p, device, dtype)
    dt_t = ensure_tensor(dt, device, dtype)
    
    x_list = [x]
    z_list = []
    
    t = 0.0
    for _ in range(num_steps):
        # Solve algebraic step to get current derivative and algebraic states
        x_dot, z = solve_algebraic_step(system, t, x, p)
        z_list.append(z)
        
        # Euler integration step
        x = x + dt_t * x_dot
        t = t + dt
        
        x_list.append(x)
        
    x_traj = torch.stack(x_list, dim=0)
    z_traj = torch.stack(z_list, dim=0)
    
    return x_traj, z_traj
