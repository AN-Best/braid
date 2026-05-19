import sys
import os
import sympy as sp
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.rotational_mechanical_1D import Inertia, TorsionalSpring, RotationalDamper, Fixed, Torque
from base import System, Node
from index_reduction import order_reduction_pass
from backends.torch_backend import from_dae
from ode_solvers import solve_algebraic_step, rk4_rollout

def test_rotational_torch_compilation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("==================================================")
    print("Testing PyTorch Backend for 1D Rotational System")
    print(f"Target Device: {device.upper()}")
    print("==================================================")
    
    inertia = Inertia('inertia', J=2.5)
    spring  = TorsionalSpring('spring', k=15.0)
    damper  = RotationalDamper('damper', c=0.5)
    fixed   = Fixed('fixed')
    torque  = Torque('torque', tau=10.0)

    # Assemble system
    system = System([inertia, spring, damper, fixed, torque])
    
    Node(system, [(fixed, 'p'), (spring, 'p1'), (damper, 'p1')])
    Node(system, [(inertia, 'p'), (spring, 'p2'), (damper, 'p2'), (torque, 'p')])

    dae = system.to_dae()
    reduced_dae = order_reduction_pass(dae)
    
    print("\n[Step 1] Compiling Rotational DAE to PyTorch module...")
    torch_sys = from_dae(reduced_dae).to(device)
    
    print("\nParsed Variables:")
    print("Differential States (x):", torch_sys.state_names)
    print("State Derivatives (x_dot):", torch_sys.state_dot_names)
    print("Algebraic Variables (z):", torch_sys.algebraic_names)
    print("Parameters (p):", torch_sys.param_names)
    
    assert len(torch_sys.state_names) == 2, f"Expected 2 states, got {len(torch_sys.state_names)}"
    assert len(torch_sys.state_dot_names) == 2, f"Expected 2 state derivatives, got {len(torch_sys.state_dot_names)}"
    assert len(torch_sys.param_names) == 4, f"Expected 4 parameters (J_inertia, k_spring, c_damper, tau_torque), got {len(torch_sys.param_names)}"
    
    print("\n[Step 2] Verifying Numerical RK4 DAE Rollout...")
    x0 = torch.tensor([1.0, 0.0], device=device)  # Initial: position = 1.0, velocity = 0.0
    p_val = torch.tensor(system.get_param_vector(), device=device)  # Default params
    dt = 0.05
    num_steps = 40
    
    x_traj, z_traj = rk4_rollout(torch_sys, x0, p_val, dt, num_steps)
    print("Simulation Rollout Shape:", x_traj.shape)
    assert x_traj.shape == (num_steps + 1, len(torch_sys.state_names))
    
    print("Initial state (t=0.00):", x_traj[0].tolist())
    print("State at t=0.50 (step 10):", x_traj[10].tolist())
    print("Final state (t=2.00):", x_traj[-1].tolist())
    
    # Verify autograd compatibility
    print("\n[Step 3] Verifying Differentiable Batched Rollouts...")
    x0_ref = torch.tensor([1.0, 0.0], device=device, requires_grad=True)
    p_ref = torch.tensor(system.get_param_vector(), device=device, requires_grad=True)
    
    x0_batch = x0_ref.unsqueeze(0).repeat(2, 1)
    p_batch = p_ref.unsqueeze(0).repeat(2, 1)
    
    x_traj_batch, z_traj_batch = rk4_rollout(torch_sys, x0_batch, p_batch, dt, num_steps)
    loss = x_traj_batch[-1].pow(2).sum()
    loss.backward()
    
    print("Gradient w.r.t initial state x0:", x0_ref.grad)
    print("Gradient w.r.t parameters p:", p_ref.grad)
    
    assert x0_ref.grad is not None
    assert p_ref.grad is not None
    assert torch.any(x0_ref.grad != 0.0)
    assert torch.any(p_ref.grad != 0.0)
    
    print("\n[SUCCESS] Rotational PyTorch simulation and backpropagation tests passed!")
    print("==================================================")

if __name__ == "__main__":
    test_rotational_torch_compilation()
