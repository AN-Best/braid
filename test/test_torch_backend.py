import sys
import os
import sympy as sp
import torch

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.linear_mechanical_1D import Mass, Spring, Damper, Ground
from base import System, Node
from index_reduction import order_reduction_pass
from backends.torch_backend import from_dae
from ode_solvers import solve_algebraic_step, rk4_rollout

def test_torch_backend_compilation():
    # Detect GPU/CUDA support dynamically
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("==================================================")
    print("Testing PyTorch Backend DAE Compilation & Execution")
    print(f"Target Device: {device.upper()}")
    print("==================================================")
    
    # 1. Recreate the Mass-Spring-Damper system
    print("\n[Step 1] Building Mass-Spring-Damper system...")
    mass   = Mass('mass', m=2.0)
    spring = Spring('spring', k=10.0)
    damper = Damper('damper', c=0.2)
    ground = Ground('ground')

    system = System([mass, spring, damper, ground])
    Node(system, [(mass, 'p'), (spring, 'p2'), (damper, 'p2')])
    Node(system, [(ground, 'p'), (spring, 'p1'), (damper, 'p1')])
    
    dae = system.to_dae()
    reduced_dae = order_reduction_pass(dae)
    
    # 2. Compile to PyTorch and move to device
    print("\n[Step 2] Compiling DAE to PyTorch module...")
    torch_sys = from_dae(reduced_dae).to(device)
    
    # 3. Verify variable parsing
    print("\nParsed Variables:")
    print("Differential States (x):", torch_sys.state_names)
    print("State Derivatives (x_dot):", torch_sys.state_dot_names)
    print("Algebraic Variables (z):", torch_sys.algebraic_names)
    print("Parameters (p):", torch_sys.param_names)
    
    assert len(torch_sys.state_names) == 2, f"Expected 2 states, got {len(torch_sys.state_names)}"
    assert len(torch_sys.state_dot_names) == 2, f"Expected 2 state derivatives, got {len(torch_sys.state_dot_names)}"
    assert len(torch_sys.algebraic_names) == 16, f"Expected 16 algebraic variables, got {len(torch_sys.algebraic_names)}"
    assert len(torch_sys.param_names) == 3, f"Expected 3 parameters, got {len(torch_sys.param_names)}"
    
    # 4. Evaluate single input
    print("\n[Step 3] Evaluating system residual with single input...")
    t = torch.tensor(0.0, device=device, requires_grad=True)
    # x = [x_mass, x_mass_dot]
    x = torch.tensor([1.0, 0.5], device=device, requires_grad=True)
    # x_dot = [d_x_mass, d_x_mass_dot]
    x_dot = torch.tensor([0.5, -2.5], device=device, requires_grad=True)
    # z = algebraic variables
    z = torch.zeros(len(torch_sys.algebraic_names), device=device, requires_grad=True)
    # p = params (m_mass, k_spring, c_damper)
    p = torch.tensor(system.get_param_vector(), device=device, requires_grad=True)
    
    res = torch_sys(t, x, x_dot, z, p)
    print("Single residual value:\n", res)
    assert res.shape == (len(reduced_dae.equations),), f"Expected residual shape {(len(reduced_dae.equations),)}, got {res.shape}"
    
    # 5. Verify autograd (backward pass)
    print("\n[Step 4] Verifying Autograd / Backpropagation compatibility...")
    loss = res.pow(2).sum()
    loss.backward()
    
    print("Gradient w.r.t x:", x.grad)
    print("Gradient w.r.t x_dot:", x_dot.grad)
    print("Gradient w.r.t z:", z.grad)
    print("Gradient w.r.t p:", p.grad)
    
    assert x.grad is not None, "Expected non-None gradient for x"
    assert x_dot.grad is not None, "Expected non-None gradient for x_dot"
    assert z.grad is not None, "Expected non-None gradient for z"
    assert p.grad is not None, "Expected non-None gradient for p"
    
    # 6. Verify batched evaluation
    print("\n[Step 5] Evaluating system residual with batched inputs...")
    batch_size = 8
    t_batch = torch.linspace(0.0, 1.0, batch_size, device=device)
    x_batch = torch.randn(batch_size, len(torch_sys.state_names), device=device)
    x_dot_batch = torch.randn(batch_size, len(torch_sys.state_dot_names), device=device)
    z_batch = torch.randn(batch_size, len(torch_sys.algebraic_names), device=device)
    p_batch = p.detach().repeat(batch_size, 1).to(device)
    
    res_batch = torch_sys(t_batch, x_batch, x_dot_batch, z_batch, p_batch)
    print("Batched residual shape:", res_batch.shape)
    assert res_batch.shape == (batch_size, len(reduced_dae.equations)), f"Expected batched residual shape {(batch_size, len(reduced_dae.equations))}, got {res_batch.shape}"
    
    # 7. Verify Numerical RK4 DAE Rollout
    print("\n[Step 6] Verifying Numerical RK4 DAE Rollout...")
    x0 = torch.tensor([1.0, 0.0], device=device)  # Initial: position = 1.0, velocity = 0.0
    p_val = torch.tensor(system.get_param_vector(), device=device)  # Default params
    dt = 0.05
    num_steps = 40
    
    x_traj, z_traj = rk4_rollout(torch_sys, x0, p_val, dt, num_steps)
    print("Simulation Rollout Shape:", x_traj.shape)
    assert x_traj.shape == (num_steps + 1, len(torch_sys.state_names)), f"Expected shape {(num_steps + 1, len(torch_sys.state_names))}, got {x_traj.shape}"
    assert z_traj.shape == (num_steps, len(torch_sys.algebraic_names)), f"Expected shape {(num_steps, len(torch_sys.algebraic_names))}, got {z_traj.shape}"
    
    print("Initial state (t=0.00):", x_traj[0].tolist())
    print("State at t=0.50 (step 10):", x_traj[10].tolist())
    print("Final state (t=2.00):", x_traj[-1].tolist())
    
    # Assert physical correctness: mass should oscillate and return closer to origin 0.0 due to damping
    assert abs(x_traj[-1, 0].item()) < 1.2, "Expected spring to pull mass closer to the ground equilibrium"
    
    # 8. Verify Differentiable Batched Rollouts (Holy Grail for RL/Control)
    print("\n[Step 7] Verifying Differentiable Batched Rollouts...")
    batch_size_rollout = 4
    
    # Enable gradients on initial state and parameters
    x0_ref = torch.tensor([1.0, 0.0], device=device, requires_grad=True)
    p_ref = torch.tensor(system.get_param_vector(), device=device, requires_grad=True)
    
    # Create batched inputs that are derived from x0_ref and p_ref to trace gradients
    x0_batch = x0_ref.unsqueeze(0).repeat(batch_size_rollout, 1) * torch.tensor([[1.0], [1.5], [2.0], [2.5]], device=device)
    p_batch = p_ref.unsqueeze(0).repeat(batch_size_rollout, 1)
    
    x_traj_batch, z_traj_batch = rk4_rollout(torch_sys, x0_batch, p_batch, dt, num_steps)
    
    print("Batched state trajectory shape:", x_traj_batch.shape)
    assert x_traj_batch.shape == (num_steps + 1, batch_size_rollout, len(torch_sys.state_names))
    
    # Backpropagate through the entire simulation trajectory
    loss_rollout = x_traj_batch[-1].pow(2).sum()
    loss_rollout.backward()
    
    print("Gradient w.r.t initial state x0:", x0_ref.grad)
    print("Gradient w.r.t parameters p:", p_ref.grad)
    
    assert x0_ref.grad is not None, "Expected non-None gradient for x0_ref"
    assert p_ref.grad is not None, "Expected non-None gradient for p_ref"
    assert torch.any(x0_ref.grad != 0.0), "Expected non-zero gradient flow back to initial state"
    assert torch.any(p_ref.grad != 0.0), "Expected non-zero gradient flow back to parameters"
    
    print("\n[SUCCESS] All PyTorch backend compilation and numerical rollout tests passed!")
    print("==================================================")

if __name__ == "__main__":
    test_torch_backend_compilation()
