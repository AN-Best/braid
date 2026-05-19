import sys
import os
import sympy as sp
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.multibody_3D import World, RigidBody, FixedTranslation, RevoluteJoint
from base import System, Node
from index_reduction import order_reduction_pass
from backends.torch_backend import from_dae
from ode_solvers import rk4_rollout

def test_multibody_pendulum():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("==================================================")
    print("Testing 3D Multibody Simulation Toolbox")
    print(f"Target Device: {device.upper()}")
    print("==================================================")
    
    # 1. Build the 3D pendulum
    print("\n[Step 1] Assembling 3D Single Pendulum...")
    world = World('world')
    pivot = FixedTranslation('pivot', r_rel=[0.0, 0.0, 0.0]) # Pivot at origin
    # Joint rotates about world y-axis [0, 1, 0], swinging in the X-Z plane
    joint = RevoluteJoint('joint', axis=[0.0, 1.0, 0.0]) 
    # Link arm of length 1.0 pointing in local X direction
    arm   = FixedTranslation('arm', r_rel=[1.0, 0.0, 0.0])
    body  = RigidBody('body', m=1.5, Ixx=0.1, Iyy=0.1, Izz=0.1)

    system = System([world, pivot, joint, arm, body])
    
    Node(system, [(world, 'p'), (pivot, 'p1')])
    Node(system, [(pivot, 'p2'), (joint, 'p1')])
    Node(system, [(joint, 'p2'), (arm, 'p1')])
    Node(system, [(arm, 'p2'), (body, 'p')])

    # 2. DAE Conversion
    print("\n[Step 2] Assembling DAE System...")
    dae = system.to_dae()
    
    print(f"Total equations assembled: {len(dae.equations)}")
    print(f"Total states: {len(dae.states)}")

    # 3. Order Reduction (should pass through without changing states since it's already first-order)
    print("\n[Step 3] Running Order Reduction Pass...")
    reduced_dae = order_reduction_pass(dae)
    print(f"Post reduction equations: {len(reduced_dae.equations)}")
    print(f"Post reduction states: {len(reduced_dae.states)}")

    # 4. Torch Compilation
    print("\n[Step 4] Compiling DAE to PyTorch module...")
    torch_sys = from_dae(reduced_dae).to(device)
    
    print("Parsed Variables:")
    print("Differential States (x):", torch_sys.state_names)
    print("Parameters (p):", torch_sys.param_names)
    print(f"Total Algebraic Variables (z): {len(torch_sys.algebraic_names)}")
    
    # 5. Run Rollout with non-zero initial joint angle (phi = 0.5 rad)
    print("\n[Step 5] Simulating Pendulum Rollout...")
    # Initialize all states to zero
    x0 = torch.zeros(len(torch_sys.state_names), device=device)
    
    # Set the initial joint angle state `phi_joint` to 0.5 radians
    phi_idx = torch_sys.state_names.index('phi_joint')
    x0[phi_idx] = 0.5
    
    # We must set the initial rotation matrix R_body to match the joint angle!
    # With phi = 0.5 rad and rotation about y-axis:
    # R_body = R_y(0.5) = [[cos(0.5), 0, sin(0.5)], [0, 1, 0], [-sin(0.5), 0, cos(0.5)]]
    cos_phi = float(torch.cos(torch.tensor(0.5)))
    sin_phi = float(torch.sin(torch.tensor(0.5)))
    
    # Map rotation matrix elements in x0: R_body_i_j
    R_elements = {
        'R_body_0_0': cos_phi,  'R_body_0_1': 0.0, 'R_body_0_2': sin_phi,
        'R_body_1_0': 0.0,      'R_body_1_1': 1.0, 'R_body_1_2': 0.0,
        'R_body_2_0': -sin_phi, 'R_body_2_1': 0.0, 'R_body_2_2': cos_phi
    }
    
    for name, val in R_elements.items():
        idx = torch_sys.state_names.index(name)
        x0[idx] = val

    # Set the initial position state of the body `x_body`, `z_body` to match!
    # body position is r_body = r_pivot + R_body * r_rel = [cos(0.5), 0, -sin(0.5)]
    x_idx = torch_sys.state_names.index('x_body')
    z_idx = torch_sys.state_names.index('z_body')
    x0[x_idx] = cos_phi
    x0[z_idx] = -sin_phi

    # Default parameters vector
    p_val = torch.tensor(system.get_param_vector(), device=device)
    
    dt = 0.02
    num_steps = 50
    
    x_traj, z_traj = rk4_rollout(torch_sys, x0, p_val, dt, num_steps)
    print(f"Simulation trajectory shape: {x_traj.shape}")
    
    # Trace the joint angle trajectory
    phi_trajectory = x_traj[:, phi_idx].tolist()
    print("\nJoint angle (phi_joint) at select steps:")
    for step in [0, 10, 20, 30, 40, 50]:
        print(f"Step {step:02d} (t={step*dt:.2f}s): {phi_trajectory[step]:.4f} rad")
        
    # Check if the pendulum swings (the joint angle should decrease due to gravity, then swing back)
    # At t=0, phi = 0.5. Under gravity, it should swing downwards towards 0, then go negative.
    assert min(phi_trajectory) < 0.2, "Pendulum should swing down under gravity"
    
    # 6. Verify backpropagation through the simulation
    print("\n[Step 6] Verifying backpropagation through 3D simulation...")
    x0_ref = x0.clone().detach().requires_grad_(True)
    p_ref = p_val.clone().detach().requires_grad_(True)
    
    x_traj_batch, z_traj_batch = rk4_rollout(torch_sys, x0_ref, p_ref, dt, num_steps)
    loss = x_traj_batch[-1].pow(2).sum()
    loss.backward()
    
    print("Gradient w.r.t initial state x0:", x0_ref.grad[:5]) # Show first 5 gradients
    print("Gradient w.r.t parameters p:", p_ref.grad[:5]) # Show first 5 gradients
    
    assert x0_ref.grad is not None
    assert p_ref.grad is not None
    assert torch.any(x0_ref.grad != 0.0)
    
    print("\n[SUCCESS] 3D Multibody pendulum assembly, simulation, and backpropagation verified successfully!")
    print("==================================================")

if __name__ == "__main__":
    test_multibody_pendulum()
