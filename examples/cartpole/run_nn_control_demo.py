import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import casadi as ca

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from base import System
from index_reduction import pantelides_pass, tearing_pass
from simulation import simulate_system
from components.neural_net import NeuralNetworkPyTorch
from examples.cartpole.cartpole_model import get_cartpole_component
from examples.cartpole.cartpole_lqr import compute_lqr_gain
from examples.cartpole.animate import animate_cartpole

def main():
    print("==================================================")
    print("      Braid Neural Network Control Demo           ")
    print("==================================================")
    
    # 1. Compute LQR gain K to use as the target expert policy
    K = compute_lqr_gain()
    
    # 2. Generate training data by sampling states near the upright position
    print("Generating training data from LQR controller...")
    np.random.seed(42)
    num_samples = 2000
    
    # Sample state: [x, theta, xdot, thetadot]
    X_train = np.random.uniform(
        low=[-1.0, -0.3, -1.0, -1.0],
        high=[1.0, 0.3, 1.0, 1.0],
        size=(num_samples, 4)
    ).astype(np.float32)
    
    # target force: F = - K * state
    Y_train = (-X_train @ K.T).astype(np.float32)
    
    # 3. Train a PyTorch neural network to clone this policy
    print("Training PyTorch MLP policy network...")
    mlp = nn.Sequential(
        nn.Linear(4, 8),
        nn.Tanh(),
        nn.Linear(8, 8),
        nn.Tanh(),
        nn.Linear(8, 1)
    )
    
    X_tensor = torch.from_numpy(X_train)
    Y_tensor = torch.from_numpy(Y_train)
    
    optimizer = optim.Adam(mlp.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Fast training loop
    mlp.train()
    for epoch in range(300):
        optimizer.zero_grad()
        outputs = mlp(X_tensor)
        loss = criterion(outputs, Y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/300 | Loss: {loss.item():.6f}")
            
    mlp.eval()
    
    # 4. Create the Neural Network component and Braid system
    print("Wrapping PyTorch model in Braid NeuralNetworkPyTorch component...")
    nn_comp = NeuralNetworkPyTorch(
        name="nn_policy",
        pytorch_model=mlp,
        input_names=["x", "theta", "u1", "u2"],
        output_names=["F"]
    )
    
    cartpole = get_cartpole_component()
    
    # Assemble closed-loop system
    system = System([cartpole, nn_comp])
    
    # Connect cartpole states to NN inputs
    system.equations.append(nn_comp.ports["x"][1] - cartpole.states[0])
    system.equations.append(nn_comp.ports["theta"][1] - cartpole.states[1])
    system.equations.append(nn_comp.ports["u1"][1] - cartpole.states[2])
    system.equations.append(nn_comp.ports["u2"][1] - cartpole.states[3])
    
    # Connect NN output force to cartpole force input port
    system.equations.append(cartpole.ports["F"][0] - nn_comp.ports["F"][0])
    
    # Compile closed-loop DAE
    print("Compiling closed-loop DAE system...")
    dae = system.to_dae()
    red = pantelides_pass(dae)
    torn = tearing_pass(red)
    
    # 5. Simulate the closed-loop system controlled by the Neural Network
    x0 = [0.0, 0.25, 0.0, 0.0]  # Tilted cartpole start
    t_max = 5.0
    print(f"Simulating closed-loop system under NN control starting from x0={x0}...")
    sol = simulate_system(torn, (0.0, t_max), x0, params=None, backend='numpy', method='RK45')
    
    # Extract trajectories
    x_idx = torn.state_names.index("q_cartpole_x")
    theta_idx = torn.state_names.index("q_cartpole_theta")
    
    x_traj = sol.y[x_idx]
    theta_traj = sol.y[theta_idx]
    
    # 6. Generate and save the animation
    output_gif = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cartpole_nn_control.gif")
    animate_cartpole(
        sol.t, x_traj, theta_traj, l=0.5, save_path=output_gif,
        title="Cartpole Neural Network Control (Braid + PyTorch)"
    )
    print("==================================================")
    print("NN Control Demo Completed Successfully!")
    print(f"Visual animation saved as: {output_gif}")
    print("==================================================")

if __name__ == "__main__":
    main()
