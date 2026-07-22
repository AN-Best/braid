import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Turbine geometry and environmental constants
R = 40.0         # Rotor radius (m)
RHO = 1.225      # Air density (kg/m^3)

def compute_analytical_torque(v_wind, omega):
    """
    Computes analytical aerodynamic torque based on standard Cp(lambda) curve.
    """
    # Avoid division by zero
    v_wind = np.maximum(v_wind, 1e-3)
    omega = np.maximum(omega, 1e-3)
    
    # Tip-speed ratio lambda
    lam = (omega * R) / v_wind
    
    # Gaussian power coefficient Cp(lambda) peaking at lambda = 7.0
    cp = 0.48 * np.exp(-0.1 * (lam - 7.0)**2)
    
    # Torque: tau = 0.5 * rho * pi * R^3 * (Cp / lambda) * v_wind^2
    # Simplify Cp / lambda
    eff_coef = cp / np.maximum(lam, 0.1)
    tau = 0.5 * RHO * np.pi * (R**3) * eff_coef * (v_wind**2)
    return tau

def generate_aerodynamic_data(num_samples=2500):
    """
    Generates training data of [v_wind, omega] -> [tau_aero]
    """
    np.random.seed(42)
    
    # Sample wind speed from 4 m/s to 16 m/s
    v_wind = np.random.uniform(4.0, 16.0, (num_samples, 1))
    
    # Sample rotor speed from 0.1 rad/s to 3.0 rad/s
    omega = np.random.uniform(0.1, 3.0, (num_samples, 1))
    
    X = np.hstack((v_wind, omega)).astype(np.float32)
    Y = np.zeros((num_samples, 1), dtype=np.float32)
    
    for i in range(num_samples):
        Y[i, 0] = compute_analytical_torque(X[i, 0], X[i, 1]) / 1e6
        
    return X, Y

def train_surrogate_model(X, Y, epochs=300):
    """
    Trains a PyTorch MLP surrogate model to predict rotor torque.
    """
    print("Training PyTorch Aerodynamics Surrogate Model (CFD approximation)...")
    
    mlp = nn.Sequential(
        nn.Linear(2, 10),
        nn.Tanh(),
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 1)
    )
    
    X_tensor = torch.from_numpy(X)
    Y_tensor = torch.from_numpy(Y)
    
    optimizer = optim.Adam(mlp.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    mlp.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = mlp(X_tensor)
        loss = criterion(outputs, Y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
    mlp.eval()
    print("Surrogate training complete!")
    return mlp
