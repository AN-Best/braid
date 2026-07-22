import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import torch.nn as nn
import numpy as np
import casadi as ca

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from components.neural_net import NeuralNetworkPyTorch
from base import Component, System
from index_reduction import pantelides_pass, tearing_pass
from ir import to_json
from lowering.torch_lowering import make_torch_ode
from lowering.numpy_lowering import make_numpy_ode

def test_pytorch_neural_network_component():
    print("\n--- Testing PyTorch Neural Network Component ---")
    
    # 1. Define a simple PyTorch model
    torch.manual_seed(42)
    mlp = nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 1)
    )
    
    # 2. Instantiate the Braid component
    nn_comp = NeuralNetworkPyTorch(
        name="nn_ctrl",
        pytorch_model=mlp,
        input_names=["x1", "x2"],
        output_names=["y"]
    )
    
    # 3. Create a system with two state inputs and the neural network
    class InputStates(Component):
        def __init__(self, name="inputs"):
            super().__init__(name)
            # Create two states
            self.x1, self.x1_dot = self.add_state("x1")
            self.x2, self.x2_dot = self.add_state("x2")
            
            # Simple dynamics: x1_dot = 1, x2_dot = 2
            self.equations.append(self.x1_dot - 1.0)
            self.equations.append(self.x2_dot - 2.0)
            
            # Expose as ports
            self.ports["x1"] = [ca.SX(0), self.x1, self.x1_dot]
            self.ports["x2"] = [ca.SX(0), self.x2, self.x2_dot]
            
    inputs = InputStates()
    system = System([inputs, nn_comp])
    
    # Connect input states to NN inputs using simple equation constraints
    system.equations.append(inputs.ports["x1"][1] - nn_comp.ports["x1"][1])
    system.equations.append(inputs.ports["x2"][1] - nn_comp.ports["x2"][1])
    
    # 4. Compile the DAE system
    dae = system.to_dae()
    red = pantelides_pass(dae)
    torn = tearing_pass(red)
    
    # Verify states and equations
    assert len(torn.state_names) == 2
    assert "x1" in torn.state_names
    assert "x2" in torn.state_names
    
    # Check that y is solved algebraically in torn.alg_assignments
    y_sym = nn_comp.ports["y"][0]
    assert y_sym.name() in torn.alg_assignments
    
    # 5. Evaluate and verify numerical equivalence
    x_test = np.array([0.5, -0.3], dtype=np.float32)
    x_torch = torch.tensor(x_test)
    y_expected = mlp(x_torch).item()
    
    # Evaluate via CasADi
    # We create a CasADi Function for the algebraic assignments
    # We want to solve for y given x1 and x2
    x1_sym = inputs.x1
    x2_sym = inputs.x2
    y_expr = torn.alg_assignments[y_sym.name()]
    
    y_func = ca.Function("y_func", [x1_sym, x2_sym], [y_expr])
    y_casadi = float(y_func(x_test[0], x_test[1]))
    
    print(f"Expected (PyTorch): {y_expected}")
    print(f"Actual (CasADi):    {y_casadi}")
    assert np.allclose(y_casadi, y_expected, atol=1e-5)
    
    # 6. Verify IR lowering to PyTorch and NumPy backends
    ir = json.loads(to_json(torn))
    
    # NumPy backend check
    np_ode = make_numpy_ode(ir)
    dx_np = np_ode(0.0, np.array([0.5, -0.3]), np.array([]))
    assert np.allclose(dx_np, np.array([1.0, 2.0]))
    
    # PyTorch backend check
    torch_ode = make_torch_ode(ir)
    dx_torch = torch_ode(0.0, torch.tensor([0.5, -0.3]), torch.tensor([]))
    assert torch.allclose(dx_torch, torch.tensor([1.0, 2.0]))
    
    print("All tests passed!")

if __name__ == "__main__":
    test_pytorch_neural_network_component()
