import sys
import os
import sympy as sp
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sym_dae import SystemDAE
from index_reduction import pantelides_pass, tearing_pass, simplification_pass
from simulation import simulate_system
from components.linear_mechanical_1D import Mass, Spring, Damper, Ground
from base import System, Node

def test_simulation_pendulum():
    print("\n--- Testing Simulation of Pendulum System ---")
    dae = SystemDAE()
    t = dae.t
    
    # States
    x = sp.Function('x')(t)
    y = sp.Function('y')(t)
    x_dot = sp.Function('x_dot')(t)
    y_dot = sp.Function('y_dot')(t)
    lam = sp.Function('lambda')(t)
    
    dae.states.extend([x, y, x_dot, y_dot, lam])
    
    # Params: m=1.0, g=9.81, L=1.0
    m = sp.Symbol('m')
    g = sp.Symbol('g')
    L = sp.Symbol('L')
    dae.params.extend([m, g, L])
    
    # Equations
    eq0 = m * sp.Derivative(x_dot, t) + lam * x
    eq1 = m * sp.Derivative(y_dot, t) + lam * y - m * g
    eq2 = x**2 + y**2 - L**2
    eq3 = sp.Derivative(x, t) - x_dot
    eq4 = sp.Derivative(y, t) - y_dot
    
    dae.equations.extend([eq0, eq1, eq2, eq3, eq4])
    
    # Compile
    red = pantelides_pass(dae)
    torn = tearing_pass(red)
    simp = simplification_pass(torn)
    
    # Write to a JSON file first to avoid recompiling in simulation
    from json_ir import to_json
    json_str = to_json(simp)
    compiled_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compiled_models")
    os.makedirs(compiled_dir, exist_ok=True)
    json_path = os.path.join(compiled_dir, "pendulum_ode_test.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_str)
        
    # Initial conditions: start at x=1.0, y=0.0 (horizontal position), velocity=0
    # States order: x, x_dot, y, y_dot, lambda
    # Wait, the states list is simp.states. Let's make sure we map y0 correctly!
    state_to_idx = {s: idx for idx, s in enumerate(simp.states)}
    y0 = [0.0] * len(simp.states)
    y0[state_to_idx[x]] = 1.0     # x = 1.0 (L=1.0)
    y0[state_to_idx[y]] = 0.0     # y = 0.0
    y0[state_to_idx[x_dot]] = 0.0 # x_dot = 0.0
    y0[state_to_idx[y_dot]] = 0.0 # y_dot = 0.0
    
    params_val = [1.0, 9.81, 1.0] # m, g, L
    t_span = (0.0, 1.5)
    
    # Test NumPy Backend with various methods
    numpy_methods = ['RK45', 'BDF', 'rk4', 'backward_euler']
    for method in numpy_methods:
        print(f"Simulating Pendulum with NumPy backend ({method})...")
        sol_num = simulate_system(json_path, t_span, y0, params_val, backend='numpy', method=method, rtol=1e-6, atol=1e-6, num_steps=3000)
        assert sol_num.success
        
        # Extract results
        x_val = sol_num.y[state_to_idx[x]]
        y_val = sol_num.y[state_to_idx[y]]
        x_dot_val = sol_num.y[state_to_idx[x_dot]]
        y_dot_val = sol_num.y[state_to_idx[y_dot]]
        
        # Assert constraint x^2 + y^2 = L^2 (L=1.0)
        len_sq = x_val**2 + y_val**2
        np.testing.assert_allclose(len_sq, 1.0, atol=1.5e-2)
        
        # Assert Energy Conservation: E = 0.5 * m * (x_dot^2 + y_dot^2) - m * g * y
        energy = 0.5 * 1.0 * (x_dot_val**2 + y_dot_val**2) - 1.0 * 9.81 * y_val
        np.testing.assert_allclose(energy, 0.0, atol=1.5e-2)
    
    # Test PyTorch Backend with various methods
    pytorch_methods = ['rk4', 'backward_euler']
    for method in pytorch_methods:
        print(f"Simulating Pendulum with PyTorch backend ({method})...")
        sol_torch = simulate_system(json_path, t_span, y0, params_val, backend='pytorch', method=method, num_steps=3000)
        assert sol_torch.success
        
        x_torch = sol_torch.y[state_to_idx[x]]
        y_torch = sol_torch.y[state_to_idx[y]]
        x_dot_torch = sol_torch.y[state_to_idx[x_dot]]
        y_dot_torch = sol_torch.y[state_to_idx[y_dot]]
        
        len_sq_torch = x_torch**2 + y_torch**2
        np.testing.assert_allclose(len_sq_torch, 1.0, atol=1.5e-2)
        
        energy_torch = 0.5 * 1.0 * (x_dot_torch**2 + y_dot_torch**2) - 1.0 * 9.81 * y_torch
        np.testing.assert_allclose(energy_torch, 0.0, atol=1.5e-2)
            
    print("Pendulum simulation tests passed successfully!")

def test_simulation_mass_spring_damper():
    print("\n--- Testing Simulation of Mass-Spring-Damper System ---")
    mass = Mass('mass', m=2.0)
    spring = Spring('spring', k=10.0)
    damper = Damper('damper', c=1.0) # slightly high damping to see decay quickly
    ground = Ground('ground')
    
    system = System([mass, spring, damper, ground])
    Node(system, [(mass, 'p'), (spring, 'p2'), (damper, 'p2')])
    Node(system, [(ground, 'p'), (spring, 'p1'), (damper, 'p1')])
    
    dae = system.to_dae()
    red = pantelides_pass(dae)
    torn = tearing_pass(red)
    simp = simplification_pass(torn)
    
    # Write to a JSON file first to avoid recompiling in simulation
    from json_ir import to_json
    json_str = to_json(simp)
    compiled_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compiled_models")
    os.makedirs(compiled_dir, exist_ok=True)
    json_path = os.path.join(compiled_dir, "mass_spring_damper_test_ode.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_str)
        
    # States: x1_damper, x2_damper, x_ground, x_mass, Derivative(x_mass, t)
    # Let's map states:
    state_to_idx = {s: idx for idx, s in enumerate(simp.states)}
    
    # Let's find state symbols
    # Helper to find state by string name:
    def find_state_by_name(name):
        for s in simp.states:
            if s.func.__name__ == name:
                return s
        raise ValueError(f"State {name} not found")
        
    x_mass = find_state_by_name('x_mass')
    v_mass = [s for s in simp.states if isinstance(s, sp.Derivative) and s.expr == x_mass][0]
    
    # Initial state: stretch the spring/mass to x_mass = 2.0, other positions at 0.
    y0 = [0.0] * len(simp.states)
    y0[state_to_idx[x_mass]] = 2.0
    
    # params: m_mass=2.0, k_spring=10.0, c_damper=3.0
    # Wait, the order of params is: m_mass, k_spring, c_damper
    params_val = [2.0, 10.0, 3.0]
    t_span = (0.0, 10.0)
    
    # Simulating NumPy
    print("Simulating Mass-Spring-Damper with NumPy backend...")
    sol_num = simulate_system(json_path, t_span, y0, params_val, backend='numpy', rtol=1e-6, atol=1e-6)
    assert sol_num.success
    
    x_mass_vals = sol_num.y[state_to_idx[x_mass]]
    v_mass_vals = sol_num.y[state_to_idx[v_mass]]
    
    # Check that amplitude decays over time (decay behavior)
    # The initial amplitude is 2.0. By t=10.0, it should be close to 0.
    assert abs(x_mass_vals[-1]) < 0.1
    assert abs(v_mass_vals[-1]) < 0.05
    
    # Simulating PyTorch
    print("Simulating Mass-Spring-Damper with PyTorch backend...")
    sol_torch = simulate_system(json_path, t_span, y0, params_val, backend='pytorch', num_steps=3000)
    assert sol_torch.success
    
    x_mass_torch = sol_torch.y[state_to_idx[x_mass]]
    v_mass_torch = sol_torch.y[state_to_idx[v_mass]]
    
    # Check that amplitude decays over time (decay behavior)
    # The initial amplitude is 2.0. By t=10.0, it should be close to 0.
    assert abs(x_mass_torch[-1]) < 0.1
    assert abs(v_mass_torch[-1]) < 0.05
            
    print("Mass-Spring-Damper simulation tests passed successfully!")

def test_parallel_gpu_simulation():
    import torch
    print("\n--- Testing Parallel GPU Simulation of Pendulum System ---")
    dae = SystemDAE()
    t = dae.t
    
    # States
    x = sp.Function('x')(t)
    y = sp.Function('y')(t)
    x_dot = sp.Function('x_dot')(t)
    y_dot = sp.Function('y_dot')(t)
    lam = sp.Function('lambda')(t)
    
    dae.states.extend([x, y, x_dot, y_dot, lam])
    
    # Params: m=1.0, g=9.81, L=1.0 (will sweep L)
    m = sp.Symbol('m')
    g = sp.Symbol('g')
    L = sp.Symbol('L')
    dae.params.extend([m, g, L])
    
    # Equations
    eq0 = m * sp.Derivative(x_dot, t) + lam * x
    eq1 = m * sp.Derivative(y_dot, t) + lam * y - m * g
    eq2 = x**2 + y**2 - L**2
    eq3 = sp.Derivative(x, t) - x_dot
    eq4 = sp.Derivative(y, t) - y_dot
    
    dae.equations.extend([eq0, eq1, eq2, eq3, eq4])
    
    # Compile
    red = pantelides_pass(dae)
    torn = tearing_pass(red)
    simp = simplification_pass(torn)
    
    # Write to a JSON file first to avoid recompiling in simulation
    from json_ir import to_json
    json_str = to_json(simp)
    compiled_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compiled_models")
    os.makedirs(compiled_dir, exist_ok=True)
    json_path = os.path.join(compiled_dir, "parallel_pendulum_test_ode.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_str)
        
    state_to_idx = {s: idx for idx, s in enumerate(simp.states)}
    
    # Let's create a batch of 8 simulation instances.
    # We will sweep the pendulum length L from 0.5 to 2.0.
    batch_size = 8
    L_vals = np.linspace(0.5, 2.0, batch_size)
    
    # Construct y0 batch of shape (batch_size, num_states)
    y0_batch = np.zeros((batch_size, len(simp.states)))
    for b in range(batch_size):
        y0_batch[b, state_to_idx[x]] = L_vals[b]
        y0_batch[b, state_to_idx[y]] = 0.0
        y0_batch[b, state_to_idx[x_dot]] = 0.0
        y0_batch[b, state_to_idx[y_dot]] = 0.0
        
    # Construct params batch of shape (batch_size, num_params)
    params_batch = np.zeros((batch_size, 3))
    for b in range(batch_size):
        params_batch[b, 0] = 1.0       # m = 1.0
        params_batch[b, 1] = 9.81      # g = 9.81
        params_batch[b, 2] = L_vals[b] # L
        
    t_span = (0.0, 1.5)
    
    # Decide device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running batch simulation on device: {device}...")
    
    # Run simulation with RK4 method in parallel
    sol_batch = simulate_system(json_path, t_span, y0_batch, params_batch, backend='pytorch', method='rk4', device=device, num_steps=3000)
    assert sol_batch.success
    assert sol_batch.y.shape == (batch_size, len(simp.states), 3001)
    
    # Verify constraints and energy conservation for each simulation run in the batch
    for b in range(batch_size):
        L_val = L_vals[b]
        x_val = sol_batch.y[b, state_to_idx[x]]
        y_val = sol_batch.y[b, state_to_idx[y]]
        x_dot_val = sol_batch.y[b, state_to_idx[x_dot]]
        y_dot_val = sol_batch.y[b, state_to_idx[y_dot]]
        
        # Constraint: x^2 + y^2 = L^2
        len_sq = x_val**2 + y_val**2
        np.testing.assert_allclose(len_sq, L_val**2, atol=1.5e-2)
        
        # Energy: E = 0.5 * m * (x_dot^2 + y_dot^2) - m * g * y
        energy = 0.5 * 1.0 * (x_dot_val**2 + y_dot_val**2) - 1.0 * 9.81 * y_val
        np.testing.assert_allclose(energy, 0.0, atol=1.5e-2)
            
    print(f"Parallel GPU simulation test on device {device} passed successfully!")
 
def test_simulation_with_structured_params():
    print("\n--- Testing Structured Parameters (Dict & Defaults) ---")
    mass = Mass('mass', m=2.0)
    spring = Spring('spring', k=10.0)
    damper = Damper('damper', c=3.0)
    ground = Ground('ground')
    
    system = System([mass, spring, damper, ground])
    Node(system, [(mass, 'p'), (spring, 'p2'), (damper, 'p2')])
    Node(system, [(ground, 'p'), (spring, 'p1'), (damper, 'p1')])
    
    dae = system.to_dae()
    red = pantelides_pass(dae)
    torn = tearing_pass(red)
    simp = simplification_pass(torn)
    
    from json_ir import to_json
    json_str = to_json(simp)
    
    compiled_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compiled_models")
    os.makedirs(compiled_dir, exist_ok=True)
    json_path = os.path.join(compiled_dir, "structured_params_test.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_str)
        
    state_to_idx = {s: idx for idx, s in enumerate(simp.states)}
    x_mass = [s for s in simp.states if s.func.__name__ == 'x_mass'][0]
    
    y0 = [0.0] * len(simp.states)
    y0[state_to_idx[x_mass]] = 2.0
    t_span = (0.0, 5.0)
    
    # 1. Test simulation using default values (params=None)
    sol_defaults = simulate_system(json_path, t_span, y0, params=None, backend='numpy')
    assert sol_defaults.success
    
    # 2. Test simulation overrides using symbol names as keys
    sol_dict_sym = simulate_system(json_path, t_span, y0, params={"m_mass": 2.0, "k_spring": 10.0, "c_damper": 3.0}, backend='numpy')
    assert sol_dict_sym.success
    np.testing.assert_allclose(sol_defaults.y, sol_dict_sym.y)
    
    # 3. Test simulation overrides using component-dot-parameter style
    sol_dict_comp = simulate_system(json_path, t_span, y0, params={"mass.m": 2.0, "spring.k": 10.0, "damper.c": 3.0}, backend='numpy')
    assert sol_dict_comp.success
    np.testing.assert_allclose(sol_defaults.y, sol_dict_comp.y)
    
    # 4. Test partial overrides (others fall back to defaults)
    sol_dict_partial = simulate_system(json_path, t_span, y0, params={"damper.c": 3.0}, backend='numpy')
    assert sol_dict_partial.success
    np.testing.assert_allclose(sol_defaults.y, sol_dict_partial.y)

    # 5. Test batch simulation with parameter sweeps using dictionaries
    batch_size = 4
    k_vals = np.linspace(5.0, 15.0, batch_size)
    y0_batch = np.zeros((batch_size, len(simp.states)))
    for b in range(batch_size):
        y0_batch[b, state_to_idx[x_mass]] = 2.0
        
    sol_batch = simulate_system(
        json_path, t_span, y0_batch, 
        params={"mass.m": 2.0, "spring.k": k_vals, "damper.c": 3.0}, 
        backend='pytorch', method='rk4', num_steps=1000
    )
    assert sol_batch.success
    assert sol_batch.y.shape == (batch_size, len(simp.states), 1001)
    
    print("Structured parameters tests passed successfully!")

def test_differentiation_wrt_parameters():
    import torch
    from simulation import lambdify_system
    print("\n--- Testing Differentiability W.R.T. Parameters ---")
    mass = Mass('mass', m=2.0)
    spring = Spring('spring', k=10.0)
    damper = Damper('damper', c=3.0)
    ground = Ground('ground')
    
    system = System([mass, spring, damper, ground])
    Node(system, [(mass, 'p'), (spring, 'p2'), (damper, 'p2')])
    Node(system, [(ground, 'p'), (spring, 'p1'), (damper, 'p1')])
    
    dae = system.to_dae()
    red = pantelides_pass(dae)
    torn = tearing_pass(red)
    simp = simplification_pass(torn)
    
    state_to_idx = {s: idx for idx, s in enumerate(simp.states)}
    x_mass = [s for s in simp.states if s.func.__name__ == 'x_mass'][0]
    
    y0 = [0.0] * len(simp.states)
    y0[state_to_idx[x_mass]] = 2.0
    
    ode_func_raw = lambdify_system(simp, 'torch')
    
    # Define PyTorch parameters with gradient tracking
    k_param = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    m_param = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    
    # Set up parameter dictionary overrides
    params_dict = {"mass.m": m_param, "spring.k": k_param, "damper.c": 3.0}
    
    # Emulate the parameter conversion mapping using SystemDAE param_meta
    flat_params = []
    param_meta = getattr(simp, "param_meta", {})
    for p in simp.params:
        sym_repr = sp.srepr(p)
        meta = param_meta[sym_repr]
        comp_dot_name = f"{meta['component']}.{meta['name']}"
        val = params_dict[comp_dot_name]
        flat_params.append(val)
        
    torch_params = [val if isinstance(val, torch.Tensor) else torch.tensor(val, dtype=torch.float64) for val in flat_params]
    params_tensor = torch.stack(torch_params)
    
    # Run a step of the ODE function
    t_val = 0.0
    y_vec = torch.tensor(y0, dtype=torch.float64)
    y_list_input = [y_vec[i] for i in range(len(y_vec))]
    params_list_input = [params_tensor[i] for i in range(len(params_tensor))]
    
    res = ode_func_raw(t_val, y_list_input, params_list_input)
    res_tensor = torch.stack([torch.as_tensor(r, dtype=torch.float64) for r in res])
    
    # Compute a loss and backpropagate
    loss = res_tensor.sum()
    loss.backward()
    
    assert k_param.grad is not None
    assert m_param.grad is not None
    print(f"Gradients computed successfully! dLoss/dK = {k_param.grad.item()}, dLoss/dM = {m_param.grad.item()}")

    # Verify simulate_system runs with dictionary of tensors (converts and executes successfully)
    sol = simulate_system(
        dae = simp,
        t_span = (0.0, 1.0),
        y0 = y0,
        params = {"mass.m": m_param.detach(), "spring.k": k_param.detach(), "damper.c": 3.0},
        backend = 'pytorch',
        method = 'rk4',
        num_steps = 100
    )
    assert sol.success
    print("simulate_system executed successfully with dictionary containing PyTorch tensors!")

if __name__ == "__main__":
    test_simulation_pendulum()
    test_simulation_mass_spring_damper()
    test_parallel_gpu_simulation()
    test_simulation_with_structured_params()
    test_differentiation_wrt_parameters()
