import sys
import os
import sympy as sp
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sym_dae import SystemDAE
from index_reduction import pantelides_pass, tearing_pass, simplification_pass
from json_ir import to_json, from_json
from components.linear_mechanical_1D import Mass, Spring, Damper, Ground
from base import System, Node

def test_pantelides_pendulum():
    print("\n--- Testing Pantelides on Pendulum System ---")
    dae = SystemDAE()
    t = dae.t
    
    # States
    x = sp.Function('x')(t)
    y = sp.Function('y')(t)
    x_dot = sp.Function('x_dot')(t)
    y_dot = sp.Function('y_dot')(t)
    lam = sp.Function('lambda')(t)
    
    dae.states.extend([x, y, x_dot, y_dot, lam])
    
    # Params
    m = sp.Symbol('m')
    g = sp.Symbol('g')
    L = sp.Symbol('L')
    dae.params.extend([m, g, L])
    
    # Equations:
    # 0: m * x_dot'(t) + lambda(t) * x(t) = 0
    # 1: m * y_dot'(t) + lambda(t) * y(t) - m * g = 0
    # 2: x(t)^2 + y(t)^2 - L^2 = 0
    # 3: x'(t) - x_dot(t) = 0
    # 4: y'(t) - y_dot(t) = 0
    eq0 = m * sp.Derivative(x_dot, t) + lam * x
    eq1 = m * sp.Derivative(y_dot, t) + lam * y - m * g
    eq2 = x**2 + y**2 - L**2
    eq3 = sp.Derivative(x, t) - x_dot
    eq4 = sp.Derivative(y, t) - y_dot
    
    dae.equations.extend([eq0, eq1, eq2, eq3, eq4])
    
    # Run Pantelides
    reduced_dae = pantelides_pass(dae)
    
    print("Differentiation indices:", reduced_dae.differentiation_indices)
    # Pendulum index-3 DAE:
    # eq2 (constraint) differentiated 2 times.
    # eq3, eq4 differentiated 1 time.
    # eq0, eq1 differentiated 0 times.
    assert reduced_dae.differentiation_indices[2] == 2
    assert reduced_dae.differentiation_indices[3] == 1
    assert reduced_dae.differentiation_indices[4] == 1
    assert reduced_dae.differentiation_indices[0] == 0
    assert reduced_dae.differentiation_indices[1] == 0
    
    print("Solved variables:", reduced_dae.solved_variables)
    print("Differential states:", reduced_dae.states)
    
    # Active equations count should equal solved variables count
    assert len(reduced_dae.active_equations) == 5
    assert len(reduced_dae.solved_variables) == 5
    assert len(reduced_dae.matching) == 5
    
    # Run Tearing Pass
    print("Running tearing pass...")
    torn_dae = tearing_pass(reduced_dae)
    assert len(torn_dae.solved_assignments) == 5
    # Verify we solved for all solved variables
    for var in torn_dae.solved_variables:
        assert var in torn_dae.solved_assignments

    # Run Simplification & Elimination Pass
    print("Running simplification pass...")
    simplified_dae = simplification_pass(torn_dae)
    assert len(simplified_dae.solved_assignments) == 5
    # Pendulum has 5 states after elimination: x, y, x_dot, y_dot, lambda.
    # Only the first 4 are differential states. lambda is algebraic, so its derivative is not solved.
    # Therefore, ode_assignments should have exactly 4 entries!
    assert len(simplified_dae.ode_assignments) == 4
    for state in simplified_dae.states:
        if state != sp.Function('lambda')(simplified_dae.t):
            assert sp.Derivative(state, simplified_dae.t) in simplified_dae.ode_assignments

    # JSON Serialization & Deserialization Test for Reduced DAE (Untorn)
    print("Testing JSON Serialization (Untorn)...")
    json_str_dae = to_json(reduced_dae)
    json_data_dae = json.loads(json_str_dae)
    assert json_data_dae["system_form"] == "DAE"
    assert json_data_dae["index"] == 3
    
    # JSON Serialization & Deserialization Test for Torn & Simplified ODE
    print("Testing JSON Serialization (Torn & Simplified ODE)...")
    json_str_ode = to_json(simplified_dae)
    json_data_ode = json.loads(json_str_ode)
    assert json_data_ode["system_form"] == "ODE"
    assert json_data_ode["index"] == 0
    assert "solved_assignments" in json_data_ode
    assert len(json_data_ode["solved_assignments"]) == 5
    assert "ode_assignments" in json_data_ode
    assert len(json_data_ode["ode_assignments"]) == 4
    
    print("Testing JSON Deserialization...")
    loaded_dae = from_json(json_str_ode)
    
    assert str(loaded_dae.t) == str(dae.t)
    assert len(loaded_dae.states) == len(simplified_dae.states)
    assert len(loaded_dae.solved_variables) == len(simplified_dae.solved_variables)
    assert len(loaded_dae.active_equations) == len(simplified_dae.active_equations)
    assert len(loaded_dae.matching) == len(simplified_dae.matching)
    assert loaded_dae.system_form == "ODE"
    assert loaded_dae.index == 0
    assert len(loaded_dae.solved_assignments) == 5
    assert len(loaded_dae.ode_assignments) == 4
    
    # Verify exact equality of loaded equations
    for eq_loaded, eq_orig in zip(loaded_dae.active_equations, simplified_dae.active_equations):
        assert eq_loaded == eq_orig
        
    print("Pendulum tests passed successfully!")

def test_pantelides_mass_spring_damper():
    print("\n--- Testing Pantelides on Mass-Spring-Damper System ---")
    mass = Mass('mass', m=2.0)
    spring = Spring('spring', k=10.0)
    damper = Damper('damper', c=0.2)
    ground = Ground('ground')

    # Assemble system
    system = System([mass, spring, damper, ground])
    Node(system, [(mass, 'p'), (spring, 'p2'), (damper, 'p2')])
    Node(system, [(ground, 'p'), (spring, 'p1'), (damper, 'p1')])
    
    dae = system.to_dae()
    
    # Check component metadata domains
    assert len(dae.components) == 4
    for comp in dae.components:
        assert comp['domain'] == 'continuous'
        
    # Introduce a discrete component for testing domain tagging
    discrete_comp = Mass('discrete_mass', m=1.5)
    discrete_comp.domain = 'discrete'
    system_with_discrete = System([mass, spring, damper, ground, discrete_comp])
    dae_with_discrete = system_with_discrete.to_dae()
    
    discrete_meta = [c for c in dae_with_discrete.components if c['name'] == 'discrete_mass'][0]
    assert discrete_meta['domain'] == 'discrete'
    
    # Run Pantelides
    reduced_dae = pantelides_pass(dae)
    
    print("Differentiation indices:", reduced_dae.differentiation_indices)
    print("Solved variables:", reduced_dae.solved_variables)
    print("States:", reduced_dae.states)
    
    # Check that structural index reduction was performed successfully
    assert sum(reduced_dae.differentiation_indices) == 3
    assert reduced_dae.differentiation_indices[5] == 1
    assert reduced_dae.differentiation_indices[7] == 1
    assert reduced_dae.differentiation_indices[10] == 1
    
    # Run Tearing Pass
    print("Running tearing pass...")
    torn_dae = tearing_pass(reduced_dae)
    # The active equations has 12 equations after redundant derivative removal
    assert len(torn_dae.solved_assignments) == 12
    for var in torn_dae.solved_variables:
        assert var in torn_dae.solved_assignments
        
    # JSON Serialization & Deserialization for Untorn DAE
    json_str_dae = to_json(reduced_dae)
    loaded_dae_dae = from_json(json_str_dae)
    assert loaded_dae_dae.system_form == "DAE"
    assert loaded_dae_dae.index == 2

    # Run Simplification Pass
    print("Running simplification pass...")
    simplified_dae = simplification_pass(torn_dae)
    assert len(simplified_dae.solved_assignments) == 12
    assert len(simplified_dae.states) == 5
    assert len(simplified_dae.ode_assignments) == 5
    for state in simplified_dae.states:
        assert sp.Derivative(state, simplified_dae.t) in simplified_dae.ode_assignments

    # JSON Serialization & Deserialization for Torn & Simplified ODE
    print("Testing JSON Serialization (Torn & Simplified ODE)...")
    json_str_ode = to_json(simplified_dae)
    loaded_dae_ode = from_json(json_str_ode)
    
    assert len(loaded_dae_ode.components) == 4
    assert loaded_dae_ode.components[0]['domain'] == 'continuous'
    assert loaded_dae_ode.system_form == "ODE"
    assert loaded_dae_ode.index == 0
    assert len(loaded_dae_ode.solved_assignments) == 12
    assert len(loaded_dae_ode.ode_assignments) == 5
    
    print("Mass-Spring-Damper tests passed successfully!")

if __name__ == "__main__":
    test_pantelides_pendulum()
    test_pantelides_mass_spring_damper()
