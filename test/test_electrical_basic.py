import sys
import os
import sympy as sp
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.electrical_basic import Resistor, Capacitor, Inductor, VoltageSource, CurrentSource, ElectricalGround, VoltageSensor, CurrentSensor
from base import System, Node
from index_reduction import pantelides_pass, tearing_pass
from simulation import simulate_system

def test_rc_circuit():
    print("\n--- Testing RC Circuit with Sensors ---")
    # Components
    v_src = VoltageSource('v_src', V=10.0)
    r = Resistor('r', R=2.0)
    c = Capacitor('c', C=0.5)
    gnd = ElectricalGround('gnd')
    
    # Sensors
    v_sens = VoltageSensor('v_sens')
    i_sens = CurrentSensor('i_sens')

    # Assemble system (i_sens is in series between r and c)
    system = System([v_src, r, c, gnd, v_sens, i_sens])
    Node(system, [(v_src, 'p'), (r, 'p')])
    Node(system, [(r, 'n'), (i_sens, 'p')])
    Node(system, [(i_sens, 'n'), (c, 'p'), (v_sens, 'p')])
    Node(system, [(c, 'n'), (v_src, 'n'), (gnd, 'p')])

    # Convert and reduce
    dae = system.to_dae()
    red = pantelides_pass(dae)
    torn = tearing_pass(red)

    # Initial condition: capacitor uncharged, so v_c = 0.0
    # Let's find the index of state variables
    state_to_idx = {s: idx for idx, s in enumerate(torn.state_names)}
    
    # Check that the capacitor state is present
    v_c_sym = None
    for name in torn.state_names:
        if name.startswith('v_c_c'):
            v_c_sym = name
            break

    assert v_c_sym is not None, "Capacitor state v_c not found"

    # Check sensor mappings resolution
    assert 'v_sens' in torn.sensor_mappings
    assert 'i_sens' in torn.sensor_mappings
    
    print("\n--- Sensor Mappings ---")
    for k, v in torn.sensor_mappings.items():
        print(f"{k} -> {v}")

    # Voltage sensor should resolve to the node voltage v_n_i_sens (which is equal to v_p_c)
    assert torn.sensor_mappings['v_sens']['target'] in ('v_p_c', 'v_n_i_sens')
    # Current sensor should resolve directly to the current name
    assert torn.sensor_mappings['i_sens']['target'] == 'i_sens_i_sens'

    y0 = [0.0] * len(torn.state_names)
    # Simulate
    t_span = (0.0, 3.0)
    t_eval = np.linspace(0.0, 3.0, 301)
    sol = simulate_system(torn, t_span, y0, params=None, backend='numpy', method='RK45', t_eval=t_eval)
    assert sol.success

    # Check value at t=1.0. With R=2, C=0.5, tau=1.0.
    # v_c(1.0) should be 10 * (1 - exp(-1)) ~= 6.3212 V
    t_idx = np.argmin(np.abs(sol.t - 1.0))
    v_c_val = sol.y[state_to_idx[v_c_sym]][t_idx]

    expected_v = 10.0 * (1.0 - np.exp(-1.0))

    print(f"Simulated v_c(1.0): {v_c_val:.4f}, Expected: {expected_v:.4f}")

    assert np.abs(v_c_val - expected_v) < 1e-2

def test_rlc_circuit():
    print("\n--- Testing RLC Circuit ---")
    # Components
    v_src = VoltageSource('v_src', V=12.0)
    r = Resistor('r', R=1.0)
    l = Inductor('l', L=0.5)
    c = Capacitor('c', C=0.1)
    gnd = ElectricalGround('gnd')

    # Assemble system
    system = System([v_src, r, l, c, gnd])
    Node(system, [(v_src, 'p'), (r, 'p')])
    Node(system, [(r, 'n'), (l, 'p')])
    Node(system, [(l, 'n'), (c, 'p')])
    Node(system, [(c, 'n'), (v_src, 'n'), (gnd, 'p')])

    # Convert and reduce
    dae = system.to_dae()
    red = pantelides_pass(dae)
    torn = tearing_pass(red)

    # Find the indices of state variables
    state_to_idx = {s: idx for idx, s in enumerate(torn.state_names)}
    
    v_c_sym = None
    i_L_sym = None
    for name in torn.state_names:
        if name.startswith('v_c_c'):
            v_c_sym = name
        elif name.startswith('i_L_l'):
            i_L_sym = name

    assert v_c_sym is not None, "Capacitor state v_c not found"
    assert i_L_sym is not None, "Inductor state i_L not found"

    y0 = [0.0] * len(torn.state_names)
    t_span = (0.0, 5.0)
    
    sol = simulate_system(torn, t_span, y0, params=None, backend='numpy', method='RK45')
    assert sol.success
    print("RLC simulation succeeded!")

if __name__ == "__main__":
    test_rc_circuit()
    test_rlc_circuit()
