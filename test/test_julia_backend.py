import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import System, Node
from components.linear_mechanical_1D import Mass, Spring, Damper, Ground
from simulation import simulate_system

def test_julia_backend_msd():
    print("\n--- Testing Julia Backend on Mass-Spring-Damper ---")
    
    # 1. Assemble Mass-Spring-Damper system
    mass = Mass('mass', m=2.0)
    spring = Spring('spring', k=10.0)
    damper = Damper('damper', c=1.0)
    ground = Ground('ground')
    
    system = System([mass, spring, damper, ground])
    Node(system, [(mass, 'p'), (spring, 'p2'), (damper, 'p2')])
    Node(system, [(ground, 'p'), (spring, 'p1'), (damper, 'p1')])
    
    # 2. Run simulation using Julia backend
    t_span = (0.0, 5.0)
    # States: [x_mass, v_mass]
    # Initial conditions: stretch spring to 1.0, release from rest
    y0 = [1.0, 0.0]
    
    sol = simulate_system(
        system, 
        t_span, 
        y0, 
        params=None, 
        backend='julia', 
        method='Tsit5()',
        num_steps=100
    )
    
    assert sol.success
    assert len(sol.t) == 100
    
    # The system is a damped oscillator; amplitude should decay
    # Assert initial position is 1.0
    assert np.abs(sol.y[0][0] - 1.0) < 1e-3
    
    # Assert amplitude at t=5.0 is much smaller than 1.0 (decayed due to damper)
    assert np.abs(sol.y[0][-1]) < 0.5
    
    print("Julia simulation completed and verified successfully!")
