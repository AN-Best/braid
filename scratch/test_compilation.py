import sys
import os
import sympy as sp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import System, Node
from components.linear_mechanical_1D import Mass, Spring, Damper, Ground, Force, PositionSensor, VelocitySensor
from index_reduction import pantelides_pass, tearing_pass

def test_diag():
    # Recreate the system layout:
    # ground_1, spring_1, damper_1, mass_1, force_1, positionsensor_1, velocitysensor_1
    ground_1 = Ground('ground_1')
    spring_1 = Spring('spring_1', k=10.0)
    damper_1 = Damper('damper_1', c=1.0)
    mass_1 = Mass('mass_1', m=2.0)
    force_1 = Force('force_1', F=0.0)
    positionsensor_1 = PositionSensor('positionsensor_1')
    velocitysensor_1 = VelocitySensor('velocitysensor_1')

    system = System([ground_1, spring_1, damper_1, mass_1, force_1, positionsensor_1, velocitysensor_1])
    
    # Connections:
    # 1. ground_1.p <-> spring_1.p1 <-> damper_1.p1
    Node(system, [(ground_1, 'p'), (spring_1, 'p1'), (damper_1, 'p1')])
    
    # 2. spring_1.p2 <-> mass_1.p <-> damper_1.p2 <-> force_1.p <-> positionsensor_1.p <-> velocitysensor_1.p
    Node(system, [
        (spring_1, 'p2'), 
        (mass_1, 'p'), 
        (damper_1, 'p2'), 
        (force_1, 'p'), 
        (positionsensor_1, 'p'), 
        (velocitysensor_1, 'p')
    ])

    print("Converting to DAE...")
    dae = system.to_dae()
    
    print("Running Pantelides pass...")
    red = pantelides_pass(dae)
    
    print("Running Tearing pass...")
    torn = tearing_pass(red)
    print("Torn solved assignments empty?", len(torn.alg_assignments) == 0)
    print("Solved variables:", torn.solved_variables)
    
    print("\nActive Equations:")
    for eq in red.active_equations:
        print("  ", eq)

if __name__ == "__main__":
    test_diag()
