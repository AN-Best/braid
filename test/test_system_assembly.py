import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.linear_mechanical_1D import Mass, Spring, Damper, Ground
from base import System, Node

def test_assembly():
    print("Building Mass-Spring-Damper System...")
    mass   = Mass('mass', m=2.0)
    spring = Spring('spring', k=10.0)
    damper = Damper('damper', c=0.2)
    ground = Ground('ground')

    # Assemble system
    system = System([mass, spring, damper, ground])
    Node(system, [(mass, 'p'), (spring, 'p2'), (damper, 'p2')])
    Node(system, [(ground, 'p'), (spring, 'p1'), (damper, 'p1')])

    print("Converting to CasadiDAE...")
    dae = system.to_dae()
    
    print("\n--- Original Equations ---")
    for eq in dae.equations:
        print(eq)

    print("\nStates count:", len(dae.x_vars))
    print("Equations count:", len(dae.equations))

if __name__ == "__main__":
    test_assembly()
