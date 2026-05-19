import sys
import os
import sympy as sp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.rotational_mechanical_1D import Inertia, TorsionalSpring, RotationalDamper, Fixed, Torque
from base import System, Node
from index_reduction import order_reduction_pass

def test_rotational_assembly():
    print("Building Inertia-Spring-Damper Rotational System...")
    inertia = Inertia('inertia', J=2.5)
    spring  = TorsionalSpring('spring', k=15.0)
    damper  = RotationalDamper('damper', c=0.5)
    fixed   = Fixed('fixed')
    torque  = Torque('torque', tau=10.0)

    # Assemble system
    # Let's connect:
    # fixed to spring.p1 and damper.p1
    # spring.p2, damper.p2, torque.p, and inertia.p to a common node
    system = System([inertia, spring, damper, fixed, torque])
    
    Node(system, [(fixed, 'p'), (spring, 'p1'), (damper, 'p1')])
    Node(system, [(inertia, 'p'), (spring, 'p2'), (damper, 'p2'), (torque, 'p')])

    print("Converting to SystemDAE...")
    dae = system.to_dae()
    
    print("\n--- Original Equations ---")
    for eq in dae.equations:
        print(eq)
        
    print("\n--- After Order Reduction ---")
    reduced_dae = order_reduction_pass(dae)
    for eq in reduced_dae.equations:
        print(eq)

    print("\nStates count:", len(reduced_dae.states))
    print("Equations count:", len(reduced_dae.equations))
    
    assert len(reduced_dae.states) == 2, "Should have 2 states (theta_inertia and theta_inertia_dot)"
    print("\nAll rotational assembly tests passed successfully!")

if __name__ == "__main__":
    test_rotational_assembly()
