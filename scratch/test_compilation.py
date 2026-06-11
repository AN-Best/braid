import sys
import os
import sympy as sp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import System, Node
from components.linear_mechanical_1D import Mass, Spring, Damper, Ground, Force, PositionSensor, VelocitySensor
from index_reduction import pantelides_pass, tearing_pass, simplification_pass

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
    print("Torn solved assignments empty?", len(torn.solved_assignments) == 0)
    print("Solved variables:", torn.solved_variables)
    
    # If it failed to solve, try to solve step-by-step or print the equations
    if len(torn.solved_assignments) == 0:
        print("\nActive Equations:")
        for eq in red.active_equations:
            print("  ", eq)
        try:
            print("\nAttempting direct sympy solve...")
            sol = sp.solve(red.active_equations, red.solved_variables, dict=True)
            print("Direct solve success! Result size:", len(sol[0]) if sol else 0)
        except Exception as e:
            print("Direct solve failed:", e)

    print("\nRunning Simplification pass...")
    simp = simplification_pass(torn)
    print("States:", simp.states)
    print("ODE Assignments:")
    for k, v in simp.ode_assignments.items():
        print(f"  {k} = {v}")
    
    print("\nSolved Assignments:")
    for k, v in simp.solved_assignments.items():
        print(f"  {k} = {v}")
        
    print("\nTesting lambdifying solved assignments...")
    try:
        vars_list = list(simp.solved_assignments.keys())
        exprs_list = list(simp.solved_assignments.values())
        fn = sp.lambdify((simp.t, simp.states, simp.params), exprs_list, 'numpy')
        print("Lambdify succeeded! Variables:", [str(v) for v in vars_list])
    except Exception as e:
        print("Lambdify failed:", e)

if __name__ == "__main__":
    test_diag()
