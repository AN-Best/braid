import sys
import os
import sympy as sp
import sympy.physics.mechanics as me
import numpy as np
import casadi as ca

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.rigid_body import RigidBodySymPy
from base import System
from index_reduction import pantelides_pass, tearing_pass
from simulation import simulate_system

def test_sympy_pendulum():
    print("\n--- Testing SymPy Pendulum via KanesMethod ---")
    
    # 1. Set up the pendulum using SymPy Physics Mechanics
    q, u = me.dynamicsymbols('q u')
    m, l, g = sp.symbols('m l g')
    
    N = me.ReferenceFrame('N')
    O = me.Point('O')
    O.set_vel(N, 0)
    
    A = N.orientnew('A', 'Axis', [q, N.z])
    A.set_ang_vel(N, u * N.z)
    
    P = O.locatenew('P', l * A.x)
    P.v2pt_theory(O, N, A)
    
    Pa = me.Particle('Pa', P, m)
    
    # Let's add gravity and a control torque
    F_gravity = (P, -m * g * N.y)
    
    tau = sp.Symbol('tau')
    T_input = (A, tau * N.z)
    
    t = sp.Symbol('t')
    kane = me.KanesMethod(N, q_ind=[q], u_ind=[u], kd_eqs=[sp.Derivative(q, t) - u])
    fr, frstar = kane.kanes_equations([Pa], [F_gravity, T_input])
    
    # 2. Instantiate RigidBodySymPy Braid Component
    pendulum = RigidBodySymPy(
        name="pendulum",
        q=[q],
        u=[u],
        equations=kane,
        control_symbols=[tau],
        param_symbols={m: 2.0, l: 1.5, g: 9.81}
    )
    
    # Assemble into a Braid System
    system = System([pendulum])
    
    # Define constant torque control input: tau = 5.0
    system.equations.append(pendulum.ports["tau"][0] - 5.0)
    
    # Compile
    dae = system.to_dae()
    
    # Run index reduction and tearing
    red = pantelides_pass(dae)
    torn = tearing_pass(red)
    
    # Verify states and equations
    assert len(torn.state_names) == 2
    assert "q_pendulum_q" in torn.state_names
    assert "u_pendulum_u" in torn.state_names
    
    # Run simulation
    y0 = [0.1, 0.0]  # initial angle = 0.1 rad, initial speed = 0.0 rad/s
    t_span = (0.0, 2.0)
    
    # Run with KMP_DUPLICATE_LIB_OK=TRUE behavior
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    sol = simulate_system(torn, t_span, y0, params=None, backend='numpy', method='RK45')
    assert sol.success
    print("Simulation succeeded!")
    print("Final angle:", sol.y[0][-1])
    print("Final speed:", sol.y[1][-1])

def test_sympy_pendulum_lagrange():
    print("\n--- Testing SymPy Pendulum via LagrangesMethod ---")
    
    q = me.dynamicsymbols('q')
    qd = me.dynamicsymbols('q', 1)
    m, l, g = sp.symbols('m l g')
    
    # Kinetic and Potential Energy
    T = 0.5 * m * (l * qd)**2
    V = -m * g * l * sp.cos(q)
    L = T - V
    tau = sp.Symbol('tau')
    N = me.ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q, N.z])
    fl = [(A, tau * N.z)]
    
    LM = me.LagrangesMethod(L, [q], forcelist=fl, frame=N)
    LM.form_lagranges_equations()

    
    pendulum = RigidBodySymPy(
        name="pendulum_lagrange",
        q=[q],
        u=[qd],
        equations=LM,
        control_symbols=[tau],
        param_symbols={m: 2.0, l: 1.5, g: 9.81}
    )
    
    system = System([pendulum])
    system.equations.append(pendulum.ports["tau"][0] - 5.0)
    
    dae = system.to_dae()
    red = pantelides_pass(dae)
    torn = tearing_pass(red)
    
    assert len(torn.state_names) == 2
    assert "q_pendulum_lagrange_q" in torn.state_names
    assert "u_pendulum_lagrange_q_dot" in torn.state_names
    
    y0 = [0.1, 0.0]
    t_span = (0.0, 2.0)
    
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    sol = simulate_system(torn, t_span, y0, params=None, backend='numpy', method='RK45')
    assert sol.success
    print("Lagrange simulation succeeded!")
    print("Final angle:", sol.y[0][-1])
    print("Final speed:", sol.y[1][-1])

if __name__ == "__main__":
    test_sympy_pendulum()
    test_sympy_pendulum_lagrange()

