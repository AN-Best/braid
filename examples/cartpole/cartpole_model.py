import sympy as sp
import sympy.physics.mechanics as me
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from components.rigid_body import RigidBodySymPy

def get_cartpole_component(name="cartpole", mc_val=1.0, mp_val=0.2, l_val=0.5, g_val=9.81):
    """
    Creates a Cartpole Braid component using sympy.physics.mechanics KanesMethod.
    
    Coordinates:
      x: horizontal cart position
      theta: pole angle from vertical upright (0 is upright, positive clockwise)
      
    Speeds:
      u1: cart velocity (dx/dt)
      u2: pole angular velocity (dtheta/dt)
      
    Parameters:
      mc: cart mass
      mp: pole mass
      l: pole length (to center of mass/mass particle)
      g: gravitational acceleration
      
    Inputs:
      F: force applied to the cart
    """
    # 1. Setup SymPy Symbols
    mc, mp, l, g = sp.symbols('mc mp l g')
    F = sp.Symbol('F')
    
    # 2. Setup dynamic symbols
    t = sp.Symbol('t')
    x, theta = me.dynamicsymbols('x theta')
    u1, u2 = me.dynamicsymbols('u1 u2')
    
    # Kinematic differential equations
    kd_eqs = [sp.Derivative(x, t) - u1, sp.Derivative(theta, t) - u2]
    
    # 3. Define Reference Frames and Points
    N = me.ReferenceFrame('N')
    O = me.Point('O')
    O.set_vel(N, 0)
    
    # Cart point and frame
    C = O.locatenew('C', x * N.x)
    C.set_vel(N, u1 * N.x)
    
    # Pole frame and particle
    # Rotating by theta around N.z
    A = N.orientnew('A', 'Axis', [theta, N.z])
    A.set_ang_vel(N, u2 * N.z)
    
    # Pole center of mass (at length l along negative A.y so that theta=0 points UPWARDS)
    # If theta=0, A.y is along N.y.
    # When theta is positive, it rotates towards N.x.
    P = C.locatenew('P', l * A.y)
    P.v2pt_theory(C, N, A)
    
    # 4. Define Inertial Bodies
    cart = me.Particle('cart', C, mc)
    pole = me.Particle('pole', P, mp)
    
    # 5. Define Forces
    # Gravity acts in the -N.y direction.
    forces = [
        (C, F * N.x),
        (P, -mp * g * N.y)
    ]
    
    # 6. Formulate Equations of Motion using KanesMethod
    kane = me.KanesMethod(N, q_ind=[x, theta], u_ind=[u1, u2], kd_eqs=kd_eqs)
    fr, frstar = kane.kanes_equations([cart, pole], forces)
    
    # 7. Create Braid Component
    cartpole_comp = RigidBodySymPy(
        name=name,
        q=[x, theta],
        u=[u1, u2],
        equations=kane,
        control_symbols=[F],
        param_symbols={mc: mc_val, mp: mp_val, l: l_val, g: g_val}
    )
    
    return cartpole_comp
