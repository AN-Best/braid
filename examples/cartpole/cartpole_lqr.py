import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import casadi as ca
import scipy.linalg

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base import Component, System, Node
from index_reduction import pantelides_pass, tearing_pass
from simulation import simulate_system
from examples.cartpole.cartpole_model import get_cartpole_component

class ParamComponent(Component):
    def __init__(self, name="param_comp"):
        super().__init__(name)
        self.F = ca.SX.sym("F_in")
        self.register_param("F_in", self.F, 0.0)

def compute_lqr_gain():
    print("Designing LQR Controller...")
    
    # 1. Instantiate the Cartpole component and compile the open-loop system
    cartpole = get_cartpole_component()
    param_comp = ParamComponent()
    
    system = System([cartpole, param_comp])
    system.equations.append(cartpole.ports["F"][0] - param_comp.F)
    
    dae = system.to_dae()
    red = pantelides_pass(dae)
    torn = tearing_pass(red)
    
    # 2. Linearize the open-loop dynamics around the upright equilibrium
    # State vector order: [q_cartpole_x, q_cartpole_theta, u_cartpole_u1, u_cartpole_u2]
    x_eq = np.array([0.0, 0.0, 0.0, 0.0]) # x=0, theta=0 (upright), x_dot=0, theta_dot=0
    
    # Retrieve parameters and locate control force F_in
    p_vals = np.array(torn.get_param_defaults())
    f_idx = torn.param_names.index("F_in")
    p_vals[f_idx] = 0.0
    
    # Form Jacobians symbolically
    f_x = ca.jacobian(torn.ode_rhs_vec, torn.x)
    f_p = ca.jacobian(torn.ode_rhs_vec, torn.p)
    
    # Evaluate Jacobians at equilibrium
    jac_A_func = ca.Function('jac_A', [torn.x, torn.p], [f_x])
    jac_B_func = ca.Function('jac_B', [torn.x, torn.p], [f_p])
    
    A = np.array(jac_A_func(x_eq, p_vals))
    B_full = np.array(jac_B_func(x_eq, p_vals))
    B = B_full[:, f_idx:f_idx+1]
    
    print("Linearized A:\n", A)
    print("Linearized B:\n", B)
    
    # 3. Define Q and R cost matrices
    Q = np.diag([10.0, 100.0, 1.0, 10.0])
    R = np.array([[0.01]])
    
    # Solve Algebraic Riccati Equation
    P_mat = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P_mat
    
    print("Computed LQR gain K:", K)
    return K

def simulate_cartpole_lqr(K_gain, t_max=5.0, x0=None):
    if x0 is None:
        x0 = [0.0, 0.2, 0.0, 0.0]  # x=0, theta=0.2 rad, xdot=0, thetadot=0
        
    print(f"Simulating closed-loop system starting from x0={x0}...")
    
    cartpole_cl = get_cartpole_component()
    system_cl = System([cartpole_cl])
    
    # States of cartpole_cl
    x_var = cartpole_cl.states[0]      # q_cartpole_x
    theta_var = cartpole_cl.states[1]  # q_cartpole_theta
    u1_var = cartpole_cl.states[2]     # u_cartpole_u1
    u2_var = cartpole_cl.states[3]     # u_cartpole_u2
    
    # Feedback law: F = - K * x
    F_expr = -(K_gain[0, 0] * x_var + K_gain[0, 1] * theta_var + 
               K_gain[0, 2] * u1_var + K_gain[0, 3] * u2_var)
               
    system_cl.equations.append(cartpole_cl.ports["F"][0] - F_expr)
    
    dae_cl = system_cl.to_dae()
    red_cl = pantelides_pass(dae_cl)
    torn_cl = tearing_pass(red_cl)
    
    t_span = (0.0, t_max)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    sol = simulate_system(torn_cl, t_span, x0, params=None, backend='numpy', method='RK45')
    return sol, torn_cl.state_names
