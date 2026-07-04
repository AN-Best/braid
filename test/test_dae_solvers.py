"""
test_dae_solvers.py
===================
Tests the NumPy (solve-dae) and PyTorch (torchdae) DAE integration backends.
"""

import os
import sys
import numpy as np
import casadi as ca
import pytest
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casadi_dae import CasadiDAE
from index_reduction import tearing_pass
from simulation import simulate_system
import ir as braid_ir

# Link lengths
L0_VAL = 4.0
L1_VAL = 2.0
L2_VAL = 3.0
L3_VAL = 3.0
OMEGA1_VAL = 1.0


def _find_initial_angles(theta1_0: float):
    from scipy.optimize import fsolve
    def residuals(angles):
        t2, t3 = angles
        g1 = L1_VAL*np.cos(theta1_0) + L2_VAL*np.cos(t2) - L3_VAL*np.cos(t3) - L0_VAL
        g2 = L1_VAL*np.sin(theta1_0) + L2_VAL*np.sin(t2) - L3_VAL*np.sin(t3)
        return [g1, g2]
    t2_0, t3_0 = fsolve(residuals, [np.pi/3, np.pi/2])
    return float(t2_0), float(t3_0)


def test_four_bar_dae_solver():
    """
    Test the four-bar linkage simulated directly as a DAE.
    We use a 5-state formulation:
      States: theta1, theta2, omega2, theta3, omega3
      Derivatives: dtheta1, dtheta2, dalpha2, dtheta3, dalpha3
    """
    print("\n--- Test DAE Solver: Four-Bar Linkage ---")

    dae = CasadiDAE()

    theta1 = ca.SX.sym('theta1')
    theta2 = ca.SX.sym('theta2'); omega2 = ca.SX.sym('omega2')
    theta3 = ca.SX.sym('theta3'); omega3 = ca.SX.sym('omega3')

    dtheta1 = ca.SX.sym('dtheta1')
    dtheta2 = ca.SX.sym('dtheta2'); dalpha2 = ca.SX.sym('dalpha2')
    dtheta3 = ca.SX.sym('dtheta3'); dalpha3 = ca.SX.sym('dalpha3')

    dae.x_vars      = [theta1, theta2, omega2, theta3, omega3]
    dae.xdot_vars   = [dtheta1, dtheta2, dalpha2, dtheta3, dalpha3]
    dae.state_names = ['theta1', 'theta2', 'omega2', 'theta3', 'omega3']
    dae.xdot_names  = ['dtheta1', 'dtheta2', 'dalpha2', 'dtheta3', 'dalpha3']

    L1 = ca.SX.sym('L1'); L2 = ca.SX.sym('L2')
    L3 = ca.SX.sym('L3'); L0 = ca.SX.sym('L0')
    omega1_param = ca.SX.sym('omega1_param')

    dae.p_vars      = [L1, L2, L3, L0, omega1_param]
    dae.param_names = ['L1', 'L2', 'L3', 'L0', 'omega1_param']
    dae.param_meta  = {
        'L1': {'default': L1_VAL}, 'L2': {'default': L2_VAL},
        'L3': {'default': L3_VAL}, 'L0': {'default': L0_VAL},
        'omega1_param': {'default': OMEGA1_VAL},
    }

    vc1 = -L1*ca.sin(theta1)*omega1_param - L2*ca.sin(theta2)*omega2 + L3*ca.sin(theta3)*omega3
    vc2 =  L1*ca.cos(theta1)*omega1_param + L2*ca.cos(theta2)*omega2 - L3*ca.cos(theta3)*omega3

    # DAE equations (dalpha2 and dalpha3 are not solved, making this a true DAE)
    dae.equations = [
        dtheta1 - omega1_param,
        dtheta2 - omega2,
        dtheta3 - omega3,
        vc1,
        vc2,
    ]
    dae.active_equations = dae.equations
    dae.solved_variables = [dtheta1, dtheta2, dtheta3, None, None]

    print("  Running tearing pass with DAE support...")
    torn = tearing_pass(dae, allow_dae=True)
    assert len(torn.residuals) == 5, f"Expected 5 residuals, got {len(torn.residuals)}"
    
    # Serialize and check model_type
    json_str = braid_ir.to_json(torn)
    parsed = json.loads(json_str)
    assert parsed['model_type'] == 'DAE'
    print("  DAE model successfully compiled and serialized!")

    # Initial conditions
    theta1_0 = 0.0
    theta2_0, theta3_0 = _find_initial_angles(theta1_0)
    omega1_0 = OMEGA1_VAL

    J_vel = np.array([
        [-L2_VAL*np.sin(theta2_0),  L3_VAL*np.sin(theta3_0)],
        [ L2_VAL*np.cos(theta2_0), -L3_VAL*np.cos(theta3_0)],
    ])
    rhs_vel = np.array([
        L1_VAL*np.sin(theta1_0)*omega1_0,
        -L1_VAL*np.cos(theta1_0)*omega1_0,
    ])
    omega2_0, omega3_0 = np.linalg.solve(J_vel, rhs_vel)

    y0 = [theta1_0, theta2_0, omega2_0, theta3_0, omega3_0]
    yp0 = [omega1_0, omega2_0, 0.0, omega3_0, 0.0]  # Initial derivatives consistent with F=0

    # ── Test 1: NumPy solve_dae Backend ──────────────────────────────────────
    print("  Simulating via NumPy solve_dae backend...")
    t_span = (0.0, 1.0)
    sol_np = simulate_system(torn, t_span, y0, params=None, backend='numpy', yp0=yp0, rtol=1e-8, atol=1e-10)
    assert sol_np.success
    print("  NumPy DAE simulation succeeded!")

    # ── Test 2: PyTorch torchdae Backend ─────────────────────────────────────
    print("  Simulating via PyTorch torchdae backend...")
    sol_torch = simulate_system(torn, t_span, y0, params=None, backend='torch', yp0=yp0, device='cpu', rtol=1e-8, atol=1e-10)
    assert sol_torch.success
    print("  PyTorch DAE simulation succeeded!")

    # ── Test 3: Julia DAE Backend ────────────────────────────────────────────
    print("  Simulating via Julia DAE backend...")
    sol_julia = simulate_system(torn, t_span, y0, params=None, backend='julia', yp0=yp0, rtol=1e-6, atol=1e-6)

    assert sol_julia.success
    # Ensure they agree closely
    assert np.allclose(sol_np.y[:, -1], sol_julia.y[:, -1], atol=1e-4)
    print("  Julia DAE simulation succeeded!")


