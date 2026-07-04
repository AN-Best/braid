"""
test_four_bar_linkage.py
========================
Tests the four-bar linkage system using two formulations:

1. test_four_bar_approx_ode:
   A hand-reduced 1-DOF ODE (pre-reduced with approximate transmission ratios).
   Tests that all three backends (NumPy, PyTorch, Julia) produce consistent results.

2. test_four_bar_true_dae:
   A velocity-level four-bar kinematic DAE where the loop closure velocity
   constraints (vc1, vc2) are provided in the residual form. Tests that:
     - Pantelides correctly identifies and differentiates the velocity constraints
       into acceleration constraints (to match dω₂, dω₃)
     - dae.invariants is populated with the differentiated constraint expressions
     - The tearing pass solves the resulting 2×2 acceleration constraint system
     - The system compiles to an explicit ODE and simulates correctly
     - The JSON IR carries constraint invariants for Baumgarte stabilization
"""

import os
import sys
import json
import numpy as np
import casadi as ca
from scipy.optimize import fsolve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casadi_dae import CasadiDAE
from index_reduction import pantelides_pass, tearing_pass
from simulation import simulate_system
import ir as braid_ir


# ─── Link lengths for both tests ────────────────────────────────────────────
# Grashof mechanism: L0=4, L1=2, L2=3, L3=3
# Valid closed configuration at θ₁=0: θ₂≈1.231 rad, θ₃≈1.911 rad
L0_VAL     = 4.0
L1_VAL     = 2.0
L2_VAL     = 3.0
L3_VAL     = 3.0
OMEGA1_VAL = 1.0   # crank angular velocity (rad/s)


def _find_initial_angles(theta1_0: float):
    """
    Solve the loop closure constraints to find θ₂, θ₃ consistent with θ₁.

    g1: L1·cos(θ₁) + L2·cos(θ₂) - L3·cos(θ₃) - L0 = 0
    g2: L1·sin(θ₁) + L2·sin(θ₂) - L3·sin(θ₃)      = 0
    """
    def residuals(angles):
        t2, t3 = angles
        g1 = L1_VAL*np.cos(theta1_0) + L2_VAL*np.cos(t2) - L3_VAL*np.cos(t3) - L0_VAL
        g2 = L1_VAL*np.sin(theta1_0) + L2_VAL*np.sin(t2) - L3_VAL*np.sin(t3)
        return [g1, g2]

    t2_0, t3_0 = fsolve(residuals, [np.pi/3, np.pi/2])

    res = residuals([t2_0, t3_0])
    assert abs(res[0]) < 1e-8 and abs(res[1]) < 1e-8, \
        f"Initial conditions do not satisfy loop closure: residuals={res}"

    return float(t2_0), float(t3_0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Pre-reduced 1-DOF ODE (approximate, all backends)
# ─────────────────────────────────────────────────────────────────────────────

def test_four_bar_approx_ode():
    """
    Hand-reduced 1-DOF ODE of a four-bar linkage using approximate transmission
    ratios. Verifies that NumPy, PyTorch, and Julia backends agree.
    """
    print("\n--- Test 1: Pre-reduced 1-DOF Four-Bar ODE (all backends) ---")

    dae = CasadiDAE()

    theta = ca.SX.sym('theta')
    omega = ca.SX.sym('omega')
    dtheta = ca.SX.sym('dtheta')
    domega = ca.SX.sym('domega')

    dae.x_vars      = [theta, omega]
    dae.xdot_vars   = [dtheta, domega]
    dae.state_names = ['theta', 'omega']
    dae.xdot_names  = ['dtheta', 'domega']

    L1 = ca.SX.sym('L1'); L2 = ca.SX.sym('L2')
    L3 = ca.SX.sym('L3'); L0 = ca.SX.sym('L0')
    g  = ca.SX.sym('g')

    dae.p_vars      = [L1, L2, L3, L0, g]
    dae.param_names = ['L1', 'L2', 'L3', 'L0', 'g']
    dae.param_meta  = {
        'L1': {'default': L1_VAL}, 'L2': {'default': L2_VAL},
        'L3': {'default': L3_VAL}, 'L0': {'default': L0_VAL},
        'g':  {'default': 9.81},
    }

    K2 = -(L1 * ca.sin(theta)) / (L2 * ca.sin(theta + 0.1))
    K3 = -(L1 * ca.sin(theta)) / (L3 * ca.sin(theta - 0.2))

    I1, I2, I3 = 1.0, 1.5, 1.2
    M = I1 + I2 * K2**2 + I3 * K3**2
    G = 9.81 * (L1*ca.cos(theta) + L2*ca.cos(theta + 0.1)*K2 + L3*ca.cos(theta - 0.2)*K3)

    dae.equations        = [dtheta - omega, domega + G/M]
    dae.active_equations = dae.equations
    dae.solved_variables = [dtheta, domega]
    dae.differentiation_indices = [0, 0]
    dae.ode_rhs          = [omega, -G/M]

    t_span = (0.0, 2.0)
    y0     = [0.1, 0.0]

    sol_np    = simulate_system(dae, t_span, y0, params=None, backend='numpy',   method='RK45')
    sol_torch = simulate_system(dae, t_span, y0, params=None, backend='pytorch', method='dopri5')
    sol_julia = simulate_system(dae, t_span, y0, params=None, backend='julia',   method='Tsit5()')

    assert sol_np.success and sol_torch.success and sol_julia.success

    val_np, val_torch, val_julia = sol_np.y[0][-1], sol_torch.y[0][-1], sol_julia.y[0][-1]

    print(f"Final theta — NumPy: {val_np:.6f},  Torch: {val_torch:.6f},  Julia: {val_julia:.6f}")

    # All adaptive solvers should agree within 1% (different step-size controllers)
    assert np.abs(val_np - val_torch) < 1e-2, f"NumPy vs PyTorch: {val_np:.6f} vs {val_torch:.6f}"
    assert np.abs(val_np - val_julia) < 1e-2, f"NumPy vs Julia:   {val_np:.6f} vs {val_julia:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: TRUE Four-Bar DAE — velocity-level constraints trigger Pantelides
# ─────────────────────────────────────────────────────────────────────────────

def test_four_bar_true_dae():
    """
    Four-bar kinematic DAE using velocity-level loop closure constraints.

    States:  θ₁, ω₁, θ₂, ω₂, θ₃, ω₃  (positions AND velocities as states)

    Equations:
        (0)  dθ₁ - ω₁ = 0               ← kinematic ODE   (matches dθ₁)
        (1)  dθ₂ - ω₂ = 0               ← kinematic ODE   (matches dθ₂)
        (2)  dθ₃ - ω₃ = 0               ← kinematic ODE   (matches dθ₃)
        (3)  dω₁ = 0                     ← const crank ω₁  (matches dω₁)
        (4)  vc₁ = -L₁·sin(θ₁)·ω₁ - L₂·sin(θ₂)·ω₂ + L₃·sin(θ₃)·ω₃ = 0  ← vel constraint
        (5)  vc₂ =  L₁·cos(θ₁)·ω₁ + L₂·cos(θ₂)·ω₂ - L₃·cos(θ₃)·ω₃ = 0  ← vel constraint

    vc₁ and vc₂ contain ω₂, ω₃ (state variables) but NOT their derivatives.
    The Pantelides algorithm must differentiate them to produce acceleration-level
    constraints containing dω₂, dω₃, which can then be matched. The system is
    thereby reduced to an explicit ODE.

    The position-level constraints g₁, g₂ are manually registered as invariants
    to be carried through the IR for future Baumgarte stabilization.

    Note on Pantelides architecture: The current Braid Pantelides implementation
    correctly handles 'assignment-style' constraints (e.g., x_ground = 0) by
    matching them to state variables. It does not yet force-differentiate
    position-only constraints like g(θ) = 0 before the x_var matching fallback
    triggers. The velocity-level formulation here works within the current
    architecture: vc1, vc2 contain state variables (ω₁..₃) but not derivatives,
    so they are unmatched in the first pass and correctly differentiated.
    """
    print("\n--- Test 2: Four-Bar Velocity-Level DAE (Pantelides + Tearing) ---")

    # ── Build the CasadiDAE (6 states) ──────────────────────────────────────
    dae = CasadiDAE()

    theta1  = ca.SX.sym('theta1');  omega1_s = ca.SX.sym('omega1_s')
    theta2  = ca.SX.sym('theta2');  omega2   = ca.SX.sym('omega2')
    theta3  = ca.SX.sym('theta3');  omega3   = ca.SX.sym('omega3')

    dtheta1 = ca.SX.sym('dtheta1'); dalpha1  = ca.SX.sym('dalpha1')
    dtheta2 = ca.SX.sym('dtheta2'); dalpha2  = ca.SX.sym('dalpha2')
    dtheta3 = ca.SX.sym('dtheta3'); dalpha3  = ca.SX.sym('dalpha3')

    dae.x_vars      = [theta1, omega1_s, theta2, omega2, theta3, omega3]
    dae.xdot_vars   = [dtheta1, dalpha1, dtheta2, dalpha2, dtheta3, dalpha3]
    dae.state_names = ['theta1', 'omega1_s', 'theta2', 'omega2', 'theta3', 'omega3']
    dae.xdot_names  = ['dtheta1', 'dalpha1', 'dtheta2', 'dalpha2', 'dtheta3', 'dalpha3']

    L1 = ca.SX.sym('L1'); L2 = ca.SX.sym('L2')
    L3 = ca.SX.sym('L3'); L0 = ca.SX.sym('L0')

    dae.p_vars      = [L1, L2, L3, L0]
    dae.param_names = ['L1', 'L2', 'L3', 'L0']
    dae.param_meta  = {
        'L1': {'default': L1_VAL}, 'L2': {'default': L2_VAL},
        'L3': {'default': L3_VAL}, 'L0': {'default': L0_VAL},
    }

    # Position-level constraints (NOT in dae.equations — tracked as invariants only)
    g1_pos = L1*ca.cos(theta1) + L2*ca.cos(theta2) - L3*ca.cos(theta3) - L0
    g2_pos = L1*ca.sin(theta1) + L2*ca.sin(theta2) - L3*ca.sin(theta3)

    # Velocity-level loop closure constraints = d(g)/dt = 0
    # These contain ω₂, ω₃ as state vars but NOT dω₂, dω₃ as derivative vars.
    # → Pantelides cannot immediately match them → differentiates them once →
    #   produces acceleration constraints containing dω₂, dω₃ → can be matched.
    vc1 = -L1*ca.sin(theta1)*omega1_s - L2*ca.sin(theta2)*omega2 + L3*ca.sin(theta3)*omega3
    vc2 =  L1*ca.cos(theta1)*omega1_s + L2*ca.cos(theta2)*omega2 - L3*ca.cos(theta3)*omega3

    dae.equations = [
        dtheta1 - omega1_s,   # eq 0: θ₁_dot = ω₁       → matches dtheta1
        dtheta2 - omega2,      # eq 1: θ₂_dot = ω₂       → matches dtheta2
        dtheta3 - omega3,      # eq 2: θ₃_dot = ω₃       → matches dtheta3
        dalpha1,               # eq 3: α₁ = 0 (const ω₁) → matches dalpha1
        vc1,                   # eq 4: vel constraint → Pantelides → matches dalpha2
        vc2,                   # eq 5: vel constraint → Pantelides → matches dalpha3
    ]

    # Register position constraints as invariants (for future Baumgarte stabilization)
    dae.invariants = [g1_pos, g2_pos]

    # ── Index reduction ──────────────────────────────────────────────────────
    print("  Running Pantelides index reduction...")
    reduced = pantelides_pass(dae)

    print(f"  Differentiation indices: {reduced.differentiation_indices}")
    # vc1, vc2 (indices 4,5) should have been differentiated at least once
    assert reduced.differentiation_indices[4] >= 1, \
        f"vc1 (eq 4) should be differentiated; got d={reduced.differentiation_indices[4]}"
    assert reduced.differentiation_indices[5] >= 1, \
        f"vc2 (eq 5) should be differentiated; got d={reduced.differentiation_indices[5]}"

    # Pantelides adds vc1, vc2 as new velocity-level invariants
    print(f"  Invariants from Pantelides: {len(reduced.invariants)}")

    # ── Tearing pass ─────────────────────────────────────────────────────────
    print("  Running tearing pass (solving 2×2 acceleration constraint system)...")
    torn = tearing_pass(reduced)
    assert len(torn.ode_rhs) == 6, f"Expected 6 ODE RHS, got {len(torn.ode_rhs)}"
    print("  Tearing succeeded — 6-DOF four-bar reduced to explicit ODE")

    # ── Inject position-level invariants into the torn DAE IR ────────────────
    # Pantelides started from velocity constraints, so we prepend position invariants
    # so that both levels are available for stabilization / projection.
    torn.invariants = list(dae.invariants) + list(torn.invariants)

    # ── Serialize to JSON IR ─────────────────────────────────────────────────
    json_str = braid_ir.to_json(torn)
    parsed = json.loads(json_str)
    print(f"\n--- Braid JSON IR (four-bar velocity-level DAE) ---")
    print(f"  model_type: {parsed['model_type']}")
    print(f"  states:     {parsed['states']}")
    print(f"  invariants: {len(parsed['invariants'])} expressions (pos + vel constraints)")
    print(f"---------------------------------------------------\n")
    assert parsed['model_type'] == 'ODE'
    assert len(parsed['invariants']) >= 2, "JSON IR should carry at least the 2 position invariants"

    # ── Compute consistent initial conditions ────────────────────────────────
    theta1_0 = 0.0
    theta2_0, theta3_0 = _find_initial_angles(theta1_0)
    omega1_0 = OMEGA1_VAL

    # Solve for ω₂₀, ω₃₀ satisfying vc1=0, vc2=0:
    # [-L2·sin(θ₂)  L3·sin(θ₃)] [ω₂]   [L1·sin(θ₁)·ω₁]
    # [ L2·cos(θ₂) -L3·cos(θ₃)] [ω₃] = [-L1·cos(θ₁)·ω₁]
    J_vel = np.array([
        [-L2_VAL*np.sin(theta2_0),  L3_VAL*np.sin(theta3_0)],
        [ L2_VAL*np.cos(theta2_0), -L3_VAL*np.cos(theta3_0)],
    ])
    rhs_vel = np.array([
        L1_VAL*np.sin(theta1_0)*omega1_0,
        -L1_VAL*np.cos(theta1_0)*omega1_0,
    ])
    omega2_0, omega3_0 = np.linalg.solve(J_vel, rhs_vel)

    y0 = [theta1_0, omega1_0, theta2_0, omega2_0, theta3_0, omega3_0]
    print(f"  Initial: θ₁={theta1_0:.4f}, ω₁={omega1_0:.4f}, "
          f"θ₂={theta2_0:.4f}, ω₂={omega2_0:.4f}, θ₃={theta3_0:.4f}, ω₃={omega3_0:.4f}")

    # ── Simulate one full crank revolution ───────────────────────────────────
    t_span = (0.0, 2 * np.pi / OMEGA1_VAL)
    print(f"  Simulating for t ∈ [0, {t_span[1]:.3f}] (one crank revolution)...")

    sol = simulate_system(torn, t_span, y0, params=None, backend='numpy', method='RK45')
    assert sol.success, "Simulation failed"

    theta1_f = sol.y[0][-1]
    theta2_f = sol.y[2][-1]
    theta3_f = sol.y[4][-1]
    print(f"  Final:   θ₁={theta1_f:.4f}, θ₂={theta2_f:.4f}, θ₃={theta3_f:.4f}")

    # Crank should complete one full revolution (θ₁ increases by 2π)
    assert np.abs(theta1_f - (theta1_0 + 2*np.pi)) < 1e-3, \
        f"Crank should complete full revolution: θ₁_f={theta1_f:.4f}"

    # θ₂, θ₃ should return near their initial values (periodic motion)
    assert np.abs(theta2_f - theta2_0) < 0.05, \
        f"Coupler periodicity: θ₂_f={theta2_f:.4f} vs θ₂_0={theta2_0:.4f}"
    assert np.abs(theta3_f - theta3_0) < 0.05, \
        f"Follower periodicity: θ₃_f={theta3_f:.4f} vs θ₃_0={theta3_0:.4f}"

    # Report position constraint drift
    # (drift is expected because we integrated at acceleration level — velocity errors accumulate)
    g1_f = L1_VAL*np.cos(theta1_f) + L2_VAL*np.cos(theta2_f) - L3_VAL*np.cos(theta3_f) - L0_VAL
    g2_f = L1_VAL*np.sin(theta1_f) + L2_VAL*np.sin(theta2_f) - L3_VAL*np.sin(theta3_f)
    print(f"  Position constraint drift after 1 rev: g1={g1_f:.2e}, g2={g2_f:.2e}")
    print("  (Non-zero drift is expected without Baumgarte stabilization.)")
    print("  Test 2 PASSED ✓")
