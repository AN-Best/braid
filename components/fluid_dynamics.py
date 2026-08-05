"""
components/fluid_dynamics.py
=============================
Incompressible and hydraulic fluid dynamics components for Braid.

Port convention (Fluid):
    port = [flow_rate (effort), pressure (across), d(pressure)/dt]

    - Effort variable:  Volume flow rate Q [m³/s] (positive flows into node/component)
    - Across variable:  Pressure P [Pa]
    - Derivative variable: dP/dt [Pa/s]

Node() enforces:
    Σ Q = 0   (Fluid volume/mass balance)
    Pressures equal (Pressure equilibrium)
"""

import casadi as ca
from base import Component

class FluidCapacitance(Component):
    """
    Hydraulic storage / accumulator / fluid compressibility volume.
    
    Equation: C_f * dP/dt = Q_p
    State: P - Fluid pressure [Pa]
    """
    def __init__(self, name: str, C_f: float = 1e-9): # C_f = V / Bulk_Modulus [m³/Pa]
        super().__init__(name)

        Cf_sym = ca.SX.sym(f'C_f_{name}')
        self.register_param('C_f', Cf_sym, default=C_f)

        P, P_dot = self.add_state(f'P_{name}')
        Q_p = ca.SX.sym(f'Q_p_{name}')

        self.equations = [
            Q_p - Cf_sym * P_dot
        ]

        self.ports = {
            'p': [Q_p, P, P_dot]
        }


class PipeResistance(Component):
    """
    Laminar / turbulent hydraulic pipe resistance.
    
    Equation: (P_a - P_b) - R_f * Q_a = 0
    KCL: Q_a + Q_b = 0
    """
    def __init__(self, name: str, R_f: float = 1e6):
        super().__init__(name)

        Rf_sym = ca.SX.sym(f'R_f_{name}')
        self.register_param('R_f', Rf_sym, default=R_f)

        P_a = ca.SX.sym(f'P_a_{name}')
        P_b = ca.SX.sym(f'P_b_{name}')
        Q_a = ca.SX.sym(f'Q_a_{name}')
        Q_b = ca.SX.sym(f'Q_b_{name}')
        P_a_dot = ca.SX.sym(f'P_a_{name}_dot')
        P_b_dot = ca.SX.sym(f'P_b_{name}_dot')

        self.equations = [
            (P_a - P_b) - Rf_sym * Q_a,
            Q_a + Q_b
        ]

        self.ports = {
            'a': [Q_a, P_a, P_a_dot],
            'b': [Q_b, P_b, P_b_dot]
        }


class ControlValve(Component):
    """
    Variable orifice / control valve flow rate model.
    
    Equation: Q_a = Cv * opening * sign(P_a - P_b) * sqrt(|P_a - P_b|)
    """
    def __init__(self, name: str, Cv: float = 1e-4, opening: float = 1.0):
        super().__init__(name)

        Cv_sym = ca.SX.sym(f'Cv_{name}')
        open_sym = ca.SX.sym(f'opening_{name}')
        self.register_param('Cv', Cv_sym, default=Cv)
        self.register_param('opening', open_sym, default=opening)

        P_a = ca.SX.sym(f'P_a_{name}')
        P_b = ca.SX.sym(f'P_b_{name}')
        Q_a = ca.SX.sym(f'Q_a_{name}')
        Q_b = ca.SX.sym(f'Q_b_{name}')
        P_a_dot = ca.SX.sym(f'P_a_{name}_dot')
        P_b_dot = ca.SX.sym(f'P_b_{name}_dot')

        dP = P_a - P_b
        # Smooth absolute value and sign for numerical stability in CasADi DAE
        smooth_abs_dP = ca.sqrt(dP**2 + 1e-6)
        
        self.equations = [
            Q_a - Cv_sym * open_sym * dP / ca.sqrt(smooth_abs_dP),
            Q_a + Q_b
        ]

        self.ports = {
            'a': [Q_a, P_a, P_a_dot],
            'b': [Q_b, P_b, P_b_dot]
        }


class HydraulicPump(Component):
    """
    Prescribed flow / displacement hydraulic pump.
    
    Equation: Q_a = - Q_flow
    """
    def __init__(self, name: str, Q_flow: float = 1e-3):
        super().__init__(name)

        Q_flow_sym = ca.SX.sym(f'Q_flow_{name}')
        self.register_param('Q_flow', Q_flow_sym, default=Q_flow)

        P_a = ca.SX.sym(f'P_a_{name}')
        Q_a = ca.SX.sym(f'Q_a_{name}')
        P_a_dot = ca.SX.sym(f'P_a_{name}_dot')

        self.equations = [
            Q_a + Q_flow_sym
        ]

        self.ports = {
            'a': [Q_a, P_a, P_a_dot]
        }


class PressureSource(Component):
    """
    Prescribed fluid boundary pressure source (e.g. atmospheric reservoir).
    """
    def __init__(self, name: str, P_val: float = 101325.0):
        super().__init__(name)

        P_val_sym = ca.SX.sym(f'P_val_{name}')
        self.register_param('P_val', P_val_sym, default=P_val)

        P_a = ca.SX.sym(f'P_a_{name}')
        Q_a = ca.SX.sym(f'Q_a_{name}')
        P_a_dot = ca.SX.sym(f'P_a_{name}_dot')

        self.equations = [
            P_a - P_val_sym
        ]

        self.ports = {
            'a': [Q_a, P_a, P_a_dot]
        }
