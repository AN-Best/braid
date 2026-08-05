"""
components/refrigeration.py
===========================
Two-phase vapor compression refrigeration cycle components for Braid.
"""

import numpy as np
import casadi as ca
from base import Component


class VaporCompressor(Component):
    """
    Refrigerant compressor dynamic model.
    
    Work input = m_dot * (h_out - h_in)
    State:
        w_comp - Compressor rotational speed [rad/s]
    """
    def __init__(self, name: str, V_disp: float = 1e-5, eta_vol: float = 0.85, eta_is: float = 0.8):
        super().__init__(name)

        V_disp_sym = ca.SX.sym(f'V_disp_{name}')
        eta_vol_sym = ca.SX.sym(f'eta_vol_{name}')
        eta_is_sym = ca.SX.sym(f'eta_is_{name}')

        self.register_param('V_disp', V_disp_sym, default=V_disp)
        self.register_param('eta_vol', eta_vol_sym, default=eta_vol)
        self.register_param('eta_is', eta_is_sym, default=eta_is)

        w_comp, w_comp_dot = self.add_state(f'w_comp_{name}')

        P_in = ca.SX.sym(f'P_in_{name}')
        P_out = ca.SX.sym(f'P_out_{name}')
        m_dot = ca.SX.sym(f'm_dot_{name}')
        P_in_dot = ca.SX.sym(f'P_in_{name}_dot')
        P_out_dot = ca.SX.sym(f'P_out_{name}_dot')

        # Mass flow rate = V_disp * speed * rho_in * eta_vol
        # Ideal gas density scaling approximation for suction pressure P_in
        rho_in = P_in / (287.0 * 293.15)
        
        self.equations = [
            m_dot - V_disp_sym * (w_comp / (2 * np.pi)) * rho_in * eta_vol_sym,
            w_comp_dot # Constant speed unless driven by external torque
        ]

        self.ports = {
            'suction': [m_dot, P_in, P_in_dot],
            'discharge': [-m_dot, P_out, P_out_dot]
        }


class ExpansionValve(Component):
    """
    Thermostatic / Electronic expansion valve throttling refrigerant flow.
    
    m_dot = Cd * A_valve * sqrt(2 * rho * (P_cond - P_evap))
    """
    def __init__(self, name: str, Cd: float = 0.6, A_valve: float = 1e-5):
        super().__init__(name)

        Cd_sym = ca.SX.sym(f'Cd_{name}')
        Av_sym = ca.SX.sym(f'A_valve_{name}')

        self.register_param('Cd', Cd_sym, default=Cd)
        self.register_param('A_valve', Av_sym, default=A_valve)

        P_in = ca.SX.sym(f'P_in_{name}')
        P_out = ca.SX.sym(f'P_out_{name}')
        m_dot_in = ca.SX.sym(f'm_dot_in_{name}')
        m_dot_out = ca.SX.sym(f'm_dot_out_{name}')
        P_in_dot = ca.SX.sym(f'P_in_{name}_dot')
        P_out_dot = ca.SX.sym(f'P_out_{name}_dot')

        dP = P_in - P_out
        smooth_abs_dP = ca.sqrt(dP**2 + 1e-4)

        self.equations = [
            m_dot_in - Cd_sym * Av_sym * ca.sqrt(2.0 * 1000.0 * smooth_abs_dP),
            m_dot_in + m_dot_out
        ]

        self.ports = {
            'inlet': [m_dot_in, P_in, P_in_dot],
            'outlet': [m_dot_out, P_out, P_out_dot]
        }
