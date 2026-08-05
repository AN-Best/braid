"""
components/power_electronics.py
================================
Electric drives, motors, and power electronics components for Braid.
"""

import casadi as ca
from base import Component

class DCMotor(Component):
    """
    Electromechanical DC Motor coupling electrical port and 1D rotational mechanical port.
    
    Electrical: V_p - V_n - R*I - L*dI/dt - Back_EMF = 0
    Rotational: Torque = K_t * I
    
    Ports:
        'el_p': positive electrical terminal [I_p, V_p, V_p_dot]
        'el_n': negative electrical terminal [I_n, V_n, V_n_dot]
        'flange': rotational mechanical port [Torque, w, w_dot]
    State:
        i_arm - Armature current [A]
    """
    def __init__(self, name: str, R: float = 1.0, L: float = 0.01, Kt: float = 0.1, Ke: float = 0.1):
        super().__init__(name)

        R_sym = ca.SX.sym(f'R_{name}')
        L_sym = ca.SX.sym(f'L_{name}')
        Kt_sym = ca.SX.sym(f'Kt_{name}')
        Ke_sym = ca.SX.sym(f'Ke_{name}')

        self.register_param('R', R_sym, default=R)
        self.register_param('L', L_sym, default=L)
        self.register_param('Kt', Kt_sym, default=Kt)
        self.register_param('Ke', Ke_sym, default=Ke)

        i_arm, i_arm_dot = self.add_state(f'i_arm_{name}')

        V_p = ca.SX.sym(f'V_p_{name}')
        V_n = ca.SX.sym(f'V_n_{name}')
        I_p = ca.SX.sym(f'I_p_{name}')
        I_n = ca.SX.sym(f'I_n_{name}')
        V_p_dot = ca.SX.sym(f'V_p_{name}_dot')
        V_n_dot = ca.SX.sym(f'V_n_{name}_dot')

        tau = ca.SX.sym(f'tau_{name}')
        w = ca.SX.sym(f'w_{name}')
        w_dot = ca.SX.sym(f'w_{name}_dot')

        # Back EMF = Ke * w
        # Electrical Kirchhoff: (V_p - V_n) - R * i_arm - L * i_arm_dot - Ke * w = 0
        self.equations = [
            I_p - i_arm,
            I_p + I_n,
            (V_p - V_n) - R_sym * i_arm - L_sym * i_arm_dot - Ke_sym * w,
            tau + Kt_sym * i_arm  # Mechanical torque generation (opposite sign convention for output flange)
        ]

        self.ports = {
            'el_p': [I_p, V_p, V_p_dot],
            'el_n': [I_n, V_n, V_n_dot],
            'flange': [tau, w, w_dot]
        }


class EquivalentCircuitBattery(Component):
    """
    Dynamic lithium-ion equivalent circuit battery cell with State of Charge (SOC).
    
    States:
        soc - State of charge [0 to 1]
        v_rc - Polarization capacitor voltage [V]
    """
    def __init__(self, name: str, Q_capacity: float = 36000.0, Voc_ref: float = 3.7, R0: float = 0.01, R1: float = 0.02, C1: float = 1000.0):
        super().__init__(name)

        Q_sym = ca.SX.sym(f'Q_cap_{name}')
        Voc_sym = ca.SX.sym(f'Voc_ref_{name}')
        R0_sym = ca.SX.sym(f'R0_{name}')
        R1_sym = ca.SX.sym(f'R1_{name}')
        C1_sym = ca.SX.sym(f'C1_{name}')

        self.register_param('Q_capacity', Q_sym, default=Q_capacity)
        self.register_param('Voc_ref', Voc_sym, default=Voc_ref)
        self.register_param('R0', R0_sym, default=R0)
        self.register_param('R1', R1_sym, default=R1)
        self.register_param('C1', C1_sym, default=C1)

        soc, soc_dot = self.add_state(f'soc_{name}')
        v_rc, v_rc_dot = self.add_state(f'v_rc_{name}')

        V_p = ca.SX.sym(f'V_p_{name}')
        V_n = ca.SX.sym(f'V_n_{name}')
        I_p = ca.SX.sym(f'I_p_{name}')
        I_n = ca.SX.sym(f'I_n_{name}')
        V_p_dot = ca.SX.sym(f'V_p_{name}_dot')
        V_n_dot = ca.SX.sym(f'V_n_{name}_dot')

        # SOC dynamic rate of change: d(SOC)/dt = - I_p / Q_capacity
        # Polarization dynamic voltage: dv_rc/dt = (I_p - v_rc / R1) / C1
        # Terminal voltage: V_p - V_n = Voc(soc) - v_rc - I_p * R0
        Voc_soc = Voc_sym + 0.5 * (soc - 0.5) # Linear Voc-SOC approximation curve
        
        self.equations = [
            soc_dot + I_p / Q_sym,
            v_rc_dot - (I_p - v_rc / R1_sym) / C1_sym,
            (V_p - V_n) - (Voc_soc - v_rc - I_p * R0_sym),
            I_p + I_n
        ]

        self.ports = {
            'p': [I_p, V_p, V_p_dot],
            'n': [I_n, V_n, V_n_dot]
        }
