"""
components/control_blocks.py
=============================
Control system & signal blocks for Braid multi-physics models.
"""

import casadi as ca
from base import Component

class PIDController(Component):
    """
    Continuous PID controller with integral state accumulation.
    
    Inputs: error = target - measured
    Output: u = Kp * error + Ki * integral_error + Kd * d_error/dt
    State:
        x_int - Integral of error
    """
    def __init__(self, name: str, Kp: float = 1.0, Ki: float = 0.1, Kd: float = 0.05):
        super().__init__(name)

        Kp_sym = ca.SX.sym(f'Kp_{name}')
        Ki_sym = ca.SX.sym(f'Ki_{name}')
        Kd_sym = ca.SX.sym(f'Kd_{name}')

        self.register_param('Kp', Kp_sym, default=Kp)
        self.register_param('Ki', Ki_sym, default=Ki)
        self.register_param('Kd', Kd_sym, default=Kd)

        x_int, x_int_dot = self.add_state(f'x_int_{name}')

        setpoint = ca.SX.sym(f'setpoint_{name}')
        measured = ca.SX.sym(f'measured_{name}')
        u_out = ca.SX.sym(f'u_out_{name}')

        err = setpoint - measured
        
        # Integral rate of change = error
        # Output control signal u_out
        self.equations = [
            x_int_dot - err,
            u_out - (Kp_sym * err + Ki_sym * x_int)
        ]

        self.ports = {
            'setpoint': [0.0, setpoint, 0.0],
            'measured': [0.0, measured, 0.0],
            'out': [0.0, u_out, 0.0]
        }


class FirstOrderFilter(Component):
    """
    First-order low pass signal filter: Tau * dy/dt + y = u
    State:
        y_filt - Filtered signal output
    """
    def __init__(self, name: str, Tau: float = 1.0):
        super().__init__(name)

        Tau_sym = ca.SX.sym(f'Tau_{name}')
        self.register_param('Tau', Tau_sym, default=Tau)

        y_filt, y_filt_dot = self.add_state(f'y_filt_{name}')
        u_in = ca.SX.sym(f'u_in_{name}')

        self.equations = [
            Tau_sym * y_filt_dot + y_filt - u_in
        ]

        self.ports = {
            'in': [0.0, u_in, 0.0],
            'out': [0.0, y_filt, 0.0]
        }
