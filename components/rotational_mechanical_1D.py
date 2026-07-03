"""
components/rotational_mechanical_1D.py
=======================================
1D rotational mechanical components for Braid's acausal modelling framework.

All components use CasADi SX symbolic expressions.

Port convention (rotational mechanical):
    port = [torque (effort), angle (across), angular_velocity (d_angle/dt)]

    - Effort variable:  torque           T  (applied torque)
    - Across variable:  angle            θ  (absolute rotation)
    - Derivative:       angular velocity ω = dθ/dt

Node() enforces:
    Σ torques = 0          (rotational force balance)
    angles equal           (rigid connection / compatibility)
    angular velocities equal
"""

import casadi as ca
from base import Component


class RotationalInertia(Component):
    """
    Rotational inertia (flywheel): T = I * α  →  I * d²θ/dt² - T = 0
    States: θ (angle) and ω (angular velocity).
        dθ/dt = ω
        I * dω/dt = T_net
    Port: p — torque/angle/omega connection
    """
    def __init__(self, name: str, I: float = 1.0):
        super().__init__(name)

        I_sym = ca.SX.sym(f'I_{name}')
        self.register_param('I', I_sym, default=I)

        theta, theta_dot = self.add_state(f'theta_{name}')    # angle
        omega, omega_dot = self.add_state(f'omega_{name}')    # angular velocity

        T = ca.SX.sym(f'T_{name}')

        # dθ/dt = ω  →  theta_dot - omega = 0
        # I * dω/dt = T  →  I * omega_dot - T = 0
        self.equations = [
            theta_dot - omega,
            I_sym * omega_dot - T,
        ]

        # Port: [torque, angle, angular_velocity]
        self.ports = {'p': [T, theta, theta_dot]}


# Keep old name as alias for backward compatibility
RotationalIntertia = RotationalInertia


class RotationalSpring(Component):
    """
    Torsional spring: T = k * (θ1 - θ2)
    Ports: p1, p2
    No states (algebraic).
    """
    def __init__(self, name: str, k: float = 1.0):
        super().__init__(name)

        k_sym = ca.SX.sym(f'k_{name}')
        self.register_param('k', k_sym, default=k)

        theta1 = ca.SX.sym(f'theta1_{name}')
        theta2 = ca.SX.sym(f'theta2_{name}')
        T1     = ca.SX.sym(f'T1_{name}')
        T2     = ca.SX.sym(f'T2_{name}')
        theta1_dot = ca.SX.sym(f'theta1_{name}_dot')
        theta2_dot = ca.SX.sym(f'theta2_{name}_dot')

        self.equations = [
            -k_sym * (theta1 - theta2) - T2,
            -T2 - T1,
        ]

        self.ports = {
            'p1': [T1, theta1, theta1_dot],
            'p2': [T2, theta2, theta2_dot],
        }


class RotationalDamper(Component):
    """
    Rotational viscous damper: T = c * (ω1 - ω2)
    Ports: p1, p2
    No states (algebraic).
    """
    def __init__(self, name: str, c: float = 0.1):
        super().__init__(name)

        c_sym = ca.SX.sym(f'c_{name}')
        self.register_param('c', c_sym, default=c)

        theta1 = ca.SX.sym(f'theta1_{name}')
        theta2 = ca.SX.sym(f'theta2_{name}')
        T1     = ca.SX.sym(f'T1_{name}')
        T2     = ca.SX.sym(f'T2_{name}')
        theta1_dot = ca.SX.sym(f'theta1_{name}_dot')
        theta2_dot = ca.SX.sym(f'theta2_{name}_dot')

        # Damper torque: T2 = -c*(ω1 - ω2) = -c*(theta1_dot - theta2_dot)
        self.equations = [
            -c_sym * (theta1_dot - theta2_dot) - T2,
            -T2 - T1,
        ]

        self.ports = {
            'p1': [T1, theta1, theta1_dot],
            'p2': [T2, theta2, theta2_dot],
        }


class RotationalGround(Component):
    """
    Fixed rotational anchor: angle = 0, angular velocity = 0.
    Port: p
    """
    def __init__(self, name: str):
        super().__init__(name)

        theta = ca.SX.sym(f'theta_{name}')
        T     = ca.SX.sym(f'T_{name}')
        theta_dot = ca.SX.sym(f'theta_{name}_dot')

        self.equations = [theta, theta_dot]

        self.ports = {'p': [T, theta, theta_dot]}


class Torque(Component):
    """
    External constant torque source.
    Port: p — torque/angle connection.
    """
    def __init__(self, name: str, Text: float = 0.0):
        super().__init__(name)

        T_sym = ca.SX.sym(f'Text_{name}')
        self.register_param('T', T_sym, default=Text)

        T     = ca.SX.sym(f'T_{name}')
        theta = ca.SX.sym(f'theta_{name}')
        theta_dot = ca.SX.sym(f'theta_{name}_dot')

        self.equations = [T - T_sym]

        self.ports = {'p': [T, theta, theta_dot]}