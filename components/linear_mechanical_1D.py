"""
components/linear_mechanical_1D.py
===================================
1D translational mechanical components for Braid's acausal modelling framework.

All components use CasADi SX symbolic expressions.

Port convention (translational mechanical):
    port = [force (effort), position (across), velocity (d_position/dt)]

    - Effort variable:  force     f  (applied to / reacted at port)
    - Across variable:  position  x  (absolute position of port)
    - Derivative:       velocity  v = dx/dt

Node() enforces:
    Σ forces = 0         (Newton's 3rd law / force balance)
    positions equal      (rigid connection / compatibility)
    velocities equal     (velocity compatibility)
"""

import casadi as ca
from base import Component


class Mass(Component):
    """
    Point mass: F = m * a  →  m * d²x/dt² - F = 0
    State: x (position), with velocity v = dx/dt as the second state.

    We use two states: position x and velocity v.
        dx/dt = v
        m * dv/dt = F_net
    Port: p — force/position/velocity connection
    """
    def __init__(self, name: str, m: float = 1.0):
        super().__init__(name)

        m_sym = ca.SX.sym(f'm_{name}')
        self.register_param('m', m_sym, default=m)

        x, x_dot = self.add_state(f'x_{name}')    # position
        v, v_dot = self.add_state(f'v_{name}')    # velocity

        f = ca.SX.sym(f'f_{name}')

        # dx/dt = v  →  x_dot - v = 0
        # m * dv/dt = f  →  m * v_dot - f = 0
        self.equations = [
            x_dot - v,
            m_sym * v_dot - f,
        ]

        # Port: [force, position, velocity]
        self.ports = {'p': [f, x, x_dot]}


class Spring(Component):
    """
    Linear spring: F = k * (x1 - x2)
    Ports: p1, p2 — force/position connections at each end.
    No states (algebraic).
    """
    def __init__(self, name: str, k: float = 1.0):
        super().__init__(name)

        k_sym = ca.SX.sym(f'k_{name}')
        self.register_param('k', k_sym, default=k)

        x1 = ca.SX.sym(f'x1_{name}')
        x2 = ca.SX.sym(f'x2_{name}')
        f1 = ca.SX.sym(f'f1_{name}')
        f2 = ca.SX.sym(f'f2_{name}')
        x1_dot = ca.SX.sym(f'x1_{name}_dot')
        x2_dot = ca.SX.sym(f'x2_{name}_dot')

        # Spring force: f2 = -k*(x1-x2), f1 = -f2 (Newton's 3rd)
        self.equations = [
            -k_sym * (x1 - x2) - f2,
            -f2 - f1,
        ]

        self.ports = {
            'p1': [f1, x1, x1_dot],
            'p2': [f2, x2, x2_dot],
        }


class Damper(Component):
    """
    Linear viscous damper: F = c * (v1 - v2)
    Ports: p1, p2 — force/position connections at each end.
    No states (algebraic).
    """
    def __init__(self, name: str, c: float = 0.1):
        super().__init__(name)

        c_sym = ca.SX.sym(f'c_{name}')
        self.register_param('c', c_sym, default=c)

        x1 = ca.SX.sym(f'x1_{name}')
        x2 = ca.SX.sym(f'x2_{name}')
        f1 = ca.SX.sym(f'f1_{name}')
        f2 = ca.SX.sym(f'f2_{name}')
        x1_dot = ca.SX.sym(f'x1_{name}_dot')
        x2_dot = ca.SX.sym(f'x2_{name}_dot')

        # Damper force: f2 = -c*(v1-v2) = -c*(x1_dot - x2_dot)
        self.equations = [
            -c_sym * (x1_dot - x2_dot) - f2,
            -f2 - f1,
        ]

        self.ports = {
            'p1': [f1, x1, x1_dot],
            'p2': [f2, x2, x2_dot],
        }


class Ground(Component):
    """
    Fixed ground anchor: position = 0, velocity = 0.
    Port: p — force/position connection.
    """
    def __init__(self, name: str):
        super().__init__(name)

        x = ca.SX.sym(f'x_{name}')
        f = ca.SX.sym(f'f_{name}')
        x_dot = ca.SX.sym(f'x_{name}_dot')

        # x = 0 and x_dot = 0 (ground is fixed)
        self.equations = [x, x_dot]

        self.ports = {'p': [f, x, x_dot]}


class Force(Component):
    """
    External constant force source.
    Port: p — force/position connection.
    """
    def __init__(self, name: str, F: float = 0.0):
        super().__init__(name)

        F_sym = ca.SX.sym(f'F_{name}')
        self.register_param('F', F_sym, default=F)

        f = ca.SX.sym(f'f_{name}')
        x = ca.SX.sym(f'x_{name}')
        x_dot = ca.SX.sym(f'x_{name}_dot')

        self.equations = [f - F_sym]

        self.ports = {'p': [f, x, x_dot]}


class PositionSensor(Component):
    """
    Ideal position sensor: zero force, measures across (position).
    is_sensor=True → Node() treats it as a passive observer.
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.is_sensor        = True
        self.measure_quantity = 'across'

        f = ca.SX.sym(f'f_{name}')
        x = ca.SX.sym(f'x_{name}')
        x_dot = ca.SX.sym(f'x_{name}_dot')

        self.ports = {'p': [f, x, x_dot]}


class VelocitySensor(Component):
    """
    Ideal velocity sensor: zero force, measures derivative of across (velocity).
    is_sensor=True → Node() treats it as a passive observer.
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.is_sensor        = True
        self.measure_quantity = 'derivative'

        f = ca.SX.sym(f'f_{name}')
        x = ca.SX.sym(f'x_{name}')
        x_dot = ca.SX.sym(f'x_{name}_dot')

        self.ports = {'p': [f, x, x_dot]}
