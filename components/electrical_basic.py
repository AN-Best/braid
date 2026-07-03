"""
components/electrical_basic.py
================================
Basic electrical components for Braid's acausal modelling framework.

All components use CasADi SX symbolic expressions.

Port convention (electrical):
    port = [current (effort), voltage (across), d(voltage)/dt]

    - Effort variable:  current  i  (flows through)
    - Across variable:  voltage  v  (measured across)

Node() enforces:
    Σ currents = 0  (KCL)
    voltages equal  (KVL)
"""

import casadi as ca
from base import Component

# ────────────────────────────────────────────────────────────────────────────
# Passive Elements
# ────────────────────────────────────────────────────────────────────────────

class Resistor(Component):
    """
    Ideal resistor: V = R * I
    Ports: p (positive terminal), n (negative terminal)
    """
    def __init__(self, name: str, R: float = 1.0):
        super().__init__(name)

        R_sym = ca.SX.sym(f'R_{name}')
        self.register_param('R', R_sym, default=R)

        v_p = ca.SX.sym(f'v_p_{name}')
        v_n = ca.SX.sym(f'v_n_{name}')
        i_p = ca.SX.sym(f'i_p_{name}')
        i_n = ca.SX.sym(f'i_n_{name}')
        v_p_dot = ca.SX.sym(f'v_p_{name}_dot')
        v_n_dot = ca.SX.sym(f'v_n_{name}_dot')

        # Ohm's law + KCL
        self.equations = [
            (v_p - v_n) - R_sym * i_p,
            i_p + i_n,
        ]

        self.ports = {
            'p': [i_p, v_p, v_p_dot],
            'n': [i_n, v_n, v_n_dot],
        }


class Capacitor(Component):
    """
    Ideal capacitor: I = C * dV/dt
    Ports: p (positive terminal), n (negative terminal)
    State: v_c — voltage across capacitor
    """
    def __init__(self, name: str, C: float = 1.0):
        super().__init__(name)

        C_sym = ca.SX.sym(f'C_{name}')
        self.register_param('C', C_sym, default=C)

        v_p = ca.SX.sym(f'v_p_{name}')
        v_n = ca.SX.sym(f'v_n_{name}')
        i_p = ca.SX.sym(f'i_p_{name}')
        i_n = ca.SX.sym(f'i_n_{name}')
        v_p_dot = ca.SX.sym(f'v_p_{name}_dot')
        v_n_dot = ca.SX.sym(f'v_n_{name}_dot')

        v_c, v_c_dot = self.add_state(f'v_c_{name}')

        # i_p = C * dv_c/dt   →   i_p - C * v_c_dot = 0
        # v_c = v_p - v_n
        # KCL: i_p + i_n = 0
        self.equations = [
            v_c - (v_p - v_n),
            i_p - C_sym * v_c_dot,
            i_p + i_n,
        ]

        self.ports = {
            'p': [i_p, v_p, v_p_dot],
            'n': [i_n, v_n, v_n_dot],
        }


class Inductor(Component):
    """
    Ideal inductor: V = L * dI/dt
    Ports: p (positive terminal), n (negative terminal)
    State: i_L — current through inductor
    """
    def __init__(self, name: str, L: float = 1.0):
        super().__init__(name)

        L_sym = ca.SX.sym(f'L_{name}')
        self.register_param('L', L_sym, default=L)

        v_p = ca.SX.sym(f'v_p_{name}')
        v_n = ca.SX.sym(f'v_n_{name}')
        i_p = ca.SX.sym(f'i_p_{name}')
        i_n = ca.SX.sym(f'i_n_{name}')
        v_p_dot = ca.SX.sym(f'v_p_{name}_dot')
        v_n_dot = ca.SX.sym(f'v_n_{name}_dot')

        i_L, i_L_dot = self.add_state(f'i_L_{name}')

        # (v_p - v_n) = L * di_L/dt   →   (v_p - v_n) - L * i_L_dot = 0
        # i_p = i_L
        # KCL: i_p + i_n = 0
        self.equations = [
            i_p - i_L,
            (v_p - v_n) - L_sym * i_L_dot,
            i_p + i_n,
        ]

        self.ports = {
            'p': [i_p, v_p, v_p_dot],
            'n': [i_n, v_n, v_n_dot],
        }


# ────────────────────────────────────────────────────────────────────────────
# Sources
# ────────────────────────────────────────────────────────────────────────────

class VoltageSource(Component):
    """
    Ideal voltage source: V_p - V_n = V_src (constant).
    Ports: p (positive terminal), n (negative terminal)
    """
    def __init__(self, name: str, V: float = 1.0):
        super().__init__(name)

        V_sym = ca.SX.sym(f'V_{name}')
        self.register_param('V', V_sym, default=V)

        v_p = ca.SX.sym(f'v_p_{name}')
        v_n = ca.SX.sym(f'v_n_{name}')
        i_p = ca.SX.sym(f'i_p_{name}')
        i_n = ca.SX.sym(f'i_n_{name}')
        v_p_dot = ca.SX.sym(f'v_p_{name}_dot')
        v_n_dot = ca.SX.sym(f'v_n_{name}_dot')

        self.equations = [
            (v_p - v_n) - V_sym,
            i_p + i_n,
        ]

        self.ports = {
            'p': [i_p, v_p, v_p_dot],
            'n': [i_n, v_n, v_n_dot],
        }


class CurrentSource(Component):
    """
    Ideal current source: I_p = I_src (constant).
    Ports: p (positive terminal), n (negative terminal)
    """
    def __init__(self, name: str, I: float = 1.0):
        super().__init__(name)

        I_sym = ca.SX.sym(f'I_{name}')
        self.register_param('I', I_sym, default=I)

        v_p = ca.SX.sym(f'v_p_{name}')
        v_n = ca.SX.sym(f'v_n_{name}')
        i_p = ca.SX.sym(f'i_p_{name}')
        i_n = ca.SX.sym(f'i_n_{name}')
        v_p_dot = ca.SX.sym(f'v_p_{name}_dot')
        v_n_dot = ca.SX.sym(f'v_n_{name}_dot')

        self.equations = [
            i_p - I_sym,
            i_p + i_n,
        ]

        self.ports = {
            'p': [i_p, v_p, v_p_dot],
            'n': [i_n, v_n, v_n_dot],
        }


class ElectricalGround(Component):
    """
    Electrical ground: V = 0.
    Port: p (ground terminal)
    """
    def __init__(self, name: str):
        super().__init__(name)

        v_p = ca.SX.sym(f'v_p_{name}')
        i_p = ca.SX.sym(f'i_p_{name}')
        v_p_dot = ca.SX.sym(f'v_p_{name}_dot')

        self.equations = [v_p]

        self.ports = {'p': [i_p, v_p, v_p_dot]}


# ────────────────────────────────────────────────────────────────────────────
# Sensors
# ────────────────────────────────────────────────────────────────────────────

class VoltageSensor(Component):
    """
    Ideal voltage sensor: zero current draw, measures across variable.
    is_sensor=True → Node() treats it as passive observer.
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.is_sensor       = True
        self.measure_quantity = 'across'

        v_p = ca.SX.sym(f'v_p_{name}')
        i_p = ca.SX.sym(f'i_p_{name}')
        v_p_dot = ca.SX.sym(f'v_p_{name}_dot')

        # Sensor draws no current
        self.equations = [i_p]

        self.ports = {'p': [i_p, v_p, v_p_dot]}


class CurrentSensor(Component):
    """
    Ideal current sensor: zero voltage drop, measures current through.
    Has a state i_sens that tracks the through-current.
    """
    def __init__(self, name: str):
        super().__init__(name)

        v_p = ca.SX.sym(f'v_p_{name}')
        v_n = ca.SX.sym(f'v_n_{name}')
        i_p = ca.SX.sym(f'i_p_{name}')
        i_n = ca.SX.sym(f'i_n_{name}')
        v_p_dot = ca.SX.sym(f'v_p_{name}_dot')
        v_n_dot = ca.SX.sym(f'v_n_{name}_dot')

        i_sens, i_sens_dot = self.add_state(f'i_sens_{name}')

        self.sensor_mapping = {
            'target':  str(i_sens),
            'resolve': False,
        }

        # Zero voltage drop, current measured via i_sens
        self.equations = [
            v_p - v_n,
            i_p + i_n,
            i_sens - i_p,
        ]

        self.ports = {
            'p': [i_p, v_p, v_p_dot],
            'n': [i_n, v_n, v_n_dot],
        }
