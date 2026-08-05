"""
components/hvac_thermal.py
==========================
Lumped-parameter thermal and HVAC modeling components for Braid.

Port convention (Thermal):
    port = [heat_flow (effort), temperature (across), d(temperature)/dt]

    - Effort variable:  heat flow rate Q_dot [W] (positive flows into node/component)
    - Across variable:  temperature T [K] or [°C]
    - Derivative variable: dT/dt [K/s]

Node() enforces:
    Σ Q_dot = 0  (Heat balance / KCL equivalent)
    Temperatures equal  (Thermal equilibrium)
"""

import casadi as ca
from base import Component

# ────────────────────────────────────────────────────────────────────────────
# Basic Thermal Lumped Components
# ────────────────────────────────────────────────────────────────────────────

class ThermalCapacitance(Component):
    """
    Lumped thermal capacity element (e.g. wall mass, air volume, thermal storage).
    
    Equation: C * dT/dt = Q_p
    Ports:
        'p': [Q_p, T, T_dot]
    State:
        T - Temperature [K]
    """
    def __init__(self, name: str, C: float = 1000.0):
        super().__init__(name)

        C_sym = ca.SX.sym(f'C_{name}')
        self.register_param('C', C_sym, default=C)

        T, T_dot = self.add_state(f'T_{name}')
        Q_p = ca.SX.sym(f'Q_p_{name}')

        # Balance equation: Heat flow into storage minus dynamic accumulation = 0
        self.equations = [
            Q_p - C_sym * T_dot
        ]

        self.ports = {
            'p': [Q_p, T, T_dot]
        }


class ThermalResistance(Component):
    """
    Thermal conduction / linear thermal resistance.
    
    Equation: (T_a - T_b) - R * Q_a = 0
    KCL: Q_a + Q_b = 0
    Ports:
        'a': [Q_a, T_a, T_a_dot]
        'b': [Q_b, T_b, T_b_dot]
    """
    def __init__(self, name: str, R: float = 1.0):
        super().__init__(name)

        R_sym = ca.SX.sym(f'R_{name}')
        self.register_param('R', R_sym, default=R)

        T_a = ca.SX.sym(f'T_a_{name}')
        T_b = ca.SX.sym(f'T_b_{name}')
        Q_a = ca.SX.sym(f'Q_a_{name}')
        Q_b = ca.SX.sym(f'Q_b_{name}')
        T_a_dot = ca.SX.sym(f'T_a_{name}_dot')
        T_b_dot = ca.SX.sym(f'T_b_{name}_dot')

        self.equations = [
            (T_a - T_b) - R_sym * Q_a,
            Q_a + Q_b
        ]

        self.ports = {
            'a': [Q_a, T_a, T_a_dot],
            'b': [Q_b, T_b, T_b_dot]
        }


class ThermalConvection(Component):
    """
    Convective heat transfer between a surface and a fluid.
    
    Equation: Q_a = h * A * (T_a - T_b)
    Ports:
        'a': surface port
        'b': fluid/ambient port
    """
    def __init__(self, name: str, G: float = 10.0): # G = h * A [W/K]
        super().__init__(name)

        G_sym = ca.SX.sym(f'G_{name}') # Conductance G = h * A
        self.register_param('G', G_sym, default=G)

        T_a = ca.SX.sym(f'T_a_{name}')
        T_b = ca.SX.sym(f'T_b_{name}')
        Q_a = ca.SX.sym(f'Q_a_{name}')
        Q_b = ca.SX.sym(f'Q_b_{name}')
        T_a_dot = ca.SX.sym(f'T_a_{name}_dot')
        T_b_dot = ca.SX.sym(f'T_b_{name}_dot')

        self.equations = [
            Q_a - G_sym * (T_a - T_b),
            Q_a + Q_b
        ]

        self.ports = {
            'a': [Q_a, T_a, T_a_dot],
            'b': [Q_b, T_b, T_b_dot]
        }


class ThermalRadiation(Component):
    """
    Non-linear radiative heat exchange between two surfaces/bodies.
    
    Equation: Q_a = Hr * (T_a^4 - T_b^4), where Hr = sigma * eps * A
    Ports:
        'a': surface a port
        'b': surface b port
    """
    def __init__(self, name: str, Hr: float = 5.67e-8):
        super().__init__(name)

        Hr_sym = ca.SX.sym(f'Hr_{name}')
        self.register_param('Hr', Hr_sym, default=Hr)

        T_a = ca.SX.sym(f'T_a_{name}')
        T_b = ca.SX.sym(f'T_b_{name}')
        Q_a = ca.SX.sym(f'Q_a_{name}')
        Q_b = ca.SX.sym(f'Q_b_{name}')
        T_a_dot = ca.SX.sym(f'T_a_{name}_dot')
        T_b_dot = ca.SX.sym(f'T_b_{name}_dot')

        self.equations = [
            Q_a - Hr_sym * (T_a**4 - T_b**4),
            Q_a + Q_b
        ]

        self.ports = {
            'a': [Q_a, T_a, T_a_dot],
            'b': [Q_b, T_b, T_b_dot]
        }


# ────────────────────────────────────────────────────────────────────────────
# Thermal Boundary Sources & Sensors
# ────────────────────────────────────────────────────────────────────────────

class TemperatureSource(Component):
    """
    Prescribed boundary temperature (e.g. ambient environment temperature).
    
    Equation: T_p = T_val
    Ports:
        'p': [Q_p, T_p, T_p_dot]
    """
    def __init__(self, name: str, T_val: float = 293.15):
        super().__init__(name)

        T_val_sym = ca.SX.sym(f'T_val_{name}')
        self.register_param('T_val', T_val_sym, default=T_val)

        T_p = ca.SX.sym(f'T_p_{name}')
        Q_p = ca.SX.sym(f'Q_p_{name}')
        T_p_dot = ca.SX.sym(f'T_p_{name}_dot')

        self.equations = [
            T_p - T_val_sym
        ]

        self.ports = {
            'p': [Q_p, T_p, T_p_dot]
        }


class HeatFlowSource(Component):
    """
    Prescribed heat flow source (e.g., HVAC heating element, solar radiation, occupant heat gain).
    
    Equation: Q_p = - Q_val (positive flow injected into the network)
    Ports:
        'p': [Q_p, T_p, T_p_dot]
    """
    def __init__(self, name: str, Q_val: float = 1000.0):
        super().__init__(name)

        Q_val_sym = ca.SX.sym(f'Q_val_{name}')
        self.register_param('Q_val', Q_val_sym, default=Q_val)

        T_p = ca.SX.sym(f'T_p_{name}')
        Q_p = ca.SX.sym(f'Q_p_{name}')
        T_p_dot = ca.SX.sym(f'T_p_{name}_dot')

        self.equations = [
            Q_p + Q_val_sym
        ]

        self.ports = {
            'p': [Q_p, T_p, T_p_dot]
        }


class TemperatureSensor(Component):
    """
    Observer component measuring temperature at a node.
    """
    is_sensor = True
    measure_quantity = 'across'

    def __init__(self, name: str):
        super().__init__(name)

        T_p = ca.SX.sym(f'T_p_{name}')
        Q_p = ca.SX.sym(f'Q_p_{name}')
        T_p_dot = ca.SX.sym(f'T_p_{name}_dot')

        self.ports = {
            'p': [Q_p, T_p, T_p_dot]
        }


# ────────────────────────────────────────────────────────────────────────────
# HVAC & Fluid Flow Dynamic Components
# ────────────────────────────────────────────────────────────────────────────

class MassFlowHeatAdvection(Component):
    """
    Enthalpy transport / forced fluid flow between inlet and outlet.
    
    Equation: Q_in = m_dot * Cp * (T_in - T_out)
    Ports:
        'inlet': [Q_in, T_in, T_in_dot]
        'outlet': [Q_out, T_out, T_out_dot]
    """
    def __init__(self, name: str, m_dot: float = 0.5, Cp: float = 1005.0):
        super().__init__(name)

        m_dot_sym = ca.SX.sym(f'm_dot_{name}')
        Cp_sym = ca.SX.sym(f'Cp_{name}')
        self.register_param('m_dot', m_dot_sym, default=m_dot)
        self.register_param('Cp', Cp_sym, default=Cp)

        T_in = ca.SX.sym(f'T_in_{name}')
        T_out = ca.SX.sym(f'T_out_{name}')
        Q_in = ca.SX.sym(f'Q_in_{name}')
        Q_out = ca.SX.sym(f'Q_out_{name}')
        T_in_dot = ca.SX.sym(f'T_in_{name}_dot')
        T_out_dot = ca.SX.sym(f'T_out_{name}_dot')

        self.equations = [
            Q_in - m_dot_sym * Cp_sym * (T_in - T_out),
            Q_in + Q_out
        ]

        self.ports = {
            'inlet': [Q_in, T_in, T_in_dot],
            'outlet': [Q_out, T_out, T_out_dot]
        }


class ThermalZone(Component):
    """
    Single-zone dynamic thermal model (air mass thermal capacity with external envelope port).
    
    Equations:
        C_air * dT_zone/dt = Q_port + Q_gain
    Ports:
        'port': [Q_port, T_zone, T_zone_dot]
    State:
        T_zone - Zone mean indoor temperature [K]
    """
    def __init__(self, name: str, V_zone: float = 100.0, rho_air: float = 1.2, Cp_air: float = 1005.0, Q_gain: float = 200.0):
        super().__init__(name)

        C_air_val = V_zone * rho_air * Cp_air
        C_air_sym = ca.SX.sym(f'C_air_{name}')
        Q_gain_sym = ca.SX.sym(f'Q_gain_{name}')

        self.register_param('C_air', C_air_sym, default=C_air_val)
        self.register_param('Q_gain', Q_gain_sym, default=Q_gain)

        T_zone, T_zone_dot = self.add_state(f'T_zone_{name}')
        Q_port = ca.SX.sym(f'Q_port_{name}')

        self.equations = [
            Q_port + Q_gain_sym - C_air_sym * T_zone_dot
        ]

        self.ports = {
            'port': [Q_port, T_zone, T_zone_dot]
        }


class HeatExchangerNTU(Component):
    """
    Effectiveness-NTU model of an HVAC heating/cooling coil transferring heat between fluid stream and air node.
    
    Equation:
        Q_transfer = epsilon * m_dot * Cp * (T_fluid_in - T_air)
    Ports:
        'fluid_in': [Q_f, T_fluid_in, T_f_dot]
        'air_node': [Q_a, T_air, T_a_dot]
    """
    def __init__(self, name: str, epsilon: float = 0.8, m_dot: float = 0.5, Cp: float = 1005.0):
        super().__init__(name)

        eps_sym = ca.SX.sym(f'epsilon_{name}')
        m_dot_sym = ca.SX.sym(f'm_dot_{name}')
        Cp_sym = ca.SX.sym(f'Cp_{name}')

        self.register_param('epsilon', eps_sym, default=epsilon)
        self.register_param('m_dot', m_dot_sym, default=m_dot)
        self.register_param('Cp', Cp_sym, default=Cp)

        T_fluid_in = ca.SX.sym(f'T_fluid_in_{name}')
        T_air = ca.SX.sym(f'T_air_{name}')
        Q_f = ca.SX.sym(f'Q_f_{name}')
        Q_a = ca.SX.sym(f'Q_a_{name}')
        T_f_dot = ca.SX.sym(f'T_fluid_in_{name}_dot')
        T_a_dot = ca.SX.sym(f'T_air_{name}_dot')

        # Heat leaves fluid stream (Q_f) and enters air (Q_a)
        self.equations = [
            Q_f - eps_sym * m_dot_sym * Cp_sym * (T_fluid_in - T_air),
            Q_f + Q_a
        ]

        self.ports = {
            'fluid_in': [Q_f, T_fluid_in, T_f_dot],
            'air_node': [Q_a, T_air, T_a_dot]
        }
