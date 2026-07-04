"""
base.py
=======
Core abstractions for Braid's acausal component-based modelling.

Uses CasADi SX for all symbolic expressions (replaces SymPy).

Design
------
* Component: declares states, parameters, equations, and connection ports.
  - States are pairs (x, xdot) of ca.SX scalar symbols.
  - Equations are ca.SX residual expressions (= 0).
  - Ports are dicts: {port_name: [effort_sym, across_sym, deriv_of_across_sym]}

* Node(system, comp_set): applies KCL/KVL connection constraints.
  - effort sum = 0  (Kirchhoff's current law / Newton's force balance)
  - across values equal  (Kirchhoff's voltage law / compatibility)
  - derivative-of-across values equal  (velocity compatibility)

* System(components): aggregates components.
  - to_dae() assembles and returns a CasadiDAE.
"""

import casadi as ca
from casadi_dae import CasadiDAE


class Component:
    """
    Base class for acausal physical components.

    Subclasses declare:
        - States via add_state(name) → (x, xdot)
        - Parameters via register_param(name, sym, default)
        - Equations (residuals) appended to self.equations
        - Ports assigned to self.ports as {name: [effort, across, dacross_dt]}
    """

    def __init__(self, name: str):
        self.name: str          = name
        self.states: list       = []   # ca.SX scalar — state variables x_i
        self.state_dots: list   = []   # ca.SX scalar — derivative symbols xdot_i
        self.params: list       = []   # ca.SX scalar — parameter symbols
        self._param_meta: dict  = {}   # {param_name: {sym, default}}
        self.equations: list    = []   # ca.SX residual expressions (= 0)
        self.ports: dict        = {}   # {port_name: [effort, across, dacross_dt]}
        self.domain: str        = 'continuous'

    def add_state(self, name: str):
        """
        Declares a new state variable and its time derivative symbol.

        Returns (x, xdot) as CasADi SX scalars.
        The pair is automatically registered in self.states / self.state_dots.
        """
        x    = ca.SX.sym(f'{name}')
        xdot = ca.SX.sym(f'{name}_dot')
        self.states.append(x)
        self.state_dots.append(xdot)
        return x, xdot

    def register_param(self, param_name: str, sym: ca.SX, default: float):
        """Registers a parameter symbol with its default value."""
        self.params.append(sym)
        self._param_meta[param_name] = {'sym': sym, 'default': default}

    # ------------------------------------------------------------------
    # Optional sensor interface (keeps backward compat with existing tests)
    # ------------------------------------------------------------------
    is_sensor: bool = False
    measure_quantity: str = 'across'


def Node(system, comp_set):
    """
    Apply KCL/KVL connection constraints across a set of component ports.

    comp_set: list of (Component, port_name) pairs to connect.
    Sensor components (is_sensor=True) are treated as passive observers —
    their effort is forced to zero and they receive the across value.

    Constraints added to system.equations:
        Σ effort_i = 0                    (effort summation)
        across_i   = across_ref  ∀ i≠0   (across equality)
        dacross_i  = dacross_ref ∀ i≠0   (derivative equality)
    """
    sensors  = []
    regular  = []
    for comp, port in comp_set:
        if getattr(comp, 'is_sensor', False):
            sensors.append((comp, port))
        else:
            regular.append((comp, port))

    if not regular:
        return

    # Reference node: first regular component
    ref_effort   = regular[0][0].ports[regular[0][1]][0]
    ref_across   = regular[0][0].ports[regular[0][1]][1]
    ref_dacross  = regular[0][0].ports[regular[0][1]][2]

    effort_sum   = ref_effort

    for comp, port in regular[1:]:
        effort_i  = comp.ports[port][0]
        across_i  = comp.ports[port][1]
        dacross_i = comp.ports[port][2]

        effort_sum = effort_sum + effort_i
        system.equations.append(ref_across  - across_i)
        system.equations.append(ref_dacross - dacross_i)

    system.equations.append(effort_sum)

    # Sensor mappings
    if not hasattr(system, 'sensor_mappings'):
        system.sensor_mappings = {}

    for sensor, port in sensors:
        quantity = getattr(sensor, 'measure_quantity', 'across')
        if quantity == 'across':
            target = ref_across
        elif quantity == 'derivative':
            target = ref_dacross
        else:
            target = None

        system.sensor_mappings[sensor.name] = {
            'type':    sensor.__class__.__name__,
            'target':  str(target) if target is not None else None,
            'resolve': True,
        }


class System:
    """
    Aggregates a list of components into a simulatable system.

    Collects all states, parameters, and equations from components,
    plus any inter-component equations added by Node() calls.
    """

    def __init__(self, components: list):
        self.sensors    = []
        self.components = []
        for c in components:
            if getattr(c, 'is_sensor', False):
                self.sensors.append(c)
            else:
                self.components.append(c)

        self.elsd: dict             = {c.name: c for c in self.components}
        self.equations: list        = []   # inter-component (Node) equations
        self.sensor_mappings: dict  = {}

    def to_dae(self) -> CasadiDAE:
        """
        Assembles all component equations and Node connection constraints
        into a CasadiDAE.

        Returns a CasadiDAE ready for index reduction.
        """
        dae = CasadiDAE()

        for comp in self.components:
            # States
            for x, xdot in zip(comp.states, comp.state_dots):
                name = str(x)
                dname = str(xdot)
                dae.x_vars.append(x)
                dae.xdot_vars.append(xdot)
                dae.state_names.append(name)
                dae.xdot_names.append(dname)
                # Build derivative chain: x → xdot
                dae.derivative_chain[x] = xdot

            # Params
            for param_name, meta in comp._param_meta.items():
                sym = meta['sym']
                pname = str(sym)
                dae.p_vars.append(sym)
                dae.param_names.append(pname)
                dae.param_meta[pname] = {
                    'component': comp.name,
                    'param':     param_name,
                    'default':   meta['default'],
                }

            # Component equations
            dae.equations.extend(comp.equations)

            # Component metadata
            dae.components.append({
                'name':   comp.name,
                'domain': getattr(comp, 'domain', 'continuous'),
            })

        # Node() inter-component equations
        dae.equations.extend(self.equations)

        # Merge sensor mappings
        dae.sensor_mappings.update(self.sensor_mappings)
        for sensor in self.sensors + self.components:
            if hasattr(sensor, 'sensor_mapping'):
                dae.sensor_mappings[sensor.name] = {
                    'type': sensor.__class__.__name__,
                    **sensor.sensor_mapping,
                }
            if sensor in self.sensors:
                dae.components.append({
                    'name':   sensor.name,
                    'domain': getattr(sensor, 'domain', 'continuous'),
                })

        return dae

    def get_param_vector(self, overrides: dict = None) -> 'np.ndarray':
        """Build ordered parameter array from defaults, with optional overrides."""
        import numpy as np
        overrides = overrides or {}
        values = []
        for comp in self.components:
            for param_name, meta in comp._param_meta.items():
                key = f'{comp.name}.{param_name}'
                values.append(overrides.get(key, meta['default']))
        return np.array(values)

    def param_index(self, component_name: str, param_name: str) -> int:
        """Get the flat-vector index of a parameter (for debugging)."""
        idx = 0
        for comp in self.components:
            for pname in comp._param_meta:
                if comp.name == component_name and pname == param_name:
                    return idx
                idx += 1
        raise KeyError(f'{component_name}.{param_name} not found')
