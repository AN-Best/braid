import numpy as np

class Component(object):
    def __init__(self, name):

        self.name = name
        self.states = []
        self.params = []        
        self._param_meta = {}   
        self.connection_alg_eqs = []
        self.alg_eqs = []
        self.ode_eqs = {}

    def register_param(self, param_name, sym, default):
        self.params.append(sym)
        self._param_meta[param_name] = {'sym': sym, 'default': default}

<<<<<<< HEAD
def _equate_elementwise(system, a, b):
    if isinstance(a, (list, tuple)):
        for x, y in zip(a, b):
            _equate_elementwise(system, x, y)
    elif hasattr(a, 'shape') and hasattr(a, '__getitem__'): # Matrix or ndarray
        for idx in range(len(a)):
            system.equations.append(a[idx] - b[idx])
    else:
        system.equations.append(a - b)

def _add_elementwise(a, b):
    if isinstance(a, (list, tuple)):
        return [_add_elementwise(x, y) for x, y in zip(a, b)]
    return a + b

def _register_flow_equations(system, eq):
    if isinstance(eq, (list, tuple)):
        for x in eq:
            _register_flow_equations(system, x)
    elif hasattr(eq, 'shape') and hasattr(eq, '__getitem__'):
        for idx in range(len(eq)):
            system.equations.append(eq[idx])
    else:
        system.equations.append(eq)

def Node(system, comp_set):
=======
def Node(system,comp_set):

>>>>>>> parent of a0f3fa0 (sympy backend)
    eq = comp_set[0][0].ports[comp_set[0][1]][0]
    x_ref = comp_set[0][0].ports[comp_set[0][1]][1]
    v_ref = comp_set[0][0].ports[comp_set[0][1]][2]

    for comp, port in comp_set[1:]:
        pi = comp.ports[port][0]
        x_i = comp.ports[port][1]
        v_i = comp.ports[port][2]
<<<<<<< HEAD
        eq = _add_elementwise(eq, pi)
        _equate_elementwise(system, x_ref, x_i)
        _equate_elementwise(system, v_ref, v_i)
    
    _register_flow_equations(system, eq)
=======
        eq = eq + pi
        system.connection_alg_eqs.append(x_ref- x_i)
        system.connection_alg_eqs.append(v_ref- v_i)
    node_eq = eq

    system.connection_alg_eqs.append(node_eq)

>>>>>>> parent of a0f3fa0 (sympy backend)
    
class System(object):
    def __init__(self,components):
        self.components = components
        self.elsd=dict([(e.name,e) for e in components])
        self.connection_alg_eqs = []
        self.alg_eqs = []
        self.ode_eqs = {}
        self.states = []
        self.params = []

        for e in self.components:
            self.states += e.states
            self.params += e.params
            if hasattr(e,'alg_eqs'):
                self.alg_eqs += e.alg_eqs
            if hasattr(e,'ode_eqs'):
                self.ode_eqs.update(e.ode_eqs)

    def get_param_vector(self, overrides=None):
        """Build ordered parameter array from defaults, with optional overrides.
        
        overrides: dict like {'mass.m': 2.0, 'spring.k': 10.0}
        """
        overrides = overrides or {}
        values = []
        for component in self.components:
            for param_name, meta in component._param_meta.items():
                key = f'{component.name}.{param_name}'
                values.append(overrides.get(key, meta['default']))
        return np.array(values)

    def param_index(self, component_name, param_name):
        """Get the index of a param in the flat vector, for debugging."""
        idx = 0
        for component in self.components:
            for pname in component._param_meta:
                if component.name == component_name and pname == param_name:
                    return idx
                idx += 1
        raise KeyError(f'{component_name}.{param_name} not found')

