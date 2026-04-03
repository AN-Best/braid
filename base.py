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

def Node(system,comp_set):

    eq = comp_set[0][0].ports[comp_set[0][1]][0]
    x_ref = comp_set[0][0].ports[comp_set[0][1]][1]
    v_ref = comp_set[0][0].ports[comp_set[0][1]][2]

    for comp, port in comp_set[1:]:
        pi = comp.ports[port][0]
        x_i = comp.ports[port][1]
        v_i = comp.ports[port][2]
        eq = eq + pi
        system.connection_alg_eqs.append(x_ref- x_i)
        system.connection_alg_eqs.append(v_ref- v_i)
    node_eq = eq

    system.connection_alg_eqs.append(node_eq)

    
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

