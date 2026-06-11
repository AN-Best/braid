class Component(object):
    def __init__(self, name):
        self.name = name
        self.states = []
        self.params = []        
        self._param_meta = {}   
        self.equations = []
        self.domain = 'continuous'

    def register_param(self, param_name, sym, default):
        self.params.append(sym)
        self._param_meta[param_name] = {'sym': sym, 'default': default}

def Node(system, comp_set):
    import sympy as sp
    # Separate sensor components and regular components
    sensors = []
    regular = []
    for comp, port in comp_set:
        if comp.__class__.__name__ in ('PositionSensor', 'VelocitySensor'):
            sensors.append((comp, port))
        else:
            regular.append((comp, port))
            
    if not regular:
        return

    # Connect regular components
    eq = regular[0][0].ports[regular[0][1]][0]
    x_ref = regular[0][0].ports[regular[0][1]][1]
    v_ref = regular[0][0].ports[regular[0][1]][2]

    for comp, port in regular[1:]:
        pi = comp.ports[port][0]
        x_i = comp.ports[port][1]
        v_i = comp.ports[port][2]
        eq = eq + pi
        system.equations.append(x_ref - x_i)
        system.equations.append(v_ref - v_i)
    
    node_eq = eq
    system.equations.append(node_eq)

    # Populate sensor mappings
    if not hasattr(system, 'sensor_mappings'):
        system.sensor_mappings = {}
        
    for sensor, port in sensors:
        system.sensor_mappings[sensor.name] = {
            'type': sensor.__class__.__name__,
            'x_ref': x_ref,
            'v_ref': v_ref
        }
    
class System(object):
    def __init__(self, components):
        self.sensors = []
        regular_components = []
        for c in components:
            if c.__class__.__name__ in ('PositionSensor', 'VelocitySensor'):
                self.sensors.append(c)
            else:
                regular_components.append(c)
                
        self.components = regular_components
        self.elsd = dict([(e.name, e) for e in regular_components])
        
        self.states = []
        self.params = []
        self.equations = []
        self.sensor_mappings = {}

        for e in self.components:
            self.states += e.states
            self.params += e.params
            if hasattr(e, 'equations'):
                self.equations += e.equations

    def get_param_vector(self, overrides=None):
        """Build ordered parameter array from defaults, with optional overrides.
        
        overrides: dict like {'mass.m': 2.0, 'spring.k': 10.0}
        """
        import numpy as np
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

    def to_dae(self):
        """Convert the assembled system into a SystemDAE object for index reduction."""
        import sympy as sp
        from sym_dae import SystemDAE
        
        dae = SystemDAE()
        # Ensure we use the exact same symbol instance for time
        t = sp.Symbol('t')
        dae.t = t
        
        dae.states = self.states.copy()
        dae.params = self.params.copy()
        dae.equations = self.equations.copy()
        dae.sensor_mappings = self.sensor_mappings.copy()
        
        # Keep track of component metadata (for domains, parameters, etc.)
        dae.components = [
            {
                'name': comp.name,
                'domain': getattr(comp, 'domain', 'continuous')
            }
            for comp in self.components + self.sensors
        ]
        
        # Populate parameter metadata mapping the string representation of each parameter to its metadata
        dae.param_meta = {}
        for comp in self.components:
            for param_name, meta in getattr(comp, '_param_meta', {}).items():
                sym_str = sp.srepr(meta['sym'])
                dae.param_meta[sym_str] = {
                    'name': param_name,
                    'component': comp.name,
                    'default': meta['default']
                }
            
        return dae
