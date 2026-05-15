import sympy as sp
from base import Component

t = sp.Symbol('t')

class Mass(Component):
    def __init__(self, name, m=1.0):
        super().__init__(name)

        m_sym = sp.Symbol(f'm_{self.name}')
        self.register_param('m', m_sym, default=m)

        x_sym = sp.Function(f'x_{self.name}')(t)
        f_sym = sp.Function(f'f_{self.name}')(t)

        self.states.extend([x_sym])

        # Native second-order constraint
        self.equations = [m_sym * sp.Derivative(x_sym, t, 2) - f_sym]
        
        self.ports = {'p': [f_sym, x_sym, sp.Derivative(x_sym, t)]}


class Spring(Component):
    def __init__(self, name, k=1.0):
        super().__init__(name)

        k_sym = sp.Symbol(f'k_{self.name}')
        self.register_param('k', k_sym, default=k)

        x1_sym = sp.Function(f'x1_{self.name}')(t)
        x2_sym = sp.Function(f'x2_{self.name}')(t)

        f1_sym = sp.Function(f'f1_{self.name}')(t)
        f2_sym = sp.Function(f'f2_{self.name}')(t)

        self.equations = [-k_sym * (x1_sym - x2_sym) - f2_sym,
                        -f2_sym - f1_sym]
        
        self.ports = {'p1': [f1_sym, x1_sym, sp.Derivative(x1_sym, t)], 
                      'p2': [f2_sym, x2_sym, sp.Derivative(x2_sym, t)]}
        

class Damper(Component):
    def __init__(self, name, c=0.1):
        super().__init__(name)

        c_sym = sp.Symbol(f'c_{self.name}')
        self.register_param('c', c_sym, default=c)

        x1_sym = sp.Function(f'x1_{self.name}')(t)
        x2_sym = sp.Function(f'x2_{self.name}')(t)

        f1_sym = sp.Function(f'f1_{self.name}')(t)
        f2_sym = sp.Function(f'f2_{self.name}')(t)

        self.equations = [-c_sym * (sp.Derivative(x1_sym, t) - sp.Derivative(x2_sym, t)) - f2_sym,
                        -f2_sym - f1_sym]
        
        self.ports = {'p1': [f1_sym, x1_sym, sp.Derivative(x1_sym, t)], 
                      'p2': [f2_sym, x2_sym, sp.Derivative(x2_sym, t)]}


class Ground(Component):
    def __init__(self, name):
        super().__init__(name)

        x_sym = sp.Function(f'x_{self.name}')(t)
        f_sym = sp.Function(f'f_{self.name}')(t)

        self.equations = [x_sym]

        self.ports = {'p': [f_sym, x_sym, sp.Derivative(x_sym, t)]} 

class Force(Component):
    def __init__(self, name, F=0.0):
        super().__init__(name)

        f_sym = sp.Function(f'f_{self.name}')(t)
        x_sym = sp.Function(f'x_{self.name}')(t)

        F_sym = sp.Symbol(f'F_{self.name}')
        self.register_param('F', F_sym, default=F)

        self.equations = [f_sym - F_sym]

        self.ports = {'p': [f_sym, x_sym, sp.Derivative(x_sym, t)]}
