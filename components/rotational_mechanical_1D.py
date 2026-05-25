import sympy as sp
from base import Component

t = sp.Symbol('t')

class RotationalIntertia(Component):
    def __init__(self, name, m=1.0):
        super().__init__(name)

        I_sym = sp.Symbol(f'I_{self.name}')
        self.register_param('m', I_sym, default=m)

        theta_sym = sp.Function(f'theta_{self.name}')(t)
        T_sym = sp.Function(f'T_{self.name}')(t)

        self.states.extend([theta_sym])

        # Native second-order constraint
        self.equations = [I_sym * sp.Derivative(theta_sym, t, 2) - T_sym]
        
        self.ports = {'p': [T_sym, theta_sym, sp.Derivative(theta_sym, t)]} 


class RotationalSpring(Component):
    def __init__(self, name, k=1.0):
        super().__init__(name)

        k_sym = sp.Symbol(f'k_{self.name}')
        self.register_param('k', k_sym, default=k)

        theta1_sym = sp.Function(f'theta1_{self.name}')(t)
        theta2_sym = sp.Function(f'theta2_{self.name}')(t)

        T1_sym = sp.Function(f'T1_{self.name}')(t)
        T2_sym = sp.Function(f'T2_{self.name}')(t)

        self.equations = [-k_sym * (theta1_sym - theta2_sym) - T2_sym,
                        -T2_sym - T1_sym]
        
        self.ports = {'p1': [T1_sym, theta1_sym, sp.Derivative(theta1_sym, t)], 
                      'p2': [T2_sym, theta2_sym, sp.Derivative(theta2_sym, t)]}
        

class RotationalDamper(Component):
    def __init__(self, name, c=0.1):
        super().__init__(name)

        c_sym = sp.Symbol(f'c_{self.name}')
        self.register_param('c', c_sym, default=c)

        theta1_sym = sp.Function(f'theta1_{self.name}')(t)
        theta2_sym = sp.Function(f'theta2_{self.name}')(t)

        T1_sym = sp.Function(f'T1_{self.name}')(t)
        T2_sym = sp.Function(f'T2_{self.name}')(t)

        self.equations = [-c_sym * (sp.Derivative(theta1_sym, t) - sp.Derivative(theta2_sym, t)) - T2_sym,
                        -T2_sym - T1_sym]
        
        self.ports = {'p1': [T1_sym, theta1_sym, sp.Derivative(theta1_sym, t)], 
                      'p2': [T2_sym, theta2_sym, sp.Derivative(theta2_sym, t)]}
        

class RotationalGround(Component):
    def __init__(self, name):
        super().__init__(name)

        theta_sym = sp.Function(f'theta_{self.name}')(t)
        T_sym = sp.Function(f'T_{self.name}')(t)

        self.equations = [theta_sym]

        self.ports = {'p': [T_sym, theta_sym, sp.Derivative(theta_sym, t)]} 


class Torque(Component):
    def __init__(self, name, Text=0.0):
        super().__init__(name)

        T_sym = sp.Function(f'T_{self.name}')(t)
        theta_sym = sp.Function(f'theta_{self.name}')(t)

        Text_sym = sp.Symbol(f'Text_{self.name}')
        self.register_param('T', Text_sym, default=Text)

        self.equations = [T_sym - Text_sym]

        self.ports = {'p': [T_sym, theta_sym, sp.Derivative(theta_sym, t)]}