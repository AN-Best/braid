import sympy as sp
from base import Component

t = sp.Symbol('t')

class Inertia(Component):
    def __init__(self, name, J=1.0):
        super().__init__(name)

        J_sym = sp.Symbol(f'J_{self.name}')
        self.register_param('J', J_sym, default=J)

        theta_sym = sp.Function(f'theta_{self.name}')(t)
        tau_sym = sp.Function(f'tau_{self.name}')(t)

        self.states.extend([theta_sym])

        # Native second-order constraint: J * d^2(theta)/dt^2 = tau
        self.equations = [J_sym * sp.Derivative(theta_sym, t, 2) - tau_sym]
        
        self.ports = {'p': [tau_sym, theta_sym, sp.Derivative(theta_sym, t)]}


class TorsionalSpring(Component):
    def __init__(self, name, k=1.0):
        super().__init__(name)

        k_sym = sp.Symbol(f'k_{self.name}')
        self.register_param('k', k_sym, default=k)

        theta1_sym = sp.Function(f'theta1_{self.name}')(t)
        theta2_sym = sp.Function(f'theta2_{self.name}')(t)

        tau1_sym = sp.Function(f'tau1_{self.name}')(t)
        tau2_sym = sp.Function(f'tau2_{self.name}')(t)

        self.equations = [-k_sym * (theta1_sym - theta2_sym) - tau2_sym,
                          -tau2_sym - tau1_sym]
        
        self.ports = {'p1': [tau1_sym, theta1_sym, sp.Derivative(theta1_sym, t)], 
                      'p2': [tau2_sym, theta2_sym, sp.Derivative(theta2_sym, t)]}
        

class RotationalDamper(Component):
    def __init__(self, name, c=0.1):
        super().__init__(name)

        c_sym = sp.Symbol(f'c_{self.name}')
        self.register_param('c', c_sym, default=c)

        theta1_sym = sp.Function(f'theta1_{self.name}')(t)
        theta2_sym = sp.Function(f'theta2_{self.name}')(t)

        tau1_sym = sp.Function(f'tau1_{self.name}')(t)
        tau2_sym = sp.Function(f'tau2_{self.name}')(t)

        self.equations = [-c_sym * (sp.Derivative(theta1_sym, t) - sp.Derivative(theta2_sym, t)) - tau2_sym,
                          -tau2_sym - tau1_sym]
        
        self.ports = {'p1': [tau1_sym, theta1_sym, sp.Derivative(theta1_sym, t)], 
                      'p2': [tau2_sym, theta2_sym, sp.Derivative(theta2_sym, t)]}


class Fixed(Component):
    def __init__(self, name):
        super().__init__(name)

        theta_sym = sp.Function(f'theta_{self.name}')(t)
        tau_sym = sp.Function(f'tau_{self.name}')(t)

        self.equations = [theta_sym]

        self.ports = {'p': [tau_sym, theta_sym, sp.Derivative(theta_sym, t)]} 


class Torque(Component):
    def __init__(self, name, tau=0.0):
        super().__init__(name)

        tau_sym = sp.Function(f'tau_{self.name}')(t)
        theta_sym = sp.Function(f'theta_{self.name}')(t)

        tau_val_sym = sp.Symbol(f'tau_val_{self.name}')
        self.register_param('tau', tau_val_sym, default=tau)

        self.equations = [tau_sym - tau_val_sym]

        self.ports = {'p': [tau_sym, theta_sym, sp.Derivative(theta_sym, t)]}
