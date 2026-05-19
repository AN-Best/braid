import casadi as ca 
from base import Component

class Mass(Component):
    def __init__(self,name,m = 1.0):
        super().__init__(name)

        m_sym = ca.SX.sym(f'm_{self.name}')
        self.register_param('m', m_sym, default=m)

        x_sym = ca.SX.sym(f'x_{self.name}')
        xdot_sym = ca.SX.sym(f'xdot_{self.name}')

        self.states.extend([x_sym,xdot_sym])

        f_sym = ca.SX.sym(f'f_{self.name}')
        xdotdot_sym = ca.SX.sym(f'xdotdot_{self.name}')


        self.alg_eqs = [m_sym*xdotdot_sym - f_sym]
        self.ode_eqs = {x_sym: xdot_sym, xdot_sym: xdotdot_sym}
        
        self.ports = {'p': [f_sym, x_sym, xdot_sym]}


class Spring(Component):
    def __init__(self,name,k = 1.0):
        super().__init__(name)

        k_sym = ca.SX.sym(f'k_{self.name}')
        self.register_param('k', k_sym, default=k)


        x1_sym = ca.SX.sym(f'x1_{self.name}')
        x2_sym = ca.SX.sym(f'x2_{self.name}')
        x1dot_sym = ca.SX.sym(f'x1dot_{self.name}')
        x2dot_sym = ca.SX.sym(f'x2dot_{self.name}')

        f1_sym = ca.SX.sym(f'f1_{self.name}')
        f2_sym = ca.SX.sym(f'f2_{self.name}')

        self.alg_eqs = [-k_sym*(x1_sym - x2_sym)-f2_sym,
                    -f2_sym-f1_sym]
        
        self.ports = {'p1': [f1_sym, x1_sym, x1dot_sym], 'p2': [f2_sym, x2_sym, x2dot_sym]}
        

class Damper(Component):
    def __init__(self,name,c = 0.1):
        super().__init__(name)

        c_sym = ca.SX.sym(f'c_{self.name}')
        self.register_param('c', c_sym, default=c)


        x1_sym = ca.SX.sym(f'x1_{self.name}')
        x2_sym = ca.SX.sym(f'x2_{self.name}')
        x1dot_sym = ca.SX.sym(f'x1dot_{self.name}')
        x2dot_sym = ca.SX.sym(f'x2dot_{self.name}')

        f1_sym = ca.SX.sym(f'f1_{self.name}')
        f2_sym = ca.SX.sym(f'f2_{self.name}')

        self.alg_eqs = [-c_sym*(x1dot_sym - x2dot_sym) - f2_sym,
                    -f2_sym -f1_sym]
        
        self.ports = {'p1': [f1_sym, x1_sym, x1dot_sym], 'p2': [f2_sym, x2_sym, x2dot_sym]}


class Ground(Component):
    def __init__(self,name):
        super().__init__(name)

        x_sym = ca.SX.sym(f'x_{self.name}')
        xdot_sym = ca.SX.sym(f'xdot_{self.name}')
        f_sym = ca.SX.sym(f'f_{self.name}')

        self.alg_eqs = [x_sym,
                        xdot_sym]

        self.ports = {'p':[f_sym, x_sym, xdot_sym]} 

class Force(Component):
    def __init__(self, name, F=0.0):
        super().__init__(name)

        f_sym = ca.SX.sym(f'f_{self.name}')
        x_sym = ca.SX.sym(f'x_{self.name}')
        xdot_sym = ca.SX.sym(f'xdot_{self.name}')

        F_sym = ca.SX.sym(f'F_{self.name}')
        self.register_param('F', F_sym, default=F)


        # Force imposes a value on f at its port
        self.alg_eqs = [f_sym - F_sym]

        self.ports = {'p': [f_sym, x_sym, xdot_sym]}





