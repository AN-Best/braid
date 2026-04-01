import casadi as ca 

class Translation1D(object):
    def __init__(self,name):

        self.name = name
        self.params = [] 
        self.states = [] 
        self.ode_eqs = {}   
        self.alg_eqs = []
        self.ports = {} 

class Mass(Translation1D):
    def __init__(self,name,m = 1.0):
        super().__init__(name)

        m_sym = ca.SX.sym(f'm_{self.name}')
        self.params.append(m_sym)

        x_sym = ca.SX.sym(f'x_{self.name}')
        xdot_sym = ca.SX.sym(f'xdot_{self.name}')

        self.states.extend([x_sym,xdot_sym])

        f_sym = ca.SX.sym(f'f_{self.name}')
        xdotdot_sym = ca.SX.sym(f'xdotdot_{self.name}')


        self.alg_eqs = [m_sym*xdotdot_sym - f_sym]
        self.ode_eqs = {x_sym: xdot_sym, xdot_sym: xdotdot_sym}
        
        self.ports = {'p': [f_sym, x_sym, xdot_sym]}


class Spring(Translation1D):
    def __init__(self,name,k = 1.0):
        super().__init__(name)

        k_sym = ca.SX.sym(f'k_{self.name}')
        self.params.append(k_sym)

        x1_sym = ca.SX.sym(f'x1_{self.name}')
        x2_sym = ca.SX.sym(f'x2_{self.name}')
        x1dot_sym = ca.SX.sym(f'x1dot_{self.name}')
        x2dot_sym = ca.SX.sym(f'x2dot_{self.name}')

        f1_sym = ca.SX.sym(f'f1_{self.name}')
        f2_sym = ca.SX.sym(f'f2_{self.name}')

        self.alg_eqs = [-k_sym*(x1_sym - x2_sym)-f2_sym,
                    -f2_sym-f1_sym]
        
        self.ports = {'p1': [f1_sym, x1_sym, x1dot_sym], 'p2': [f2_sym, x2_sym, x2dot_sym]}
        

class Damper(Translation1D):
    def __init__(self,name,c = 0.1):
        super().__init__(name)

        c_sym = ca.SX.sym(f'c_{self.name}')
        self.params.append(c_sym)

        x1_sym = ca.SX.sym(f'x1_{self.name}')
        x2_sym = ca.SX.sym(f'x2_{self.name}')
        x1dot_sym = ca.SX.sym(f'x1dot_{self.name}')
        x2dot_sym = ca.SX.sym(f'x2dot_{self.name}')

        f1_sym = ca.SX.sym(f'f1_{self.name}')
        f2_sym = ca.SX.sym(f'f2_{self.name}')

        self.alg_eqs = [-c_sym*(x1dot_sym - x2dot_sym) - f2_sym,
                    -f2_sym -f1_sym]
        
        self.ports = {'p1': [f1_sym, x1_sym, x1dot_sym], 'p2': [f2_sym, x2_sym, x2dot_sym]}


class Ground(Translation1D):
    def __init__(self,name):
        super().__init__(name)

        x_sym = ca.SX.sym(f'x_{self.name}')
        xdot_sym = ca.SX.sym(f'xdot_{self.name}')
        f_sym = ca.SX.sym(f'f_{self.name}')

        self.alg_eqs = [x_sym,
                            xdot_sym]

        self.ports = {'p':[f_sym, x_sym, xdot_sym]} 