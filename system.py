import casadi as ca 

class System(object):
    def __init__(self,els):
        self.els = els
        self.elsd=dict([(e.name,e) for e in els])
        self.connection_alg_eqs = []
        self.alg_eqs = []
        self.ode_eqs = {}
        self.states = []
        self.params = []

        for e in self.els:
            self.states += e.states
            self.params += e.params
            if hasattr(e,'alg_eqs'):
                self.alg_eqs += e.alg_eqs
            if hasattr(e,'ode_eqs'):
                self.ode_eqs.update(e.ode_eqs)

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
