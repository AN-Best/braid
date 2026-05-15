import sympy as sp

class SystemDAE:
    def __init__(self):
        self.t = sp.Symbol('t')
        self.states = []       # List of sympy Functions of t
        self.derivatives = {}  # Map of state to its derivative
        self.equations = []    # List of sympy expressions (implicitly = 0)
        self.params = []       # List of sympy Symbols

    def copy_structure(self):
        new_dae = SystemDAE()
        new_dae.t = self.t
        new_dae.states = list(self.states)
        new_dae.derivatives = dict(self.derivatives)
        new_dae.params = list(self.params)
        return new_dae
