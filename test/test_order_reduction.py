import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from index_reduction import *
from IR_BASE import *

pos = State("x")
mass = Parameter("m",10.0)
force = Numeric(10.0)


R = Subtract(Multiply(mass,Der(Der(pos))),force)
dae = DAE()
dae.equations.append(R)
dae.states.append(pos)
new_dae = order_reduction_pass(dae)
print(new_dae.equations)
print(new_dae.states)
print(new_dae.derivatives)