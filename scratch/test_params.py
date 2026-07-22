import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import casadi as ca

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import Component, System, Node
from index_reduction import pantelides_pass, tearing_pass
from simulation import simulate_system
from components.rigid_body import RigidBodyURDF
from components.neural_net import NeuralNetworkPyTorch
from components.electrical_basic import Resistor, ElectricalGround
from examples.renewable.aerodynamics import generate_aerodynamic_data, train_surrogate_model
from examples.renewable.generator import Generator

X, Y = generate_aerodynamic_data(num_samples=500)
mlp = train_surrogate_model(X, Y, epochs=10)

class WindComponent(Component):
    def __init__(self, name="wind_source"):
        super().__init__(name)
        self.v = ca.SX.sym("v_wind")
        self.register_param("v_wind", self.v, 11.0)

nn_aero = NeuralNetworkPyTorch(
    name="aero_cfd",
    pytorch_model=mlp,
    input_names=["v_wind", "omega"],
    output_names=["tau_aero"]
)
wind = WindComponent("wind")
urdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "wind_turbine.urdf")
turbine = RigidBodyURDF(
    name="turbine",
    urdf_path=urdf_path,
    root_link="base_link",
    tip_link="blade2",
    gravity=[0.0, 0.0, -9.81],
    external_force_links=[]
)
generator = Generator("gen", kt=2.5, ke=2.5)
load = Resistor("load", R=15.0)
ground = ElectricalGround("ground")

system = System([nn_aero, wind, turbine, generator, load, ground])

system.equations.append(nn_aero.ports["v_wind"][1] - wind.v)
system.equations.append(nn_aero.ports["omega"][1] - turbine.ports["rotor_joint"][2])
system.equations.append(generator.ports["shaft"][2] - turbine.ports["rotor_joint"][2])
system.equations.append(turbine.ports["rotor_joint"][0] - (nn_aero.ports["tau_aero"][0] * 1e6 - generator.ports["shaft"][0]))
system.equations.append(turbine.ports["yaw_joint"][0] - 0.0)

Node(system, [(generator, "p"), (load, "p")])
Node(system, [(generator, "n"), (load, "n"), (ground, "p")])

dae = system.to_dae()
red = pantelides_pass(dae)
torn = tearing_pass(red)

p_vals = np.array(torn.get_param_defaults())

print("Param Names in torn:", torn.param_names)
print("Default Param Values:", p_vals)

print("1. Running with params=None...")
sol = simulate_system(torn, (0.0, 2.0), [0.0, 0.0, 0.0, 0.0], params=None, backend='numpy', method='RK45')
print("Succeeded with params=None!")

print("2. Running with params=p_vals...")
sol = simulate_system(torn, (0.0, 2.0), [0.0, 0.0, 0.0, 0.0], params=p_vals, backend='numpy', method='RK45')
print("Succeeded with params=p_vals!")
