import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import torch
import casadi as ca
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from base import Component, System, Node
from index_reduction import pantelides_pass, tearing_pass
from simulation import simulate_system
from components.rigid_body import RigidBodyURDF
from components.neural_net import NeuralNetworkPyTorch
from components.electrical_basic import Resistor, ElectricalGround
from examples.renewable.aerodynamics import generate_aerodynamic_data, train_surrogate_model
from examples.renewable.generator import Generator

class WindComponent(Component):
    def __init__(self, name="wind_source"):
        super().__init__(name)
        self.v = ca.SX.sym("v_wind")
        self.register_param("v_wind", self.v, 11.0) # default wind speed = 11 m/s

def main():
    print("==================================================")
    print("   Braid Multiphysics Renewable Energy Demo       ")
    print("==================================================")
    
    # 1. Generate CFD aerodynamic data and train PyTorch policy surrogate
    X, Y = generate_aerodynamic_data(num_samples=2000)
    mlp = train_surrogate_model(X, Y, epochs=300)
    
    # 2. Build Braid Multiphysics components
    print("Building components...")
    
    # Aerodynamic NN component
    nn_aero = NeuralNetworkPyTorch(
        name="aero_cfd",
        pytorch_model=mlp,
        input_names=["v_wind", "omega"],
        output_names=["tau_aero"]
    )
    
    # Wind speed source
    wind = WindComponent("wind")
    
    # Wind turbine mechanical structure (URDF)
    urdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scratch", "wind_turbine.urdf")
    turbine = RigidBodyURDF(
        name="turbine",
        urdf_path=urdf_path,
        root_link="base_link",
        tip_link="blade2",
        gravity=[0.0, 0.0, -9.81],
        external_force_links=[]
    )
    
    # Electromechanical Generator
    generator = Generator("gen", kt=2.5, ke=2.5)
    
    # Grid load (15 Ohm resistor) and ground
    load = Resistor("load", R=15.0)
    ground = ElectricalGround("ground")
    
    # Assemble all components into a System
    system = System([nn_aero, wind, turbine, generator, load, ground])
    
    # 3. Apply Connection Equations
    
    # A. Wind speed to NN input
    system.equations.append(nn_aero.ports["v_wind"][1] - wind.v)
    
    # B. Rotor speed from turbine to NN input
    system.equations.append(nn_aero.ports["omega"][1] - turbine.ports["rotor_joint"][2])
    
    # C. Mechanical coupling: shaft speed equality
    system.equations.append(generator.ports["shaft"][2] - turbine.ports["rotor_joint"][2])
    
    # D. Torque balance: net rotor joint torque = aero torque - generator load torque
    system.equations.append(turbine.ports["rotor_joint"][0] - (nn_aero.ports["tau_aero"][0] * 1e6 - generator.ports["shaft"][0]))
    
    # E. Lock the turbine's yaw joint (no yawing motion)
    system.equations.append(turbine.ports["yaw_joint"][0] - 0.0)
    
    # F. Electrical acausal connections
    Node(system, [(generator, "p"), (load, "p")])
    Node(system, [(generator, "n"), (load, "n"), (ground, "p")])
    
    # 4. Compile system
    print("Compiling multiphysics DAE system...")
    dae = system.to_dae()
    red = pantelides_pass(dae)
    torn = tearing_pass(red)
    
    print("States:", torn.state_names)
    
    # 5. Simulate system starting from standstill
    # States order: [q_yaw, v_yaw, q_rotor, v_rotor, v_c...] depending on active states.
    # The turbine states are:
    #   q_turbine_yaw_joint, v_turbine_yaw_joint, q_turbine_rotor_joint, v_turbine_rotor_joint.
    # Total states: 4.
    y0 = [0.0, 0.0, 0.0, 0.0]
    t_span = (0.0, 20.0)
    
    print(f"Simulating spin-up for {t_span} seconds...")
    sol = simulate_system(torn, t_span, y0, params=None, backend='numpy', method='RK45')
    
    # 6. Analyze and plot results
    print("Plotting simulation performance...")
    
    # Extract states
    yaw_idx = torn.state_names.index("q_turbine_yaw_joint")
    rotor_idx = torn.state_names.index("q_turbine_rotor_joint")
    rotor_speed_idx = torn.state_names.index("v_turbine_rotor_joint")
    
    t = sol.t
    rotor_speed = sol.y[rotor_speed_idx]
    
    # Evaluate algebraic variables (torque, voltage, current) along the trajectory
    # We can reconstruct them using CasADi Function from torn.alg_assignments
    tau_aero_expr = torn.alg_assignments["out_aero_cfd_tau_aero"]
    v_load_expr = torn.alg_assignments["v_p_load"] # Resistor positive terminal voltage
    i_load_expr = torn.alg_assignments["i_p_load"] # Resistor current
    
    # Function input variables order
    # state_vars: q_turbine_yaw_joint, v_turbine_yaw_joint, q_turbine_rotor_joint, v_turbine_rotor_joint
    # parameter: wind.v, gen.kt, gen.ke, load.R
    state_syms = torn.x_vars
    param_syms = torn.p_vars
    
    eval_func = ca.Function("eval_func", [ca.vertcat(*state_syms), ca.vertcat(*param_syms)], [tau_aero_expr * 1e6, v_load_expr, i_load_expr])
    
    p_defaults = torn.get_param_defaults()
    
    tau_aero_vals = []
    v_load_vals = []
    i_load_vals = []
    
    for i in range(len(t)):
        x_val = sol.y[:, i]
        out = eval_func(x_val, p_defaults)
        tau_aero_vals.append(float(out[0]))
        v_load_vals.append(float(out[1]))
        i_load_vals.append(float(out[2]))
        
    tau_aero_vals = np.array(tau_aero_vals)
    v_load_vals = np.array(v_load_vals)
    i_load_vals = np.array(i_load_vals)
    
    # Electrical Power = V * I
    p_elec = v_load_vals * (-i_load_vals)  # i_load flows into Resistor p, meaning current leaving generator p is -i_load
    
    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Rotor Speed
    axs[0].plot(t, rotor_speed, 'b-', lw=2)
    axs[0].set_title("Turbine Rotor Speed (Spin-Up)")
    axs[0].set_ylabel("Speed (rad/s)")
    axs[0].grid(True)
    
    # Plot 2: Torques
    # Generator torque is kt * I_generator = kt * (-I_load)
    kt_val = p_defaults[torn.param_names.index("kt_gen")]
    tau_gen_vals = kt_val * (-i_load_vals)
    axs[1].plot(t, tau_aero_vals, 'r-', label="Aerodynamic Torque (NN)")
    axs[1].plot(t, tau_gen_vals, 'g--', label="Generator Opposing Torque")
    axs[1].set_title("Torque Balance on Shaft")
    axs[1].set_ylabel("Torque (Nm)")
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot 3: Power generated
    axs[2].plot(t, p_elec / 1000.0, 'k-', lw=2)
    axs[2].set_title("Generated Electrical Power to Grid Load")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Power (kW)")
    axs[2].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multiphysics_performance.png")
    plt.savefig(plot_path)
    print(f"Performance plot saved to: {plot_path}")
    
    print("==================================================")
    print("Multiphysics Renewable Energy Demo Completed!")
    print("==================================================")

if __name__ == "__main__":
    main()
