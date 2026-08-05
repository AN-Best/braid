"""
examples/electro_hydraulic_actuator.py
======================================
Co-simulation example of a DC Motor driving a Hydraulic Pump, pressurizing an
accumulator volume and pushing fluid through a control valve pipe.

Multi-physics domains integrated:
- Electrical
- 1D Rotational Mechanical
- Hydraulic Fluid Dynamics
"""

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base import System, Node
from components.electrical_basic import VoltageSource, ElectricalGround
from components.power_electronics import DCMotor
from components.rotational_mechanical_1D import RotationalInertia
from components.fluid_dynamics import HydraulicPump, FluidCapacitance, PipeResistance, PressureSource

def run_electro_hydraulic_example():
    print("Setting up Electro-Hydraulic Pump System...")

    # 1. Instantiate multi-physics components
    v_supply = VoltageSource('v_supply', V=24.0)
    gnd = ElectricalGround('gnd')
    
    # DC Motor
    motor = DCMotor('motor', R=0.5, L=0.005, Kt=0.15, Ke=0.15)
    
    # Pump shaft inertia
    shaft = RotationalInertia('shaft', I=0.001)

    
    # Hydraulic Pump driven by shaft (displacement flow = 1e-5 m³/rad)
    pump = HydraulicPump('pump', Q_flow=2e-4)
    
    # Hydraulic Accumulator (Fluid Capacitance)
    accumulator = FluidCapacitance('acc', C_f=1e-8)
    
    # Hydraulic Pipe Resistance & Tank Reservoir Pressure
    pipe = PipeResistance('pipe', R_f=5e6)
    tank = PressureSource('tank', P_val=101325.0)

    # Aggregate into System
    system = System([v_supply, gnd, motor, shaft, pump, accumulator, pipe, tank])

    # 2. Wire topological connection nodes across physical domains
    # Electrical domain
    Node(system, [(v_supply, 'p'), (motor, 'el_p')])
    Node(system, [(v_supply, 'n'), (motor, 'el_n'), (gnd, 'p')])
    
    # Rotational Mechanical domain (Motor flange connected to shaft inertia)
    Node(system, [(motor, 'flange'), (shaft, 'p')])


    # Hydraulic Fluid domain (Pump output feeds accumulator and pipe)
    Node(system, [(pump, 'a'), (accumulator, 'p'), (pipe, 'a')])
    Node(system, [(pipe, 'b'), (tank, 'a')])

    # 3. Assemble and inspect DAE state variables
    dae = system.to_dae()
    print("Multi-physics DAE compiled successfully.")
    print(f"System State variables count: {dae.n_states}")


if __name__ == '__main__':
    run_electro_hydraulic_example()
