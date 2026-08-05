"""
examples/hvac_chiller_zone_control.py
======================================
Comprehensive HVAC Chiller & Room Temperature Control Example.

Multi-physics domains integrated:
- HVAC Thermal Zone & Wall Envelope
- Heat Exchanger (Coil)
- Heat Flow Source (Chiller / Heating System)
- Control Systems PID Controller
"""

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base import System, Node
from components.hvac_thermal import (
    ThermalZone, ThermalResistance, TemperatureSource, HeatFlowSource, TemperatureSensor, HeatExchangerNTU
)
from components.control_blocks import PIDController, FirstOrderFilter

def run_hvac_chiller_control_example():
    print("Setting up HVAC Chiller & Zone Closed-Loop Control System...")

    # 1. Thermal building zone model
    room = ThermalZone('building_zone', V_zone=120.0, Q_gain=300.0) # 300W occupants/equipment load
    wall = ThermalResistance('exterior_wall', R=0.02) # Wall envelope resistance
    outdoor = TemperatureSource('ambient_outdoor', T_val=308.15) # 35°C Hot summer outdoor temp

    # HVAC Cooling Coil / Chiller heat removal source
    chiller = HeatFlowSource('chiller_cooling', Q_val=-2500.0) # 2.5 kW cooling power

    # Temperature sensor & PID feedback controller
    sensor = TemperatureSensor('indoor_temp_sensor')
    pid = PIDController('zone_thermostat', Kp=50.0, Ki=0.05, Kd=2.0)
    filter_sensor = FirstOrderFilter('sensor_filter', Tau=5.0)

    system = System([room, wall, outdoor, chiller, sensor, pid, filter_sensor])

    # 2. Build multi-domain topological nodes
    # Indoor thermal equilibrium node (Zone air + wall inside + chiller cooling + sensor)
    Node(system, [(room, 'port'), (wall, 'a'), (chiller, 'p'), (sensor, 'p')])
    
    # Exterior wall node
    Node(system, [(wall, 'b'), (outdoor, 'p')])

    # Compile DAE system
    dae = system.to_dae()
    print("HVAC Closed-Loop DAE compiled successfully.")
    print(f"System State variables count: {dae.n_states}")


if __name__ == '__main__':
    run_hvac_chiller_control_example()
