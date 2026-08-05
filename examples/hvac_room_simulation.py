"""
examples/hvac_room_simulation.py
================================
Example simulation of a dynamic thermal zone (room) with external ambient wall loss
and HVAC heating heat source, modeled using Braid lumped parameter components.
"""

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt

# Ensure braid module root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base import Component, System, Node
from components.hvac_thermal import (
    ThermalZone, ThermalResistance, TemperatureSource, HeatFlowSource, TemperatureSensor
)
from simulation import simulate_system

def run_hvac_example():
    print("Initializing HVAC Lumped Parameter Thermal Zone model...")

    # 1. Instantiate components
    # Zone: 50 m3 volume, starting initial temperature ~ 15 °C (288.15 K)
    zone = ThermalZone('room', V_zone=50.0, Q_gain=100.0) # 100W occupant gain
    
    # Wall thermal resistance R = 0.05 K/W
    wall = ThermalResistance('wall', R=0.05)
    
    # Ambient Outdoor Temperature (10 °C = 283.15 K)
    ambient = TemperatureSource('ambient', T_val=283.15)
    
    # HVAC Heater source: 1500 W heating power
    heater = HeatFlowSource('heater', Q_val=1500.0)
    
    # Temperature sensor for indoor air
    sensor = TemperatureSensor('T_room_sensor')

    # Aggregate system
    system = System([zone, wall, ambient, heater, sensor])

    # 2. Build topological Node connections
    # Connect room air node to wall interior side, heater, and sensor
    Node(system, [(zone, 'port'), (wall, 'a'), (heater, 'p'), (sensor, 'p')])
    
    # Connect wall exterior side to ambient temperature source
    Node(system, [(wall, 'b'), (ambient, 'p')])

    # 3. Setup initial state values & simulate
    # Room initial temperature state T_zone_room = 288.15 K (15°C)
    y0 = [288.15]
    t_span = (0.0, 7200.0)  # 2 hours simulation time

    print("Simulating thermal dynamic response over 2 hours...")
    dae = system.to_dae()
    sol = simulate_system(dae, t_span, y0, params=None, backend='numpy')

    t = sol['t']
    T_zone_k = sol['y'][:, 0]
    T_zone_c = T_zone_k - 273.15


    print(f"Simulation completed successfully.")
    print(f"Initial Room Temp: {T_zone_c[0]:.2f} °C")
    print(f"Final Room Temp after 2 hours: {T_zone_c[-1]:.2f} °C")

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(t / 60.0, T_zone_c, label='Indoor Air Temp (°C)', color='firebrick', linewidth=2)
    plt.axhline(y=10.0, color='blue', linestyle='--', label='Outdoor Ambient Temp (10 °C)')
    plt.title('HVAC Dynamic Room Heating - Lumped Parameter Model')
    plt.xlabel('Time [minutes]')
    plt.ylabel('Temperature [°C]')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    out_img = os.path.join(os.path.dirname(__file__), 'hvac_simulation_result.png')
    plt.savefig(out_img, dpi=150)
    print(f"Saved simulation plot to {out_img}")

if __name__ == '__main__':
    run_hvac_example()
