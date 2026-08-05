"""
test/test_hvac_thermal.py
==========================
Unit tests for HVAC and thermal lumped parameter components in Braid.
"""

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base import Component, System, Node
from components.hvac_thermal import (
    ThermalCapacitance, ThermalResistance, ThermalConvection, ThermalRadiation,
    TemperatureSource, HeatFlowSource, TemperatureSensor,
    MassFlowHeatAdvection, ThermalZone, HeatExchangerNTU
)

class TestHVACThermal(unittest.TestCase):
    def test_thermal_components_instantiation(self):
        """Test instantiation and port structure of thermal components."""
        cap = ThermalCapacitance('cap', C=1000.0)
        res = ThermalResistance('res', R=2.0)
        conv = ThermalConvection('conv', G=10.0)
        rad = ThermalRadiation('rad', Hr=5.67e-8)
        
        self.assertIn('p', cap.ports)
        self.assertIn('a', res.ports)
        self.assertIn('b', res.ports)
        self.assertEqual(len(cap.states), 1)
        self.assertEqual(len(cap.equations), 1)
        self.assertEqual(len(res.equations), 2)
        self.assertEqual(len(conv.equations), 2)
        self.assertEqual(len(rad.equations), 2)

    def test_hvac_zone_and_flow(self):
        """Test HVAC thermal zone, heat exchanger, and flow components."""
        zone = ThermalZone('room', V_zone=50.0, Q_gain=100.0)
        heater = HeatFlowSource('heater', Q_val=1500.0)
        amb = TemperatureSource('amb', T_val=283.15)
        flow = MassFlowHeatAdvection('flow', m_dot=0.5, Cp=1005.0)
        hx = HeatExchangerNTU('hx', epsilon=0.8)
        sensor = TemperatureSensor('sensor')

        system = System([zone, heater, amb, flow, hx, sensor])
        Node(system, [(zone, 'port'), (heater, 'p'), (sensor, 'p')])
        Node(system, [(flow, 'inlet'), (hx, 'fluid_in')])

        dae = system.to_dae()
        self.assertIsNotNone(dae)
        self.assertEqual(len(zone.states), 1)

if __name__ == '__main__':
    unittest.main()

