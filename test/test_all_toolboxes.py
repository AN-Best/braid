"""
test/test_all_toolboxes.py
===========================
Unit tests covering all Braid physical modeling domain toolboxes.
"""

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base import Component, System, Node
from components.fluid_dynamics import FluidCapacitance, PipeResistance, ControlValve, PressureSource
from components.power_electronics import DCMotor, EquivalentCircuitBattery
from components.control_blocks import PIDController, FirstOrderFilter
from components.refrigeration import VaporCompressor, ExpansionValve

class TestAllToolboxes(unittest.TestCase):
    def test_fluid_dynamics_assembly(self):
        acc = FluidCapacitance('acc', C_f=1e-9)
        pipe = PipeResistance('pipe', R_f=1e6)
        res = PressureSource('res', P_val=101325.0)

        system = System([acc, pipe, res])
        Node(system, [(acc, 'p'), (pipe, 'a')])
        Node(system, [(pipe, 'b'), (res, 'a')])

        dae = system.to_dae()
        self.assertIsNotNone(dae)
        self.assertEqual(len(acc.states), 1)

    def test_power_electronics_assembly(self):
        motor = DCMotor('motor', R=1.0, L=0.01)
        batt = EquivalentCircuitBattery('batt', Q_capacity=36000.0)

        system = System([motor, batt])
        Node(system, [(batt, 'p'), (motor, 'el_p')])
        Node(system, [(batt, 'n'), (motor, 'el_n')])

        dae = system.to_dae()
        self.assertIsNotNone(dae)
        self.assertEqual(len(batt.states), 2)
        self.assertEqual(len(motor.states), 1)

    def test_control_blocks(self):
        pid = PIDController('pid', Kp=2.0, Ki=0.5)
        lpf = FirstOrderFilter('lpf', Tau=0.1)

        system = System([pid, lpf])
        dae = system.to_dae()
        self.assertIsNotNone(dae)
        self.assertEqual(len(pid.states), 1)
        self.assertEqual(len(lpf.states), 1)

    def test_refrigeration_assembly(self):
        comp = VaporCompressor('comp', V_disp=1e-5)
        valve = ExpansionValve('valve', Cd=0.6)

        system = System([comp, valve])
        dae = system.to_dae()
        self.assertIsNotNone(dae)
        self.assertEqual(len(comp.states), 1)

if __name__ == '__main__':
    unittest.main()
