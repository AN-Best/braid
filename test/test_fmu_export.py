import os
import sys
import sympy as sp
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import System, Node
from components.linear_mechanical_1D import Mass, Spring, Damper, Ground
from index_reduction import pantelides_pass, tearing_pass, simplification_pass
from fmu_exporter import export_fmu
from fmpy import read_model_description, simulate_fmu

def test_fmi3_model_exchange_export():
    print("\n--- Testing FMI 3.0 Model Exchange Export ---")
    
    # 1. Assemble Mass-Spring-Damper system
    mass = Mass('mass', m=2.0)
    spring = Spring('spring', k=10.0)
    damper = Damper('damper', c=1.0)
    ground = Ground('ground')
    
    system = System([mass, spring, damper, ground])
    Node(system, [(mass, 'p'), (spring, 'p2'), (damper, 'p2')])
    Node(system, [(ground, 'p'), (spring, 'p1'), (damper, 'p1')])
    
    dae = system.to_dae()
    red = pantelides_pass(dae)
    torn = tearing_pass(red)
    simp = simplification_pass(torn)
    
    # 2. Export to FMI 3.0 FMU
    test_dir = os.path.dirname(os.path.abspath(__file__))
    fmu_path = os.path.join(test_dir, "mass_spring_damper_test.fmu")
    
    # Clean old file if exists
    if os.path.exists(fmu_path):
        os.remove(fmu_path)
        
    print(f"Exporting test model to: {fmu_path} ...")
    export_fmu(simp, fmu_path, model_id="mass_spring_damper_test")
    
    assert os.path.exists(fmu_path), "FMU file was not created!"
    
    try:
        # 3. Read and verify model description metadata
        print("Verifying FMU model description XML...")
        model_desc = read_model_description(fmu_path)
        assert model_desc.modelName == "mass_spring_damper_test"
        assert model_desc.fmiVersion == "3.0"
        
        # Verify that expected states and parameters are in the variables list
        var_names = [v.name for v in model_desc.modelVariables]
        assert "x_mass" in var_names
        assert "m_mass" in var_names
        assert "k_spring" in var_names
        assert "c_damper" in var_names
        assert "der(x_mass)" in var_names
        
        # 4. Run simulation using fmpy to verify equation solver integration
        print("Simulating test FMU using fmpy...")
        start_values = {'x_mass': 2.0}
        results = simulate_fmu(
            fmu_path,
            start_time=0.0,
            stop_time=10.0,
            step_size=0.01,
            start_values=start_values,
            output=['x_mass', 'der(x_mass)']
        )
        
        # Verify displacement decays over time
        x_mass_final = results['x_mass'][-1]
        print(f"Initial x_mass = 2.0, Final x_mass (t=10) = {x_mass_final:.6f}")
        assert abs(x_mass_final) < 0.25, f"Expected decayed displacement, got {x_mass_final}"
        
        # Verify velocity decays over time
        der_x_mass_final = results['der(x_mass)'][-1]
        print(f"Final velocity (t=10) = {der_x_mass_final:.6f}")
        assert abs(der_x_mass_final) < 0.15, f"Expected decayed velocity, got {der_x_mass_final}"
        
        print("FMI 3.0 Model Exchange export and simulation test passed successfully!")
        
    finally:
        # Clean up test file
        if os.path.exists(fmu_path):
            try:
                # Force garbage collection to release library files on Windows
                import gc
                gc.collect()
                os.remove(fmu_path)
            except Exception:
                pass

if __name__ == "__main__":
    test_fmi3_model_exchange_export()
