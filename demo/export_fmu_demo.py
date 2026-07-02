import os
import sys

# Add project root directory to sys.path to allow imports from sibling folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import System, Node
from components.linear_mechanical_1D import Mass, Spring, Damper, Ground
from index_reduction import pantelides_pass, tearing_pass, simplification_pass
from fmu_exporter import export_fmu

def run_fmu_export_demo():
    print("--- Braid FMI 3.0 Model Exchange Export Demo ---")
    
    # 1. Define physical components
    print("Assembling Mass-Spring-Damper system...")
    mass = Mass('mass', m=2.0)
    spring = Spring('spring', k=10.0)
    damper = Damper('damper', c=1.0)
    ground = Ground('ground')
    
    # 2. Assemble system and connections
    system = System([mass, spring, damper, ground])
    Node(system, [(mass, 'p'), (spring, 'p2'), (damper, 'p2')])
    Node(system, [(ground, 'p'), (spring, 'p1'), (damper, 'p1')])
    
    # 3. Convert to DAE and run compiler passes
    dae = system.to_dae()
    print("Running Pantelides pass...")
    red = pantelides_pass(dae)
    print("Running Tearing pass...")
    torn = tearing_pass(red)
    print("Running Simplification pass...")
    simp = simplification_pass(torn)
    
    # 4. Export to FMI 3.0 Model Exchange FMU
    fmu_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mass_spring_damper.fmu")
    print(f"Exporting model to FMI 3.0 FMU: {fmu_path} ...")
    
    # We run this inside conda run to ensure compiler paths are resolved
    export_fmu(simp, fmu_path, model_id="mass_spring_damper")
    print(f"Success! FMU generated successfully at: {fmu_path}")

if __name__ == "__main__":
    run_fmu_export_demo()
