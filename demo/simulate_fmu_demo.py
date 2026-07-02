import os
import sys
import numpy as np

# Add project root directory to sys.path to allow imports from sibling folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmpy import simulate_fmu, read_model_description

def run_fmu_simulation_demo():
    print("--- Braid FMI 3.0 Model Exchange Simulation Demo ---")
    
    # 1. Path to FMU
    fmu_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mass_spring_damper.fmu")
    if not os.path.exists(fmu_path):
        print(f"Error: FMU not found at: {fmu_path}")
        print("Please run export_fmu_demo.py first to generate the FMU.")
        return
        
    # 2. Read model description to print some info
    print(f"Reading model description from: {fmu_path} ...")
    model_desc = read_model_description(fmu_path)
    print(f"Model Name: {model_desc.modelName}")
    print(f"FMI Version: {model_desc.fmiVersion}")
    print(f"Model Variables:")
    for var in model_desc.modelVariables:
        print(f"  - {var.name} (ValueReference={var.valueReference}, Causality={var.causality})")
        
    # 3. Simulate FMU
    # In FMI 3.0 Model Exchange, simulate_fmu will automatically use a CVODE solver.
    print("Simulating FMU from t=0.0 to t=10.0 using fmpy...")
    
    # Initial states setting: we stretch the mass displacement to 2.0 (like in the simulation tests)
    # The variable for displacement is 'x_mass'. Let's find its valueReference (should be 3 based on dae.states order).
    # We can pass start values via start_values dict
    # Let's find which variable represents the mass displacement x_mass
    start_values = {}
    for var in model_desc.modelVariables:
        if var.name == 'x_mass':
            start_values[var.name] = 2.0
            
    try:
        results = simulate_fmu(
            fmu_path,
            start_time=0.0,
            stop_time=10.0,
            step_size=0.01,
            start_values=start_values,
            output=['x_mass', 'der(x_mass)']
        )
        print("Simulation complete! Results shape:", results.shape)
        
        # Print first 5 rows
        print("First 5 steps:")
        names = results.dtype.names
        print("  " + "  ".join(names))
        for i in range(5):
            row_str = "  ".join([f"{results[i][col]:.4f}" for col in names])
            print(f"  {row_str}")
            
        # 4. Plot results and save to file
        print("Plotting simulation results...")
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        # Find time, x_mass, and derivative of x_mass
        time = results['time']
        
        if 'x_mass' in names:
            plt.plot(time, results['x_mass'], label='x_mass (Displacement)', color='indigo', linewidth=2.5)
        
        # The state derivative variable is der(x_mass)
        der_x_mass_name = 'der(x_mass)'
        if der_x_mass_name in names:
            plt.plot(time, results[der_x_mass_name], label='v_mass (Velocity)', color='orange', linewidth=2)
            
        plt.title("FMI 3.0 Model Exchange Simulation: Mass-Spring-Damper", fontsize=14, fontweight='bold')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("States", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        
        plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_results.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved successfully to: {plot_path}")
        
    except Exception as e:
        print("Error during simulation:", e)

if __name__ == "__main__":
    run_fmu_simulation_demo()
