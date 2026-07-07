"""
demo/simulate_turbine.py
========================
Loads the wind turbine URDF, compiles the DAE system, simulates it under gravity / external torques,
and visualizes the results using Meshcat with a correct visual tree.
"""

import os
import sys
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.rigid_body import RigidBodyURDF
from base import System
from simulation import simulate_system

def main():
    urdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "wind_turbine.urdf")
    
    print("Loading URDF and building Braid component...")
    # Turbine with 2 actuated joints: yaw_joint, rotor_joint
    # We add external force ports on the 'nacelle' and 'rotor' links to simulate wind loading or external contact
    turbine = RigidBodyURDF(
        name="turbine",
        urdf_path=urdf_path,
        root_link="base_link",
        tip_link="blade2",
        gravity=[0.0, 0.0, -9.81],
        external_force_links=["nacelle", "rotor"]
    )
    
    system = System([turbine])
    
    # Expose and apply torques:
    # 1. Control torques (acting directly on joint axes)
    system.equations.append(turbine.tau_syms[0] - 100.0) # Constant yawing torque
    system.equations.append(turbine.tau_syms[1] - 1000.0) # Constant driving rotor torque
    
    # 2. External spatial forces (acting on specific links). Let's set them to zero for now.
    # Note: Braid ports map to self.ports = {port_name: [effort, across, dacross]}
    # We retrieve the 3D effort vectors and add their scalar components to system.equations.
    f_nacelle = turbine.ports["f_ext_nacelle"][0]
    f_rotor = turbine.ports["f_ext_rotor"][0]
    for i in range(3):
        system.equations.append(f_nacelle[i])
        system.equations.append(f_rotor[i])
        
    print("Compiling system to DAE...")
    dae = system.to_dae()
    print("States:", dae.state_names)
    
    # Start from complete standstill: all angles and velocities = 0.0
    y0 = [0.0, 0.0, 0.0, 0.0]
    
    t_span = (0.0, 10.0)
    print(f"Simulating system for t in {t_span}...")
    sol = simulate_system(dae, t_span, y0, params=None, backend='pytorch', method='dopri5')
    
    if not sol.success:
        print("Simulation failed!")
        return
        
    print("Simulation succeeded! Number of time steps:", len(sol.t))
    
    # Setup Meshcat visualizer
    print("Starting Meshcat server and visualizer...")
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    
    vis = meshcat.Visualizer()
    print("--------------------------------------------------")
    print(f"Meshcat visualizer URL: {vis.url()}")
    print("--------------------------------------------------")
    
    # Clear visualizer state
    vis.delete()
    
    # Define kinematic tree using nested paths:
    # vis["yaw_node"] -> Yaw rotation joint at [0, 0, 0]
    #   vis["yaw_node/tower"] -> Visual cylinder for tower
    #   vis["yaw_node/nacelle"] -> Sits at [0, 0, 10] relative to yaw_node
    #     vis["yaw_node/nacelle/rotor_node"] -> Spinning joint at [1.5, 0, 0] relative to nacelle
    #       vis["yaw_node/nacelle/rotor_node/hub"] -> Visual cylinder for rotor
    #       vis["yaw_node/nacelle/rotor_node/blade1"] -> Visual box at [0, 0, 0.5]
    #       vis["yaw_node/nacelle/rotor_node/blade2"] -> Visual box at [0, 0, -0.5]
    
    # Tower geometry: Cylinder defaults to Y-alignment. We rotate it to Z-alignment.
    # Meshcat Cylinder arguments: Cylinder(height, radius)
    tower_geom = g.Cylinder(height=10.0, radius=0.4)
    vis["yaw_node/tower"].set_object(tower_geom, g.MeshLambertMaterial(color=0xcccccc))
    # Place visual cylinder so it extends from Z=0 to Z=10. Center is at Z=5.
    # Cylinder initially along Y, so we rotate it -90 deg around X-axis.
    vis["yaw_node/tower"].set_transform(
        tf.translation_matrix([0, 0, 5]).dot(tf.rotation_matrix(np.pi/2, [1, 0, 0]))
    )
    
    # Nacelle geometry
    vis["yaw_node/nacelle"].set_object(g.Box([2.0, 1.0, 1.0]), g.MeshLambertMaterial(color=0x888888))
    # Place Nacelle at local offset Z=10.0 and shift center forward along X
    vis["yaw_node/nacelle"].set_transform(tf.translation_matrix([0.5, 0, 10.0]))
    
    # Rotor Hub geometry: rotates around local X-axis. Visual cylinder rotates to X-axis alignment.
    hub_geom = g.Cylinder(height=0.6, radius=0.5)
    # Child of rotor_node to spin with it. Local placement of the joint is X=1.5, Z=0.0 relative to nacelle.
    # But wait, nacelle visual center was shifted to X=0.5. The joint origin in URDF is relative to the nacelle link frame!
    # So relative to nacelle frame, rotor joint is at [1.5, 0, 0].
    # Let's set the static offset of the rotor joint relative to the nacelle frame:
    # Note: Since vis["yaw_node/nacelle"] has its own transform, we should create the rotor_node as a child of yaw_node
    # to avoid double-offsetting the nacelle's visual center.
    # Structure:
    # vis["yaw_node/nacelle_visual"] (the box shifted by [0.5, 0, 10.0])
    # vis["yaw_node/rotor_node"] (the joint at [1.5, 0, 10.0])
    vis["yaw_node/nacelle_visual"].set_object(g.Box([2.0, 1.0, 1.0]), g.MeshLambertMaterial(color=0x888888))
    vis["yaw_node/nacelle_visual"].set_transform(tf.translation_matrix([0.5, 0, 10.0]))
    
    # Joint node for rotor (spins relative to yaw_node)
    vis["yaw_node/rotor_node"].set_transform(tf.translation_matrix([1.5, 0, 10.0]))
    
    # Visuals inside rotor_node
    vis["yaw_node/rotor_node/hub"].set_object(hub_geom, g.MeshLambertMaterial(color=0x444444))
    vis["yaw_node/rotor_node/hub"].set_transform(tf.rotation_matrix(np.pi/2, [0, 0, 1])) # Rotate cylinder to X-axis
    
    # Blades
    vis["yaw_node/rotor_node/blade1"].set_object(g.Box([0.1, 0.3, 5.0]), g.MeshLambertMaterial(color=0xeeeeee))
    vis["yaw_node/rotor_node/blade1"].set_transform(tf.translation_matrix([0, 0, 3.0])) # offset + half length
    
    vis["yaw_node/rotor_node/blade2"].set_object(g.Box([0.1, 0.3, 5.0]), g.MeshLambertMaterial(color=0xeeeeee))
    vis["yaw_node/rotor_node/blade2"].set_transform(tf.translation_matrix([0, 0, -3.0]).dot(tf.rotation_matrix(np.pi, [1, 0, 0])))
    
    # Animate
    print("Animating trajectory in Meshcat...")
    fps = 30.0
    t_anim = np.arange(t_span[0], t_span[1], 1.0 / fps)
    
    yaw_positions = np.interp(t_anim, sol.t, sol.y[0])
    rotor_positions = np.interp(t_anim, sol.t, sol.y[2])
    
    try:
        for loop in range(3):
            print(f"Playing loop {loop+1}/3...")
            start_time = time.time()
            for i, t in enumerate(t_anim):
                yaw = yaw_positions[i]
                rotor = rotor_positions[i]
                
                # Apply joint rotations directly to the respective transformation nodes!
                # 1. Yaw joint: Rotates yaw_node around vertical Z axis
                vis["yaw_node"].set_transform(tf.rotation_matrix(yaw, [0, 0, 1]))
                
                # 2. Rotor joint: Spins rotor_node around horizontal local X axis
                vis["yaw_node/rotor_node"].set_transform(
                    tf.translation_matrix([1.5, 0, 10.0]).dot(tf.rotation_matrix(rotor, [1, 0, 0]))
                )
                
                # Maintain real-time playback
                elapsed = time.time() - start_time
                target_elapsed = (i + 1) / fps
                if elapsed < target_elapsed:
                    time.sleep(target_elapsed - elapsed)
        print("Animation completed. Keeping Meshcat server alive... Press Ctrl+C in terminal to stop.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Animation server stopped by user.")

if __name__ == "__main__":
    main()
