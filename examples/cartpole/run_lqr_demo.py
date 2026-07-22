import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.cartpole.cartpole_lqr import compute_lqr_gain, simulate_cartpole_lqr
from examples.cartpole.animate import animate_cartpole

def main():
    print("========================================")
    # 1. Compute LQR control gain
    K = compute_lqr_gain()
    
    # 2. Simulate closed loop system starting with cartpole tilted 0.25 rad (~14.3 degrees)
    # State: [x, theta, x_dot, theta_dot]
    x0 = [0.0, 0.25, 0.0, 0.0]
    t_max = 5.0
    
    sol, state_names = simulate_cartpole_lqr(K, t_max=t_max, x0=x0)
    
    # Extract trajectories
    # state_names order: ['q_cartpole_x', 'q_cartpole_theta', 'u_cartpole_u1', 'u_cartpole_u2']
    # which maps to indices 0, 1, 2, 3 in sol.y
    x_idx = state_names.index("q_cartpole_x")
    theta_idx = state_names.index("q_cartpole_theta")
    
    x_traj = sol.y[x_idx]
    theta_traj = sol.y[theta_idx]
    
    # 3. Create and save animation
    output_gif = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cartpole_lqr.gif")
    animate_cartpole(sol.t, x_traj, theta_traj, l=0.5, save_path=output_gif)
    print("========================================")
    print("LQR Demo Completed Successfully!")

if __name__ == "__main__":
    main()
