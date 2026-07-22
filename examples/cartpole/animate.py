import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def animate_cartpole(t, x_trajectory, theta_trajectory, l=0.5, save_path="cartpole_lqr.gif", title="Cartpole Stabilization Demo"):
    """
    Creates an animation of the cartpole motion and saves it as a GIF.
    
    Parameters:
      t: array of time steps
      x_trajectory: array of cart positions
      theta_trajectory: array of pole angles (in radians)
      l: length of the pole
      save_path: filename to save the animation
      title: title of the animation plot
    """
    print(f"Creating animation and saving to {save_path}...")
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_aspect('equal')
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-l - 0.2, l + 0.5)
    ax.grid(True)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title(title)
    
    # Elements to draw
    # Track
    ax.plot([-3.0, 3.0], [-0.1, -0.1], 'k--', lw=1.5)
    
    # Cart (rectangle)
    cart_w = 0.4
    cart_h = 0.2
    cart_rect = plt.Rectangle((0, 0), cart_w, cart_h, fc='blue', ec='k', zorder=2)
    ax.add_patch(cart_rect)
    
    # Pole (line) and bob (circle)
    pole_line, = ax.plot([], [], 'o-', lw=3, color='orange', markerfacecolor='red', markersize=8, zorder=3)
    
    # Time text annotation
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # Interpolate to constant frame rate (e.g. 30 fps) for smooth animation
    fps = 30.0
    t_constant = np.arange(t[0], t[-1], 1.0 / fps)
    x_interp = np.interp(t_constant, t, x_trajectory)
    theta_interp = np.interp(t_constant, t, theta_trajectory)
    
    def init():
        cart_rect.set_xy((-cart_w/2, -cart_h/2))
        pole_line.set_data([], [])
        time_text.set_text('')
        return cart_rect, pole_line, time_text
        
    def update(frame):
        x = x_interp[frame]
        theta = theta_interp[frame]
        
        # Update cart position (bottom-left coordinate)
        cart_rect.set_xy((x - cart_w/2, -cart_h/2))
        
        # Update pole positions
        # Pivot point is (x, 0)
        # Bob is at (x - l*sin(theta), l*cos(theta))
        x_bob = x - l * np.sin(theta)
        y_bob = l * np.cos(theta)
        
        pole_line.set_data([x, x_bob], [0, y_bob])
        
        # Update time
        time_text.set_text(f"Time: {t_constant[frame]:.2f}s\nAngle: {np.degrees(theta):.1f}°")
        
        # Automatically center the plot view around the cart
        ax.set_xlim(x - 1.5, x + 1.5)
        
        return cart_rect, pole_line, time_text
        
    ani = animation.FuncAnimation(
        fig, update, frames=len(t_constant), init_func=init, blit=True, interval=1000/fps
    )
    
    # Save the animation using pillow writer
    ani.save(save_path, writer='pillow', fps=int(fps))
    plt.close(fig)
    print(f"Animation saved successfully to {os.path.abspath(save_path)}")
