import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from VelocityProfile import TrapezoidalProfile, ZVProfile, ModifiedZVProfile

# --- Physical System Parameters ---
g = 9810                                            # Gravity ($m/s^2$)
L = 150                                               # Length of the pendulum (m)
b = 1                                             # Damping coefficient
omega_n = np.sqrt(g / L)                            # Natural frequency
zeta = b / (2 * omega_n) if omega_n > 0 else 0      # Damping ratio
T = 2 * np.pi * np.sqrt(L / g) if g > 0 and L > 0 else 0 # Period of the pendulum (s)
Td = 2 * np.pi / (omega_n * np.sqrt(1 - zeta**2)) if (1 - zeta**2) > 0 else T # Period of the damped pendulum
td = Td / 2                                         # Time for the second pulse ZV shaper (Half period)
K = np.exp(-zeta * omega_n * td)

# --- User-Defined Goal ---
distance = 500       # Total distance the cart should travel
desired_velocity = 300    # Desired cruising velocity
desired_accel = 600   # Desired acceleration
dt = 0.01           # Time step for simulation
post_motion_delay_s = 3  # Time to simulate after motion profile ends
coeff_t_acc = 0.5

# Select the profile you want to simulate
# profile_name = 'trapezoidal'
# profile_name = 'zero_vibration'
profile_name = 'modified_zero_vibration'

print(f"Natural Period T: {T:.2f} s")
print(f"Damped Period Td: {Td:.2f} s")

class SystemState:
    def __init__(self):
        self.cart_pos_x = 0.0
        self.cart_vx = 0.0
        self.cart_ax = 0.0
        self.theta, self.theta_dot, self.theta_ddot = 0.0, 0.0, 0.0
        self.time = 0.0

    def get_pendulum_pos(self):
        """Calculates the 2D position of the pendulum bob."""
        pendulum_x = self.cart_pos_x + L * np.sin(self.theta)
        pendulum_y = -L * np.cos(self.theta)
        return pendulum_x, pendulum_y

def run_simulation(state, motion_profile, dt, post_motion_delay_s):
    """
    Runs the simulation based on a pre-generated motion profile, then adds a delay.
    """
    simulation_data = []
    
    # Simulate the motion profile
    for i, frame in enumerate(motion_profile):
        try:
            time = frame['time']
            cart_vx_profile = frame['velocity']
            cart_ax = frame['acceleration']
        except KeyError as e:
            print(f"Error: Missing key in motion profile data: {e}")
            return []

        # Update the cart's state within the state object.
        state.cart_vx = cart_vx_profile
        state.cart_ax = cart_ax
        state.cart_pos_x += state.cart_vx * dt
        
        # Update pendulum dynamics based on the cart's acceleration
        state.theta_ddot = -(g / L) * np.sin(state.theta) - b * state.theta_dot - (state.cart_ax / L) * np.cos(state.theta)
        state.theta_dot += state.theta_ddot * dt
        state.theta += state.theta_dot * dt
        
        px, py = state.get_pendulum_pos()
        
        # Store data for plotting
        simulation_data.append({
            'time': time, 
            'cart_accel': state.cart_ax, 
            'cart_vel': state.cart_vx, 
            'cart_pos_x': state.cart_pos_x,
            'pendulum_theta': np.degrees(state.theta),
            'pendulum_x': px, 'pendulum_y': py
        })

    # Add a delay after the motion is complete
    current_time = simulation_data[-1]['time']
    final_pos_x = simulation_data[-1]['cart_pos_x']
    final_vx = 0.0
    final_ax = 0.0

    num_delay_steps = int(post_motion_delay_s / dt)
    for i in range(num_delay_steps):
        current_time += dt
        
        # Cart is stopped
        state.cart_vx = final_vx
        state.cart_ax = final_ax
        state.cart_pos_x = final_pos_x

        # Update pendulum dynamics (only affected by gravity and damping)
        state.theta_ddot = -(g / L) * np.sin(state.theta) - b * state.theta_dot
        state.theta_dot += state.theta_ddot * dt
        state.theta += state.theta_dot * dt
        
        px, py = state.get_pendulum_pos()
        
        simulation_data.append({
            'time': current_time,
            'cart_accel': state.cart_ax,
            'cart_vel': state.cart_vx,
            'cart_pos_x': state.cart_pos_x,
            'pendulum_theta': np.degrees(state.theta),
            'pendulum_x': px, 'pendulum_y': py
        })

    return simulation_data

# --- Animation functions for the 2D plot ---
def init():
    """Initializes the animation elements for the 2D plot."""
    cart.set_data([], [])
    pendulum_rod.set_data([], [])
    pendulum_bob.set_data([], [])
    path.set_data([], [])
    time_text.set_text('')
    acceleration_line.set_data([], [])
    velocity_line.set_data([], [])
    angle_line.set_data([], [])
    return cart, pendulum_rod, pendulum_bob, path, time_text, acceleration_line, velocity_line, angle_line

def animate(i):
    """Updates the animation frame."""
    frame_data = simulation_data[i]
    
    cart_x, cart_y = frame_data['cart_pos_x'], 0
    pendulum_x, pendulum_y = frame_data['pendulum_x'], frame_data['pendulum_y']
    
    # Updated to use a line segment for the cart, creating a rectangular shape.
    cart_width = 40 # Define a width for the cart
    cart.set_data([cart_x - cart_width/2, cart_x + cart_width/2], [cart_y, cart_y])
    
    pendulum_rod.set_data([cart_x, pendulum_x], [cart_y, pendulum_y])
    pendulum_bob.set_data([pendulum_x], [pendulum_y])
    time_text.set_text(f'Time: {frame_data["time"]:.2f}s')

    path_x_data = [d['cart_pos_x'] for d in simulation_data[:i+1]]
    path_y_data = np.zeros_like(path_x_data)
    path.set_data(path_x_data, path_y_data)
    
    time_data.append(frame_data['time'])
    acceleration_data.append(frame_data['cart_accel'])
    velocity_data.append(frame_data['cart_vel'])
    angle_data.append(frame_data['pendulum_theta'])
    
    acceleration_line.set_data(time_data, acceleration_data)
    velocity_line.set_data(time_data, velocity_data)
    angle_line.set_data(time_data, angle_data)
    
    return cart, pendulum_rod, pendulum_bob, path, time_text, acceleration_line, velocity_line, angle_line

if __name__ == "__main__":
    # Select and instantiate the desired velocity profile
    profiles = {
        'trapezoidal': TrapezoidalProfile(max_velocity=desired_velocity, max_acceleration=desired_accel),
        'zero_vibration': ZVProfile(max_velocity=desired_velocity, max_acceleration=desired_accel, T_d=Td, K=K, coeff_t_acc=coeff_t_acc),
        'modified_zero_vibration': ModifiedZVProfile(max_velocity=desired_velocity, max_acceleration=desired_accel, T_d=Td, K=K)
    }
    
    if profile_name not in profiles:
        print(f"Profile '{profile_name}' not found. Using Trapezoidal profile.")
        profile_instance = profiles['trapezoidal']
    else:
        profile_instance = profiles[profile_name]
        
    motion_profile = profile_instance.generate(distance=distance, dt=dt)
    print(f"Generated a {profile_name} profile with {len(motion_profile)} time steps.")
    print(f"Time to travel {distance} m: {motion_profile[-1]['time']:.2f} s")

    state = SystemState()
    simulation_data = run_simulation(state, motion_profile, dt, post_motion_delay_s)
    
    if not simulation_data:
        print("Simulation aborted due to an error.")
    else:
        total_duration = simulation_data[-1]['time']

        # Find max values for plot limits
        max_pos = max(abs(d['cart_pos_x']) for d in simulation_data) * 1.2
        min_pos = min(abs(d['cart_pos_x']) for d in simulation_data) * 1.2
        max_accel = max(abs(d['cart_accel']) for d in simulation_data) * 1.2 if simulation_data else 1
        max_vel = max(abs(d['cart_vel']) for d in simulation_data) * 1.2 if simulation_data else 1
        max_angle = max(abs(d['pendulum_theta']) for d in simulation_data) * 1.2 if simulation_data else 1

        print(f"Maximum Pendulum Angle while moving: {max_angle:.2f} degrees")
        
        # Matplotlib setup for 2D animation plot + 3 status graphs
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(4, 1, height_ratios=[4, 1, 1, 1])
        ax_anim = fig.add_subplot(gs[0, 0])
        ax_acc = fig.add_subplot(gs[1, 0])
        ax_vel = fig.add_subplot(gs[2, 0])
        ax_angle = fig.add_subplot(gs[3, 0])
        fig.suptitle(f"{profile_name.replace('_', ' ').title()} Input Shaper Simulation (2D)", fontsize=16)

        ax_anim.set_xlabel("Position (m)")
        ax_anim.set_ylabel("Height (m)")
        ax_anim.set_xlim(-50, max_pos)
        ax_anim.set_ylim(-L * 1.3, L * 0.5)
        ax_anim.set_aspect('equal', adjustable='box') 
        ax_anim.grid(True)

        # Updated to use a line for the cart, creating a rectangular shape.
        cart, = ax_anim.plot([], [], lw=20, solid_capstyle='butt', c='royalblue', label='Cart')
        pendulum_rod, = ax_anim.plot([], [], lw=2, c='black', label='Pendulum')
        pendulum_bob, = ax_anim.plot([], [], 'o', markersize=8, c='red')
        path, = ax_anim.plot([], [], ':', c='purple', lw=1.5, label='Path')
        time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)
        ax_anim.legend(loc='upper right')

        time_data, acceleration_data, velocity_data, angle_data = [], [], [], []
        acceleration_line, = ax_acc.plot([], [], lw=2, color='r')
        ax_acc.grid(); ax_acc.set_xlim(0, total_duration); ax_acc.set_ylim(-max_accel, max_accel)
        ax_acc.set_ylabel("Acceleration ($m/s^2$)")
        ax_acc.axhline(0, color='gray', linestyle='--')

        velocity_line, = ax_vel.plot([], [], lw=2, color='c')
        ax_vel.grid(); ax_vel.set_xlim(0, total_duration); ax_vel.set_ylim(-0.1, max_vel)
        ax_vel.set_ylabel("Velocity ($m/s$)")
        ax_vel.axhline(0, color='gray', linestyle='--')

        angle_line, = ax_angle.plot([], [], lw=2, color='g')
        ax_angle.grid(); ax_angle.set_xlim(0, total_duration); ax_angle.set_ylim(-max_angle, max_angle)
        ax_angle.set_xlabel("Time (s)"); ax_angle.set_ylabel("Angle ($^\circ$)")
        ax_angle.axhline(0, color='gray', linestyle='--')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        ani = animation.FuncAnimation(fig, animate, frames=len(simulation_data), interval=dt*1000, blit=False, init_func=init, repeat=False)
        plt.show()
        
        print("Simulation complete.")
        final_pos_x = simulation_data[-1]['cart_pos_x']
        final_angle = simulation_data[-1]['pendulum_theta']
        print(f"Final Position: (X={final_pos_x:.2f}) m")
        print(f"Final Pendulum Angle: {final_angle:.2f} degrees")
