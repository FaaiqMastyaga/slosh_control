import numpy as np
import matplotlib.pyplot as plt
from VelocityProfile import TrapezoidalProfile, ZVProfile, ModifiedZVProfile

# --- Physical System Parameters (Needed only for Td and K calculation) ---
g = 9810                                     # Gravity (m/s^2)
L = 150                                      # Length of the pendulum (m)
b = 1                                        # Damping coefficient
omega_n = np.sqrt(g / L)                     # Natural frequency
zeta = b / (2 * omega_n) if omega_n > 0 else 0 # Damping ratio
T = 2 * np.pi * np.sqrt(L / g) if g > 0 and L > 0 else 0 # Period of the pendulum (s)
Td = 2 * np.pi / (omega_n * np.sqrt(1 - zeta**2)) if (1 - zeta**2) > 0 else T # Period of the damped pendulum
td = Td / 2                                  # Time for the second pulse ZV shaper (Half period)
K = np.exp(-zeta * omega_n * td)

# --- User-Defined Goal ---
distance = 500       # Total distance the cart should travel
desired_velocity = 300     # Desired cruising velocity
desired_accel = 600   # Desired acceleration
dt = 0.01             # Time step for simulation
post_motion_delay_s = 3  # Time to simulate after motion profile ends
coeff_t_acc = 0.5

# Select the profile you want to simulate
# profile_name = 'zero_vibration'
# profile_name = 'trapezoidal'
profile_name = 'modified_zero_vibration'

print(f"Natural Period T: {T:.2f} s")
print(f"Damped Period Td: {Td:.2f} s")

class SystemState:
    """Stores the current state of the cart and pendulum."""
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
    
    # 1. Simulate the motion profile
    for i, frame in enumerate(motion_profile):
        try:
            time = frame['time']
            cart_vx_profile = frame['velocity']
            cart_ax = frame['acceleration']
        except KeyError as e:
            print(f"Error: Missing key in motion profile data: {e}")
            return []

        # Update the cart's state
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

    # 2. Add a delay after the motion is complete
    if not simulation_data:
        return []

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
        # Extract data arrays for full plotting
        times = np.array([d['time'] for d in simulation_data])
        velocities = np.array([d['cart_vel'] for d in simulation_data])
        accelerations = np.array([d['cart_accel'] for d in simulation_data])
        
        total_duration = times[-1]
        
        # Find max values for plot limits
        max_accel = max(abs(d['cart_accel']) for d in simulation_data) * 1.2 if simulation_data else 1
        max_vel = max(abs(d['cart_vel']) for d in simulation_data) * 1.2 if simulation_data else 1
        max_time = 3.5
        
        # --- Matplotlib setup for Combined Velocity and Acceleration Graph ---
        
        fig, ax_vel = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"{profile_name.replace('_', ' ').title()} Motion Profile Input", fontsize=16)

        # 1. Plot Velocity on the primary axis (left)
        color_vel = 'teal'
        ax_vel.set_xlabel("Time (s)")
        ax_vel.set_ylabel("Velocity ($m/s$)", color=color_vel)
        ax_vel.plot(times, velocities, lw=2.5, color=color_vel, label='Velocity')
        ax_vel.tick_params(axis='y', labelcolor=color_vel)
        ax_vel.set_xlim(0, max_time)
        ax_vel.set_ylim(-0.1, max_vel)
        ax_vel.grid(True, linestyle='--', alpha=0.6)
        
        # 2. Create a secondary axis for Acceleration (right)
        ax_acc = ax_vel.twinx()
        color_acc = 'firebrick'
        ax_acc.set_ylabel("Acceleration ($m/s^2$)", color=color_acc)
        ax_acc.plot(times, accelerations, lw=2.5, color=color_acc, label='Acceleration')
        ax_acc.tick_params(axis='y', labelcolor=color_acc)
        ax_acc.set_ylim(-max_accel, max_accel)
        
        # Add legend for both lines
        lines, labels = ax_vel.get_legend_handles_labels()
        lines2, labels2 = ax_acc.get_legend_handles_labels()
        ax_acc.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
        print("Simulation complete.")
        final_pos_x = simulation_data[-1]['cart_pos_x']
        final_angle = simulation_data[-1]['pendulum_theta']
        print(f"Final Position: (X={final_pos_x:.2f}) m")
        print(f"Final Pendulum Angle: {final_angle:.2f} degrees")
