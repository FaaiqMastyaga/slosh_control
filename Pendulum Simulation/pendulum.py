import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Simulation Parameters (Using values from your input) ---

g = 9810.0   # Gravity (e.g., mm/s^2)
L = 150.0    # Length of the pendulum (e.g., mm)
c = 100     # Damping coefficient (b in your original code)

# Time parameters
T_max = 5.0  # Total simulation time (s) - reduced duration due to high frequency
num_points = 500  # Number of simulation steps
t = np.linspace(0, T_max, num_points)

# Derived parameters for ODE
# For a simple pendulum of mass m, the damping term is often written as (c/m) * d(theta)/dt.
# Since mass (m) is not defined, we assume 'c' represents a generalized friction factor
# such that the damping torque is T_d = -c * d(theta)/dt.
# The ODE becomes: d^2(theta)/dt^2 = -(g/L) * sin(theta) - (c/L) * d(theta)/dt (assuming mass cancels or is factored)

# Initial conditions (Impulse given at equilibrium)
initial_angle = 0.0      # Start at equilibrium (0 radians)
initial_impulse = 5.0    # Initial angular velocity (rad/s)
Y0 = [initial_angle, initial_impulse]

# --- 2. Define the Damped Pendulum Differential Equation (ODE) ---

def damped_pendulum_ode(Y, t, g, L, c):
    """
    The system of first-order ODEs for a fixed damped pendulum.
    Y[0] = theta (angle)
    Y[1] = d_theta/dt (angular velocity)
    """
    theta = Y[0]
    dydt = [
        Y[1],  # d(theta)/dt = angular velocity
        # d^2(theta)/dt^2 = -(g/L) * sin(theta) - (c/L) * d(theta)/dt
        -(g / L) * np.sin(theta) - (c / L) * Y[1]
    ]
    return dydt

# --- 3. Solve the ODE ---

Y_solution = odeint(damped_pendulum_ode, Y0, t, args=(g, L, c))
theta_t = Y_solution[:, 0] # Extract the angular position

# Convert polar coordinates (theta, L) to Cartesian (x, y) for visualization
# Note: The pivot is fixed at (0, 0)
x_t = L * np.sin(theta_t)
y_t = -L * np.cos(theta_t)

# --- 4. Combined Plotting and Visualization Setup ---

# Create a single figure with two subplots (1 row, 2 columns)
fig, (ax_anim, ax_plot) = plt.subplots(2, 1, figsize=(7, 7))
fig.suptitle(f'Damped Pendulum Simulation', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

# Setup the Animation Visualization (Left Subplot)
# Set limits based on the pendulum length (L)
L_margin = L * 0.1
ax_anim.set_xlim(-L - L_margin, L + L_margin)
ax_anim.set_ylim(-L - L_margin, L_margin) # Pendulum hangs down
ax_anim.set_aspect('equal')
ax_anim.axis('off') # Hide axes for a cleaner visualization
# ax_anim.set_title('Pendulum Motion', fontsize=14)

# Draw the fixed pivot point
ax_anim.plot(0, 0, 'o', color='black', markersize=10, zorder=3)

# Create the line (rod) and the bob (mass) that will be updated
rod, = ax_anim.plot([], [], lw=3, color='#ef4444', zorder=1)
bob, = ax_anim.plot([], [], 'o', color='#10b981', markersize=20, zorder=2)
time_text = ax_anim.text(0.02, 0.9, '', transform=ax_anim.transAxes, fontsize=12)


# Setup the Position Plot (Right Subplot)
ax_plot.set_xlim(0, T_max)
ax_plot.set_ylim(np.min(theta_t) * 1.1, np.max(theta_t) * 1.1)
ax_plot.set_xlabel('Time (s)', fontsize=12)
ax_plot.set_ylabel('Angular Position $\\theta$ (radians)', fontsize=12)
ax_plot.grid(True, linestyle='--', alpha=0.7)
# ax_plot.set_title('Angular Position Trace', fontsize=14)

# Create the plot line that will grow frame-by-frame
plot_trace, = ax_plot.plot([], [], label=r'$\theta(t)$', color='#3b82f6', lw=2)
# Add a marker to show the current point on the trace
current_time_marker, = ax_plot.plot([], [], 'o', color='#059669', markersize=6, label='Current Time')


# --- 5. Animation Functions ---

# Initialization function for the animation
def init():
    rod.set_data([], [])
    bob.set_data([], [])
    plot_trace.set_data([], [])
    current_time_marker.set_data([], [])
    time_text.set_text('')
    # Returns all artists that will be updated
    return rod, bob, plot_trace, current_time_marker, time_text

# Animation update function
def update(frame):
    # Get current position and time
    x_current = x_t[frame]
    y_current = y_t[frame]
    t_current = t[frame]
    theta_current = theta_t[frame]

    # Update the animation (Left Subplot)
    rod.set_data([0, x_current], [0, y_current])
    bob.set_data([x_current], [y_current]) # Fixed: Must pass sequence
    # time_text.set_text(f'Time: {t_current:.2f} s\nAngle: {np.degrees(theta_current):.1f}°')

    # Update the position trace (Right Subplot)
    plot_trace.set_data(t[:frame+1], theta_t[:frame+1])
    current_time_marker.set_data([t_current], [theta_current]) # Fixed: Must pass sequence

    # Return all updated objects for blit=True
    return rod, bob, plot_trace, current_time_marker, time_text

# Create the animation object
# Interval is calculated to match the total time / number of frames
ani = FuncAnimation(
    fig,
    update,
    frames=len(t),
    init_func=init,
    blit=True,
    interval=T_max * 1000 / num_points # Interval in milliseconds
)

# Display the combined figure
plt.show()
