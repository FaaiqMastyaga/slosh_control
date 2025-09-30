import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyjet import *
import random
from VelocityProfile import ZVProfile, ModifiedZVProfile, TrapezoidalProfile

# --- Sloshing Simulation Parameters (in mm) ---
ANIM_FPS = 60
TIME_INTERVAL = 1.0 / ANIM_FPS
RESOLUTION = (100, 100) # Grid resolution remains fixed
CONTAINER_WIDTH = 95.0 # mm
CONTAINER_HEIGHT = 35.0 # mm
FILL_HEIGHT = CONTAINER_HEIGHT / 2 # Half-filled
PARTICLE_SPACING = 0.1 # mm
g_mm = 9810.0 # Gravity in mm/s^2

# --- Profile Shaper Parameters ---
L = 1.2 # Effective pendulum length, proportional to container height
b = 0.02
omega_n = np.sqrt(g_mm / L)
zeta = b / (2 * omega_n) if omega_n > 0 else 0
zeta = 0.0
T = 0.5 # Natural period, user-defined
Td = T / np.sqrt(1 - zeta**2) if (1 - zeta**2) > 0 else T # Period of the damped pendulum
td = Td / 2
K = np.exp(-zeta * omega_n * td)

# --- User-Defined Goal (in mm) ---
distance = 500.0 # 5 meters in mm
desired_velocity = 300.0 # 2 m/s in mm/s
profile = 'modified_zero_vibration'  # Options: 'trapezoidal', 'zero_vibration', 'modified_zero_vibration'
coeff_t_acc = 0.5 # Acceleration time coefficient for ZV profile
dt = TIME_INTERVAL  # Link time step to animation frame rate

# --- State and Data Storage ---
cart_ax, cart_ay = 0.0, 0.0
cart_vx, cart_vy = 0.0, 0.0
cart_pos_x, cart_pos_y = 0.0, 0.0
amplitude_data = [] # List to store amplitude at each frame

def main():
    Logging.mute()

    # --- Sloshing Solver Setup ---
    # The domain is now in mm, matching the container dimensions
    solver = ApicSolver2(resolution=RESOLUTION, domainSizeX=CONTAINER_WIDTH)
    box = Box2(lowerCorner=(0.0, 0.0), upperCorner=(CONTAINER_WIDTH, FILL_HEIGHT))
    emitter = VolumeParticleEmitter2(implicitSurface=box, isOneShot=True, spacing=PARTICLE_SPACING)
    solver.particleEmitter = emitter
    box_collider = Box2(lowerCorner=(0.0, 0.0), upperCorner=(CONTAINER_WIDTH, CONTAINER_HEIGHT))
    box_collider.isNormalFlipped = True
    collider = RigidBodyCollider2(surface=box_collider)
    solver.collider = collider

    # --- Generate Motion Profile using the new class structure ---
    if profile == 'trapezoidal':
        profile_generator = TrapezoidalProfile(max_velocity=desired_velocity, max_acceleration=2000.0)
    elif profile == 'zero_vibration':
        profile_generator = ZVProfile(max_velocity=desired_velocity, max_acceleration=2000.0, T_d=Td, K=K, coeff_t_acc=coeff_t_acc)
    elif profile == 'modified_zero_vibration':
        profile_generator = ModifiedZVProfile(max_velocity=desired_velocity, max_acceleration=2000.0, T_d=Td, K=K)
    else:
        raise ValueError("Invalid profile selected.")

    motion_data = profile_generator.generate(distance=distance, dt=dt)
    
    # --- Calculate acceleration from velocity data ---
    accel_profile = [0.0]
    for i in range(1, len(motion_data)):
        accel = (motion_data[i]['velocity'] - motion_data[i-1]['velocity']) / (motion_data[i]['time'] - motion_data[i-1]['time'])
        accel_profile.append(accel)

    # --- Find the total travel time before settling ---
    travel_time = motion_data[-1]['time']
    
    # --- Extend the simulation time to observe settling ---
    settling_time = 10.0 # seconds
    num_settling_frames = int(settling_time / dt)
    accel_profile.extend([0.0] * num_settling_frames)
    
    global total_time
    total_time = len(accel_profile) * dt
    ANIM_NUM_FRAMES = len(accel_profile)

    # --- Matplotlib Plot Setup ---
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
    ax_anim = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_vel = fig.add_subplot(gs[2, 0])
    ax_amplitude = fig.add_subplot(gs[3, 0]) # Renamed for clarity

    # --- Main Animation Plot (Sloshing + Cart) ---
    ax_anim.set_aspect('equal', 'box')
    ax_anim.set_xlim(-100, distance + CONTAINER_WIDTH + 50)
    ax_anim.set_ylim(-10, CONTAINER_HEIGHT + 30)
    ax_anim.set_xlabel('Position (mm)'); ax_anim.set_ylabel('Height (mm)')
    ax_anim.set_title(f"Sloshing Simulation with {profile.replace('_', ' ').title()} Profile")
    
    # Add ground line
    ax_anim.axhline(0, color='black', lw=1)
    
    # Create cart and container patches
    cart_height = 10.0 # mm
    cart_width = CONTAINER_WIDTH + 10.0 # mm
    cart_rect = plt.Rectangle((0, 0), cart_width, cart_height, fc='purple', ec='black')
    
    container_bottom = cart_height # Container sits on top of the cart
    container_rect = plt.Rectangle((0, container_bottom), CONTAINER_WIDTH, CONTAINER_HEIGHT, fc='none', ec='blue', lw=2)

    ax_anim.add_patch(cart_rect)
    ax_anim.add_patch(container_rect)

    # Initialize fluid particle scatter plot
    positions = np.array(solver.particleSystemData.positions, copy=False)
    initial_offset_x = 0.5 * cart_width - 0.5 * CONTAINER_WIDTH
    initial_offset = np.array([initial_offset_x, container_bottom])
    sc = ax_anim.scatter(positions[:, 0] + initial_offset[0], positions[:, 1] + initial_offset[1], s=1, c='blue')
    
    timer_text = ax_anim.text(0.02, 0.85, '', transform=ax_anim.transAxes)
    force_arrow = ax_anim.arrow(0.0, 0.0, 0.0, 0.0, head_width=5, head_length=10, fc='red', ec='red')

    # --- Status graphs ---
    time_data, acceleration_data, velocity_data, amplitude_data = [], [], [], []
    acceleration_line, = ax_acc.plot([], [], lw=2, color='r')
    velocity_line, = ax_vel.plot([], [], lw=2, color='c')
    amplitude_line, = ax_amplitude.plot([], [], lw=2, color='g')

    ax_acc.set_title("Cart Acceleration")
    ax_acc.set_ylabel("Accel (mm/s²)")
    ax_acc.set_xlabel("Time (s)")
    ax_acc.set_xlim(0, total_time)
    ax_acc.grid(True)

    ax_vel.set_title("Cart Velocity")
    ax_vel.set_ylabel("Vel (mm/s)")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_xlim(0, total_time); ax_vel.set_ylim(0, desired_velocity * 1.2)
    ax_vel.grid(True)
    
    ax_amplitude.set_title("Sloshing Amplitude (from Surface Level)")
    ax_amplitude.set_xlabel("Time (s)"); ax_amplitude.set_ylabel("Amplitude (mm)")
    ax_amplitude.set_xlim(0, total_time)
    ax_amplitude.set_ylim(0, FILL_HEIGHT + 5)
    ax_amplitude.grid(True)
    
    fig.tight_layout()

    def update_frame(frame_index):
        global cart_vx, cart_vy, cart_pos_x, cart_pos_y
        
        current_time = frame_index * dt
        
        if frame_index < len(accel_profile):
            accel = accel_profile[frame_index]
        else:
            accel = 0.0

        cart_vx += accel * dt
        cart_pos_x += cart_vx * dt

        solver.gravity = Vector2D(accel, -g_mm)
        current_frame = Frame(frame_index, dt)
        solver.update(current_frame)
        
        positions = np.array(solver.particleSystemData.positions, copy=False)
        
        # --- Update Visualizations ---
        # Translate the fluid particles to the cart's new position
        translated_positions = positions + np.array([cart_pos_x, container_bottom])
        sc.set_offsets(translated_positions)
        
        # Update the cart and container positions
        cart_rect.set_x(cart_pos_x - 0.5 * (cart_width - CONTAINER_WIDTH))
        container_rect.set_x(cart_pos_x)

        # Calculate the sloshing amplitude
        if len(positions) > 0:
            max_y = positions[:, 1].max()
            amplitude = max_y - FILL_HEIGHT
        else:
            amplitude = 0.0

        timer_text.set_text(f"Time: {current_time:.2f}s | Frame: {frame_index}")

        if abs(accel) > 0.01:
            force_arrow.set_visible(True)
            force_arrow.set_data(x=cart_pos_x + CONTAINER_WIDTH/2, y=container_bottom + CONTAINER_HEIGHT/2, dx=accel * 0.05, dy=0)
        else:
            force_arrow.set_visible(False)
        
        # --- Update graphs ---
        time_data.append(current_time)
        acceleration_data.append(accel)
        velocity_data.append(np.sqrt(cart_vx**2 + cart_vy**2))
        amplitude_data.append(amplitude)
        
        acceleration_line.set_data(time_data, acceleration_data)
        velocity_line.set_data(time_data, velocity_data)
        amplitude_line.set_data(time_data, amplitude_data)

        if len(acceleration_data) > 0:
            max_abs_accel = max(1, max(abs(a) for a in acceleration_data))
            ax_acc.set_ylim(-max_abs_accel * 1.2, max_abs_accel * 1.2)
        
        fig.canvas.draw_idle()
        
        return sc, timer_text, force_arrow, cart_rect, container_rect, acceleration_line, velocity_line, amplitude_line

    ani = animation.FuncAnimation(fig, update_frame, frames=ANIM_NUM_FRAMES, interval=1000 * dt, blit=False, repeat=False)
    plt.show()

    # --- Print the results after the animation window is closed ---
    max_amplitude = max(amplitude_data) if amplitude_data else 0.0
    travel_time = motion_data[-1]['time']
    print(f"Total time to travel {distance} mm: {travel_time:.2f} s")
    print(f"Maximum sloshing amplitude: {max_amplitude:.2f} mm")

if __name__ == '__main__':
    main()
