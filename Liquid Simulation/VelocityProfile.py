import numpy as np

class VelocityProfile:
  """Abstract base class for motion profile generators."""
  def generate(self, distance: float, max_velocity: float, **kwargs) -> list:
    """
    Plans and generates the trajectory.
    Returns a list of dictionaries: [{'time': t, 'velocity': vel}, ...]
    """
    raise NotImplementedError

  def _run_simulation(self, t_total, phase_times, phase_accels, dt: float):
      """Helper to generate the final time-stamped velocity commands from acceleration phases."""
      vel, i = 0, 0
      motion_commands = []
      for t in np.arange(0, t_total + dt, dt):
          # 1. First, append the state at the current time t.
          #    This ensures the first point is always (t=0, vel=0).
          #    For the final point, clamp velocity to exactly 0 to handle numerical float errors.
          current_velocity = 0.0 if t >= t_total else vel
          motion_commands.append({'time': t, 'velocity': current_velocity})

          # 2. Now, find the acceleration for the current interval
          current_accel = 0
          if i < len(phase_times):
              if t < phase_times[i]:
                  current_accel = phase_accels[i]
              else:
                  i += 1
                  if i < len(phase_times):
                      current_accel = phase_accels[i]
          
          # 3. Finally, update the velocity for the *next* time step.
          vel += current_accel * dt
          
      return motion_commands

  def _normalize_profile(self, motion_commands: list, desired_distance: float, desired_max_velocity: float) -> list:
      """
      Performs a two-step normalization to guarantee peak velocity and total distance.
      """
      if not motion_commands: return []

      # Step 1: Correct the Peak Velocity
      velocities = [p['velocity'] for p in motion_commands]
      actual_peak = max(velocities) if velocities else 0
      if actual_peak > 0:
          vel_scaling = desired_max_velocity / actual_peak
          for cmd in motion_commands:
              cmd['velocity'] *= vel_scaling

      # Step 2: Correct the Total Distance by scaling time
      time_pts = [p['time'] for p in motion_commands]
      vel_pts = [p['velocity'] for p in motion_commands]
      actual_distance = np.trapz(vel_pts, time_pts)
      
      if actual_distance > 0:
          time_scaling = desired_distance / actual_distance
          for cmd in motion_commands:
              cmd['time'] *= time_scaling
      
      return motion_commands

# --- Specific Profile Implementations ---

class ZVProfile(VelocityProfile):
  """
  Generates a Zero-Vibration (ZV) or 'double-kick' velocity profile.
  This logic is based on your original `profile == 1` implementation.
  """
  def __init__(self, max_velocity: float, max_acceleration: float, T_d: float = 0.414, K: float = 0.96, coeff_t_acc: float = 0.5):
    self.max_velocity = max_velocity
    self.max_acceleration = max_acceleration
    self.T_d = T_d
    self.K = K
    self.A1 = 1 / (1 + K)
    self.A2 = K / (1 + K)
    self.coeff_t_acc = coeff_t_acc

  def generate(self, distance: float, dt: float = 0.005, **kwargs) -> list:
    max_velocity = self.max_velocity
    max_acceleration = self.max_acceleration
    t_kick = self.T_d / 2
    t_acc = self.coeff_t_acc * t_kick
    
    # Guard against division by zero if max_velocity is 0
    if max_velocity <= 0: return [{'time': 0, 'velocity': 0}]

    dist_accel = (0.5*t_acc*(max_velocity/2)) + ((t_kick-t_acc)*(max_velocity/2)) + (0.5*(max_velocity/2 + max_velocity)*t_acc)
    minimum_distance = dist_accel * 2

    if distance <= minimum_distance and minimum_distance > 0:
      max_velocity = distance / ((1 + self.coeff_t_acc) * t_kick)
      t_const = 0
    else:
      t_const = (distance - minimum_distance) / max_velocity
    
    # This calibration equation was in your original code
    calibrated_velocity = max_velocity
    accel = (calibrated_velocity / 2) / t_acc if t_acc > 0 else 0
    
    phase_times = [
      t_acc, 
      t_kick, 
      t_kick + t_acc, 
      t_kick + t_acc + t_const,
      t_kick + t_acc + t_const + t_acc,
      t_kick + t_acc + t_const + t_acc + (t_kick - t_acc),
      t_kick + t_acc + t_const + t_acc + t_kick
    ]
    phase_accels = [2*self.A1*accel, 0, 2*self.A2*accel, 0, -2*self.A1*accel, 0, -2*self.A2*accel]
    t_total = phase_times[-1]

    motion_commands = self._run_simulation(t_total, phase_times, phase_accels, dt) 
    return self._normalize_profile(motion_commands, distance, max_velocity)

class ModifiedZVProfile(VelocityProfile):
  """
  Generates a multi-step or "stacked" trapezoidal velocity profile.
  This logic is based on your original `profile == 2` implementation.
  """
  def __init__(self, max_velocity: float, max_acceleration: float, T_d: float = 0.414, K: float = 1.0):
    self.max_velocity = max_velocity
    self.max_acceleration = max_acceleration
    self.T_d = T_d
    self.K = K
    self.A1 = 1 / (1 + K)
    self.A2 = K / (1 + K)

  def generate(self, distance: float, dt: float = 0.005, **kwargs) -> list:
    max_velocity = self.max_velocity
    max_acceleration = self.max_acceleration
    t_kick = self.T_d / 2
    t_acc = t_kick

    if max_velocity <= 0: return [{'time': 0, 'velocity': 0}]
    
    dist_accel = (0.5*t_acc*(max_velocity/4)) + (0.5*(max_velocity/4 + 3*max_velocity/4)*t_kick) + (0.5*(3*max_velocity/4 + max_velocity)*t_acc)
    minimum_distance = dist_accel * 2

    if distance <= minimum_distance and minimum_distance > 0:
      max_velocity = distance / (3 * t_kick)
      t_const = 0
    else:
      t_const = (distance - minimum_distance) / max_velocity
        
    calibrated_velocity = max_velocity
    accel = (calibrated_velocity / 4) / t_acc if t_acc > 0 else 0
    
    phase_times = [
      t_acc, 
      t_acc + t_kick, 
      t_acc + t_kick + t_acc,
      t_acc + t_kick + t_acc + t_const,
      t_acc + t_kick + t_acc + t_const + t_acc,
      t_acc + t_kick + t_acc + t_const + t_acc + t_kick,
      t_acc + t_kick + t_acc + t_const + t_acc + t_kick + t_acc
    ]
    phase_accels = [self.A1*2*accel, (self.A1+self.A2)*2*accel, self.A2*2*accel, 0, -self.A1*2*accel, -(self.A1+self.A2)*2*accel, -self.A2*2*accel]
    t_total = phase_times[-1]

    motion_commands = self._run_simulation(t_total, phase_times, phase_accels, dt)
    return self._normalize_profile(motion_commands, distance, max_velocity)

class TrapezoidalProfile(VelocityProfile):
  """
  Generates a classic, single-step trapezoidal velocity profile.
  This profile has higher jerk but is simpler to compute.
  """
  def __init__(self, max_velocity: float, max_acceleration: float):
    self.max_velocity = max_velocity
    self.max_acceleration = max_acceleration

  def generate(self, distance: float, dt: float = 0.005, **kwargs) -> list:
    max_velocity = self.max_velocity
    max_acceleration = self.max_acceleration

    if distance <= 1e-6 or max_velocity <= 0:
      return [{'time': 0, 'velocity': 0}]

    # Time to reach max velocity
    t_accel = max_velocity / max_acceleration
    # Distance covered during full acceleration
    dist_accel = 0.5 * max_acceleration * t_accel**2

    if distance < 2 * dist_accel:
      # Triangle profile: cannot reach max_velocity
      t_accel = np.sqrt(distance / max_acceleration)
      t_const = 0.0
      achieved_max_velocity = max_acceleration * t_accel
    else:
      # Trapezoid profile: can reach max_velocity
      dist_const = distance - 2 * dist_accel
      t_const = dist_const / max_velocity
      achieved_max_velocity = max_velocity

    t_total = 2 * t_accel + t_const
    
    phase_times = [t_accel, t_accel + t_const, t_total]
    phase_accels = [max_acceleration, 0, -max_acceleration]

    motion_commands = self._run_simulation(t_total, phase_times, phase_accels, dt)
    return self._normalize_profile(motion_commands, distance, achieved_max_velocity)