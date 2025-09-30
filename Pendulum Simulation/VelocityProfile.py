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
        if not phase_accels:
            return [{'time': t, 'velocity': 0.0, 'acceleration': 0.0} for t in np.arange(0, t_total + dt, dt)]

        vel = 0.0
        accel = phase_accels[0] # Start with the initial acceleration
        motion_commands = []
        phase_index = 0
        
        # Iterate through time steps
        for t in np.arange(0, t_total + dt, dt):
            # Check if we should transition to the next acceleration phase
            if phase_index < len(phase_times) and t >= phase_times[phase_index]:
                phase_index += 1
                if phase_index < len(phase_accels):
                    accel = phase_accels[phase_index]
            
            # Append the state at the current time t
            motion_commands.append({
                'time': t, 
                'velocity': vel, 
                'acceleration': accel
            })
            
            # Update the velocity for the next step using Euler integration
            vel += accel * dt
            
        return motion_commands

    def _normalize_profile(self, motion_commands: list, desired_distance: float, desired_max_velocity: float) -> list:
        """Performs a two-step normalization to guarantee peak velocity and total distance."""
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

class ZVProfile(VelocityProfile):
    """Generates a Zero-Vibration (ZV) or 'double-kick' velocity profile."""
    def __init__(self, max_velocity: float, max_acceleration: float, T_d: float, K: float, coeff_t_acc: float = 0.5):
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
        
        if max_velocity <= 0: return [{'time': 0, 'velocity': 0, 'acceleration': 0}]

        dist_accel = (0.5*t_acc*(max_velocity/2)) + ((t_kick-t_acc)*(max_velocity/2)) + (0.5*(max_velocity/2 + max_velocity)*t_acc)
        minimum_distance = dist_accel * 2

        if distance <= minimum_distance and minimum_distance > 0:
            max_velocity = distance / ((1 + self.coeff_t_acc) * t_kick)
            t_const = 0
        else:
            t_const = (distance - minimum_distance) / max_velocity
        
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
    """Generates a multi-step or "stacked" trapezoidal velocity profile."""
    def __init__(self, max_velocity: float, max_acceleration: float, T_d: float, K: float = 1.0):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.T_d = T_d
        self.K = K
        self.A1 = 1 / (1 + K)
        self.A2 = K / (1 + K)

    def generate(self, distance: float, dt: float = 0.005, **kwargs) -> list:
        max_velocity = self.max_velocity
        t_kick = self.T_d / 2
        t_acc = t_kick

        if max_velocity <= 0: return [{'time': 0, 'velocity': 0, 'acceleration': 0}]
        
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
    """Generates a classic, single-step trapezoidal velocity profile."""
    def __init__(self, max_velocity: float, max_acceleration: float):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

    def generate(self, distance: float, dt: float = 0.005, **kwargs) -> list:
        max_velocity = self.max_velocity
        max_acceleration = self.max_acceleration

        if distance <= 1e-6 or max_velocity <= 0:
            return [{'time': 0, 'velocity': 0, 'acceleration': 0}]

        t_accel = max_velocity / max_acceleration
        dist_accel = 0.5 * max_acceleration * t_accel**2

        if distance < 2 * dist_accel:
            t_accel = np.sqrt(distance / max_acceleration)
            t_const = 0.0
            achieved_max_velocity = max_acceleration * t_accel
        else:
            dist_const = distance - 2 * dist_accel
            t_const = dist_const / max_velocity
            achieved_max_velocity = max_velocity

        t_total = 2 * t_accel + t_const
        
        phase_times = [t_accel, t_accel + t_const, t_total]
        phase_accels = [max_acceleration, 0, -max_acceleration]

        motion_commands = self._run_simulation(t_total, phase_times, phase_accels, dt)
        return self._normalize_profile(motion_commands, distance, achieved_max_velocity)
