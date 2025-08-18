"""
Safe PID Control with Tilt Angle Limiting

This module provides a safer version of DSLPIDControl that includes attitude angle limiting
to prevent drone flipping due to excessive tilt angles during aggressive maneuvers.

Key Safety Features:
- Maximum tilt angle limiting (default: 30 degrees)
- Motor output limiting to prevent over-saturation
- Gradual ramping for aggressive corrections
- Enhanced stability during startup
"""

import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

class SafeDSLPIDControl(DSLPIDControl):
    """Safe PID controller with tilt angle limiting.
    
    Extends the standard DSLPIDControl with additional safety features:
    - Maximum tilt angle limiting to prevent flipping
    - Motor output clamping for stability
    - Gradual startup behavior
    """
    
    def __init__(self, 
                 drone_model: DroneModel,
                 g: float = 9.8,
                 max_tilt_angle: float = np.pi/6,  # 30 degrees default
                 max_motor_output_fraction: float = 0.85,  # Limit motor output to 85%
                 startup_duration: float = 2.0  # Gradual startup over 2 seconds
                 ):
        """Initialize the safe PID controller.
        
        Parameters
        ----------
        drone_model : DroneModel
            Type of drone model (CF2X or CF2P)
        g : float
            Gravitational acceleration in m/s²
        max_tilt_angle : float
            Maximum allowed tilt angle in radians (default: 30°)
        max_motor_output_fraction : float
            Maximum motor output as fraction of MAX_PWM (default: 0.85)
        startup_duration : float
            Duration for gradual startup ramping in seconds (default: 2.0)
        """
        super().__init__(drone_model, g)
        
        # Safety parameters
        self.MAX_TILT_ANGLE = max_tilt_angle
        self.MAX_MOTOR_OUTPUT_FRACTION = max_motor_output_fraction
        self.STARTUP_DURATION = startup_duration
        self.SAFE_MAX_PWM = int(self.MAX_PWM * max_motor_output_fraction)
        
        # Startup tracking
        self.startup_timer = 0.0
        self.is_startup_complete = False
        
        print(f"🛡️ SafeDSLPIDControl initialized:")
        print(f"   Max tilt angle: {np.degrees(self.MAX_TILT_ANGLE):.1f}°")
        print(f"   Max motor output: {max_motor_output_fraction*100:.1f}%")
        print(f"   Startup duration: {startup_duration:.1f}s")
    
    def reset(self):
        """Reset controller state including startup timer."""
        super().reset()
        self.startup_timer = 0.0
        self.is_startup_complete = False
        
    def _limitTiltAngles(self, target_thrust, max_tilt=None):
        """Limit the tilt angles by constraining the thrust vector.
        
        Parameters
        ----------
        target_thrust : ndarray
            (3,1) array of target thrust components [x, y, z]
        max_tilt : float, optional
            Maximum tilt angle in radians. Uses self.MAX_TILT_ANGLE if None.
            
        Returns
        -------
        ndarray
            (3,1) array of limited thrust components
        """
        if max_tilt is None:
            max_tilt = self.MAX_TILT_ANGLE
            
        # Calculate current tilt angle
        thrust_norm = np.linalg.norm(target_thrust)
        if thrust_norm < 1e-6:
            return target_thrust
            
        # Get tilt angle (angle from vertical)
        vertical_component = target_thrust[2]
        horizontal_norm = np.linalg.norm(target_thrust[0:2])
        current_tilt = np.arctan2(horizontal_norm, abs(vertical_component))
        
        # If within limits, return unchanged
        if current_tilt <= max_tilt:
            return target_thrust
            
        # Limit the tilt angle
        max_horizontal = abs(vertical_component) * np.tan(max_tilt)
        scale_factor = max_horizontal / horizontal_norm if horizontal_norm > 0 else 1.0
        
        limited_thrust = target_thrust.copy()
        limited_thrust[0] *= scale_factor
        limited_thrust[1] *= scale_factor
        
        return limited_thrust
    
    def _getStartupScale(self, control_timestep):
        """Get scaling factor for gradual startup.
        
        Parameters
        ----------
        control_timestep : float
            Time step for control update
            
        Returns
        -------
        float
            Scaling factor between 0.1 and 1.0
        """
        self.startup_timer += control_timestep
        
        if self.startup_timer >= self.STARTUP_DURATION:
            self.is_startup_complete = True
            return 1.0
        
        # Gradual ramp from 0.1 to 1.0
        progress = self.startup_timer / self.STARTUP_DURATION
        return 0.1 + 0.9 * progress
    
    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        """Safe position control with tilt angle limiting."""
        
        # Get startup scaling
        startup_scale = self._getStartupScale(control_timestep)
        
        # Standard PID computation
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        
        # Scale errors during startup
        if not self.is_startup_complete:
            pos_e *= startup_scale
            vel_e *= startup_scale
        
        self.integral_pos_e = self.integral_pos_e + pos_e * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        
        # Compute target thrust
        target_thrust = (np.multiply(self.P_COEFF_FOR, pos_e) +
                        np.multiply(self.I_COEFF_FOR, self.integral_pos_e) +
                        np.multiply(self.D_COEFF_FOR, vel_e) + 
                        np.array([0, 0, self.GRAVITY]))
        
        # 🛡️ Apply tilt angle limiting
        limited_thrust = self._limitTiltAngles(target_thrust)
        
        # Additional startup limiting
        if not self.is_startup_complete:
            # More conservative tilt during startup
            startup_max_tilt = self.MAX_TILT_ANGLE * 0.5  # 15 degrees during startup
            limited_thrust = self._limitTiltAngles(limited_thrust, startup_max_tilt)
        
        # Compute scalar thrust and PWM
        scalar_thrust = max(0., np.dot(limited_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        
        # Compute target orientation from limited thrust
        target_z_ax = limited_thrust / np.linalg.norm(limited_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        
        # Target Euler angles
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in SafeControl._dslPIDPositionControl(), values outside range [-pi,pi]")
        
        self.last_target_thrust = limited_thrust  # Store the limited thrust
        return thrust, target_euler, pos_e
    
    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """Safe attitude control with motor output limiting."""
        
        # Standard attitude control computation
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w, x, y, z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        
        rot_matrix_e = np.dot((target_rotation.transpose()), cur_rotation) - np.dot(cur_rotation.transpose(), target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy) / control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e * control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        
        # Compute target torques
        target_torques = (-np.multiply(self.P_COEFF_TOR, rot_e) +
                         np.multiply(self.D_COEFF_TOR, rpy_rates_e) +
                         np.multiply(self.I_COEFF_TOR, self.integral_rpy_e))
        
        # 🛡️ More conservative torque limiting
        max_torque = 2400 if not self.is_startup_complete else 3200
        target_torques = np.clip(target_torques, -max_torque, max_torque)
        
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        
        # 🛡️ Apply safe PWM limits
        pwm = np.clip(pwm, self.MIN_PWM, self.SAFE_MAX_PWM)
        
        self.last_target_torques = target_torques
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    def getStatus(self):
        """Get current safety controller status.
        
        Returns
        -------
        dict
            Dictionary containing controller status information
        """
        return {
            'max_tilt_angle_deg': np.degrees(self.MAX_TILT_ANGLE),
            'max_motor_output_pct': self.MAX_MOTOR_OUTPUT_FRACTION * 100,
            'startup_progress_pct': min(100, (self.startup_timer / self.STARTUP_DURATION) * 100),
            'is_startup_complete': self.is_startup_complete,
            'safe_max_pwm': self.SAFE_MAX_PWM,
            'startup_timer': self.startup_timer
        }
