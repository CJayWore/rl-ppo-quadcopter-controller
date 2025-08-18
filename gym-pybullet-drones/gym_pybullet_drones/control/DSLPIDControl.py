import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class DSLPIDControl(BaseControl):
    """PID control class for Crazyflies.

    This class implements a PID (Proportional-Integral-Derivative) controller specifically
    designed for Crazyflie drones (CF2X and CF2P models). The controller uses cascaded
    position and attitude control loops to compute motor RPMs for stable flight.

    The position controller computes desired thrust and orientation based on position
    and velocity errors, while the attitude controller computes motor torques to achieve
    the desired orientation. The implementation is based on work conducted at UTIAS' DSL.

    Based on work conducted at UTIAS' DSL. Contributors: SiQi Zhou, James Xu, 
    Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    Attributes
    ----------
    P_COEFF_FOR : ndarray
        Proportional gains for position control [x, y, z].
    I_COEFF_FOR : ndarray
        Integral gains for position control [x, y, z].
    D_COEFF_FOR : ndarray
        Derivative gains for position control [x, y, z].
    P_COEFF_TOR : ndarray
        Proportional gains for attitude control [roll, pitch, yaw].
    I_COEFF_TOR : ndarray
        Integral gains for attitude control [roll, pitch, yaw].
    D_COEFF_TOR : ndarray
        Derivative gains for attitude control [roll, pitch, yaw].
    MIXER_MATRIX : ndarray
        Motor mixing matrix to convert torques to individual motor commands.
    """

    ################################################################################
    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Initialize the DSL PID controller.

        Sets up PID gains, motor parameters, and mixer matrix based on the drone model.
        Only supports CF2X and CF2P drone models.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
            Must be either DroneModel.CF2X or DroneModel.CF2P.
        g : float, optional
            The gravitational acceleration in m/s^2. Default is 9.8.

        Raises
        ------
        SystemExit
            If the drone model is not CF2X or CF2P.
        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()


        ## PID for position control - Conservative hovering parameters
        self.P_COEFF_FOR = np.array([0.4, 0.4, 0.6])
        self.I_COEFF_FOR = np.array([0.05, 0.05, 0.1])
        self.D_COEFF_FOR = np.array([0.2, 0.2, 0.3])
        
        ## PID for attitude control - Conservative hovering parameters
        self.P_COEFF_TOR = np.array([8000, 8000, 6000])
        self.I_COEFF_TOR = np.array([1.0, 1.0, 5.0])
        self.D_COEFF_TOR = np.array([800, 800, 600])

        ## Pulse Width Modulation (PWM) to RPM conversion parameters
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535

        ## Safety parameters for preventing flip-over
        self.MAX_TILT_ANGLE = np.pi/6  # 30 degrees maximum tilt
        self.MAX_MOTOR_OUTPUT_FRACTION = 0.85  # Limit motor output to 85%
        self.STARTUP_DURATION = 2.0  # Gradual startup over 2 seconds
        self.SAFE_MAX_PWM = int(self.MAX_PWM * self.MAX_MOTOR_OUTPUT_FRACTION)
        
        # Startup tracking
        self.startup_timer = 0.0
        self.is_startup_complete = False
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])
        self.reset()

    ################################################################################

    def reset(self):
        """Reset the control class state.

        Resets all PID control variables including previous step errors and 
        integral accumulations for both position and attitude controllers.
        This should be called at the start of each episode or when restarting control.
        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
        #### Reset safety parameters ###############################
        self.startup_timer = 0.0
        self.is_startup_complete = False

    ################################################################################
    
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
    
    ################################################################################
    
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

    ################################################################################
    
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Compute the PID control action (as RPMs) for a single drone.

        This method sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`
        to compute the required motor RPMs. The position controller determines the desired
        thrust and orientation, while the attitude controller converts these into motor commands.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed, in seconds.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position [x, y, z] in meters.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion [x, y, z, w].
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity [vx, vy, vz] in m/s.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity [wx, wy, wz] in rad/s.
            Note: This parameter is currently unused in the implementation.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position [x, y, z] in meters.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw in radians.
            Default is zeros (level flight).
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity [vx, vy, vz] in m/s.
            Default is zeros (hover).
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates in rad/s.
            Default is zeros (no rotation).

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error in meters.
        float
            The current yaw error in radians.
        """
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )
        rpm = self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
    
    ################################################################################

    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        """Compute DSL's CF2.x PID position control.

        Implements the outer loop position controller that computes the desired thrust
        magnitude and target orientation based on position and velocity errors.
        Uses PID control with feedforward terms.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed, in seconds.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position [x, y, z] in meters.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion [x, y, z, w].
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity [vx, vy, vz] in m/s.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position [x, y, z] in meters.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw in radians.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity [vx, vy, vz] in m/s.

        Returns
        -------
        float
            The target thrust along the drone z-axis in PWM units.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw in radians.
        ndarray
            (3,1)-shaped array of floats containing the current position error [ex, ey, ez] in meters.
        """
        # Get startup scaling for gradual ramp-up
        startup_scale = self._getStartupScale(control_timestep)
        
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        
        # Apply startup scaling to errors for gentle startup
        if not self.is_startup_complete:
            pos_e *= startup_scale
            vel_e *= startup_scale
        
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        #### PID target thrust #####################################
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        
        #### Apply tilt angle limiting for safety ###############
        limited_thrust = self._limitTiltAngles(target_thrust)
        
        # Additional startup limiting - more conservative during startup
        if not self.is_startup_complete:
            startup_max_tilt = self.MAX_TILT_ANGLE * 0.5  # 15 degrees during startup
            limited_thrust = self._limitTiltAngles(limited_thrust, startup_max_tilt)
        
        scalar_thrust = max(0., np.dot(limited_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = limited_thrust / np.linalg.norm(limited_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        self.last_target_thrust = limited_thrust  # Store the limited thrust for debugging
        return thrust, target_euler, pos_e
    
    ################################################################################

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """Compute DSL's CF2.x PID attitude control.

        Implements the inner loop attitude controller that computes motor RPMs
        based on orientation errors and desired thrust. Uses rotation matrix
        error representation for robust attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed, in seconds.
        thrust : float
            The target thrust along the drone z-axis in PWM units.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion [x, y, z, w].
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles [roll, pitch, yaw] in radians.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates in rad/s.

        Returns
        -------
        ndarray
            (4,1)-shaped array of floats containing the RPMs to apply to each of the 4 motors.
        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        
        # More conservative torque limiting during startup
        max_torque = 2400 if not self.is_startup_complete else 3200
        target_torques = np.clip(target_torques, -max_torque, max_torque)
        
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        
        # Apply safe PWM limits to prevent motor over-saturation
        pwm = np.clip(pwm, self.MIN_PWM, self.SAFE_MAX_PWM)
        
        self.last_target_torques = target_torques
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    ################################################################################

    def _one23DInterface(self,
                         thrust
                         ):
        """Convert 1, 2, or 3D thrust input to 4-motor PWM commands.

        Utility function that handles different thrust input dimensions and converts
        them to individual motor PWM commands. Useful for simplified control interfaces.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing desired thrust input(s).
            - Length 1: Single thrust applied to all motors
            - Length 2: Differential thrust for pitch/roll control
            - Length 4: Individual motor thrusts

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        Raises
        ------
        SystemExit
            If thrust array length is not 1, 2, or 4.
        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()

    def getLastThrustAndTorques(self):
        
        return getattr(self, 'last_target_thrust', np.zeros(3)), getattr(self, 'last_target_torques', np.zeros(3))
    
    ################################################################################
    
    def getSafetyStatus(self):
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
            'startup_timer': self.startup_timer,
            'startup_duration': self.STARTUP_DURATION
        }
