import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class ANFISControl(BaseControl):
    """ANFIS (Adaptive Neuro-Fuzzy Inference System) control class for Crazyflies."""

    def __init__(self,
                 drone_model: DroneModel,
                 g: float = 9.8,
                 num_mfs: int = 3,
                 learning_rate: float = 0.01):
        """Initialize the ANFIS controller with 5-layer architecture."""
        super().__init__(drone_model=drone_model, g=g)
        
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] ANFISControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
            
        self.num_mfs = num_mfs
        self.learning_rate = learning_rate
        
        # ANFIS structure: 3 inputs (error, error_dot, error_integral) for each axis
        self.num_inputs = 3
        self.num_rules = num_mfs ** self.num_inputs
        
        # Initialize separate ANFIS modules for position and attitude control
        self.pos_anfis = {
            'x': self._init_anfis_structure(input_range=[-2.0, 2.0], output_scale=3.0),
            'y': self._init_anfis_structure(input_range=[-2.0, 2.0], output_scale=3.0), 
            'z': self._init_anfis_structure(input_range=[-1.0, 1.0], output_scale=7.9)
        }
        
        self.att_anfis = {
            'roll': self._init_anfis_structure(input_range=[-0.5, 0.5], output_scale=300.0),
            'pitch': self._init_anfis_structure(input_range=[-0.5, 0.5], output_scale=300.0),
            'yaw': self._init_anfis_structure(input_range=[-1.0, 1.0], output_scale=150.0)
        }
        
        # Motor mixing matrix
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([
                [-.5, -.5, -1],
                [-.5,  .5,  1],
                [.5,  .5, -1],
                [.5, -.5,  1]
            ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                [0, -1, -1],
                [+1, 0, 1],
                [0,  1, -1],
                [-1, 0, 1]
            ])
            
        # PWM parameters
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        
        # Hovering PWM
        self.HOVER_PWM = 42000
        
        self.reset()

    def _init_anfis_structure(self, input_range=[-1.0, 1.0], output_scale=1.0):
        """Initialize ANFIS 5-layer structure according to project proposal."""
        # Layer 1: Fuzzification Layer - Gaussian membership functions
        premise_params = {
            'centers': np.linspace(input_range[0], input_range[1], self.num_mfs).reshape(1, -1).repeat(self.num_inputs, axis=0),
            'widths': np.ones((self.num_inputs, self.num_mfs)) * (input_range[1] - input_range[0]) / (2 * self.num_mfs)
        }
        
        # Layer 4: Consequent parameters for Sugeno model
        # f_ij = p_ij * x1 + q_ij * x2 + r_ij * x3 + s_ij (bias)
        consequent_params = np.random.uniform(-0.1, 0.1, (self.num_rules, self.num_inputs + 1))
        
        # Special initialization for z-axis (altitude) control
        if output_scale >= 8.0:  # z-axis control
            # Add positive bias for hover stability
            consequent_params[:, -1] += np.random.uniform(0.1, 0.3, self.num_rules)
        
        return {
            'premise': premise_params,
            'consequent': consequent_params,
            'output_scale': output_scale,
            'input_range': input_range
        }

    def reset(self):
        """Reset the control class state."""
        super().reset()
        
        # Control state variables
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    def layer1_fuzzification(self, inputs, premise_params):
        """Layer 1: Fuzzification - Convert inputs to membership degrees."""
        centers = premise_params['centers']
        widths = premise_params['widths']
        
        membership_matrix = np.zeros((self.num_inputs, self.num_mfs))
        
        for i in range(self.num_inputs):
            for j in range(self.num_mfs):
                # Gaussian membership function
                membership_matrix[i, j] = np.exp(-((inputs[i] - centers[i, j]) / (widths[i, j] + 1e-8)) ** 2)
        
        return membership_matrix

    def layer2_rule_computation(self, membership_matrix):
        """Layer 2: Rule Layer - Compute rule firing strengths using T-norm (product)."""
        rule_strengths = np.zeros(self.num_rules)
        
        rule_idx = 0
        for i in range(self.num_mfs):
            for j in range(self.num_mfs):
                for k in range(self.num_mfs):
                    # T-norm operation (multiplication for AND operation)
                    rule_strengths[rule_idx] = (membership_matrix[0, i] * 
                                              membership_matrix[1, j] * 
                                              membership_matrix[2, k])
                    rule_idx += 1
        
        return rule_strengths

    def layer3_normalization(self, rule_strengths):
        """Layer 3: Normalization Layer - Normalize rule firing strengths."""
        total_strength = np.sum(rule_strengths) + 1e-8  # Avoid division by zero
        normalized_strengths = rule_strengths / total_strength
        return normalized_strengths

    def layer4_defuzzification(self, inputs, normalized_strengths, consequent_params):
        """Layer 4: Defuzzification Layer - Compute weighted rule outputs."""
        rule_outputs = np.zeros(self.num_rules)
        
        for i in range(self.num_rules):
            # Sugeno model: f_i = p_i*x1 + q_i*x2 + r_i*x3 + s_i
            rule_outputs[i] = (consequent_params[i, 0] * inputs[0] + 
                              consequent_params[i, 1] * inputs[1] + 
                              consequent_params[i, 2] * inputs[2] + 
                              consequent_params[i, 3])  # bias term
        
        # Weighted sum
        weighted_outputs = normalized_strengths * rule_outputs
        return weighted_outputs, rule_outputs

    def layer5_output(self, weighted_outputs, output_scale):
        """Layer 5: Output Layer - Final aggregation with scaling."""
        final_output = np.sum(weighted_outputs) * output_scale
        return final_output

    def _anfis_inference(self, inputs, anfis_params):
        """Complete ANFIS inference through all 5 layers."""
        premise = anfis_params['premise']
        consequent = anfis_params['consequent']
        output_scale = anfis_params['output_scale']
        input_range = anfis_params['input_range']
        
        # Normalize inputs to expected range
        clipped_inputs = np.clip(inputs, input_range[0], input_range[1])
        normalized_inputs = clipped_inputs / (np.abs(input_range[1]) + 1e-8)
        
        # Layer 1: Fuzzification
        membership_matrix = self.layer1_fuzzification(normalized_inputs, premise)
        
        # Layer 2: Rule computation
        rule_strengths = self.layer2_rule_computation(membership_matrix)
        
        # Layer 3: Normalization
        normalized_strengths = self.layer3_normalization(rule_strengths)
        
        # Layer 4: Defuzzification
        weighted_outputs, rule_outputs = self.layer4_defuzzification(
            normalized_inputs, normalized_strengths, consequent)
        
        # Layer 5: Final output
        final_output = self.layer5_output(weighted_outputs, output_scale)
        
        return final_output, normalized_strengths, rule_outputs, weighted_outputs

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):
        """Compute ANFIS control action as RPMs."""
        self.control_counter += 1
        
        # Position control
        target_thrust_correction, pos_e = self._anfis_position_control(
            control_timestep, cur_pos, cur_quat, cur_vel, 
            target_pos, target_rpy, target_vel)
        
        target_thrust_vec = target_thrust_correction + np.array([0, 0, self.GRAVITY])
        
        # Attitude control
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        target_z_ax = target_thrust_vec / (np.linalg.norm(target_thrust_vec) + 1e-8)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c)
        target_y_ax /= (np.linalg.norm(target_y_ax) + 1e-8)
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = np.vstack([target_x_ax, target_y_ax, target_z_ax]).transpose()
        computed_target_rpy = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)

        scalar_thrust = max(0., np.dot(target_thrust_vec, cur_rotation[:,2]))
        thrust_pwm = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        
        target_torques, pwm = self._anfis_attitude_control(
            control_timestep, thrust_pwm, cur_quat, 
            computed_target_rpy, target_rpy_rates)
        
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        yaw_error = computed_target_rpy[2] - cur_rpy[2]

        # 收集所有中间输出，用于anfis.py中的学习步骤
        intermediate_outputs = {
            'pos': {axis: data['last_inference'] for axis, data in self.pos_anfis.items()},
            'att': {axis: data['last_inference'] for axis, data in self.att_anfis.items()},
            'computed_thrust_vector': target_thrust_vec,
            'computed_torques': target_torques  # <--- 添加这一行
        }
           
        return rpm, pos_e, yaw_error, intermediate_outputs

    def _anfis_position_control(self, control_timestep, cur_pos, cur_quat, 
                               cur_vel, target_pos, target_rpy, target_vel):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        
        # Update integral with anti-windup
        self.integral_pos_e += pos_e * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -1.0, 1.0)
        
        # ANFIS control for each axis
        target_thrust_correction = np.zeros(3)
        axes = ['x', 'y', 'z']
        
        for i, axis in enumerate(axes):
            # Prepare inputs according to project proposal structure
            pos_input = pos_e[i]
            vel_input = vel_e[i]
            int_input = self.integral_pos_e[i]
            
            inputs = np.array([pos_input, vel_input, int_input])
            
            # ANFIS inference
            control_output, norm_strengths, rule_outs, weighted_outs = self._anfis_inference(inputs, self.pos_anfis[axis])
            target_thrust_correction[i] = control_output
            
            # 保存中间结果用于学习
            self.pos_anfis[axis]['last_inference'] = {
                "inputs": inputs,
                "norm_strengths": norm_strengths,
                "rule_outputs": rule_outs,
                "weighted_outputs": weighted_outs,
                "final_output": control_output
            }
        
        # Apply thrust limits for stability
        # min_z_thrust = self.GRAVITY * 0.7
        # max_z_thrust = self.GRAVITY * 1.8
        # target_thrust[2] = np.clip(target_thrust[2], min_z_thrust, max_z_thrust)
        
        # Convert to thrust and orientation
        # scalar_thrust = max(0.01, np.dot(target_thrust, cur_rotation[:, 2]))
        # thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        
        # Apply hover baseline
        # thrust = max(thrust, self.HOVER_PWM * 0.7)
        
        # Compute target orientation
        # target_z_ax = target_thrust / (np.linalg.norm(target_thrust) + 1e-8)
        # target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        # target_y_ax = np.cross(target_z_ax, target_x_c)
        # target_y_ax = target_y_ax / (np.linalg.norm(target_y_ax) + 1e-8)
        # target_x_ax = np.cross(target_y_ax, target_z_ax)
        # target_rotation = np.vstack([target_x_ax, target_y_ax, target_z_ax]).transpose()
        
        # target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)

        return target_thrust_correction, pos_e

    def _anfis_attitude_control(self, control_timestep, thrust, cur_quat, 
                               target_euler, target_rpy_rates):
        """ANFIS-based attitude control."""
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        
        # Compute attitude error using rotation matrix
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w, x, y, z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = (np.dot(target_rotation.transpose(), cur_rotation) - 
                       np.dot(cur_rotation.transpose(), target_rotation))
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        
        # Angular rate error
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy) / control_timestep
        self.last_rpy = cur_rpy
        
        # Update integral with limits
        self.integral_rpy_e += rot_e * control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -50.0, 50.0)
        
        # ANFIS control for each rotation axis
        target_torques = np.zeros(3)
        axes = ['roll', 'pitch', 'yaw']
        
        for i, axis in enumerate(axes):
            # Prepare normalized inputs
            rot_input = rot_e[i]
            rate_input = rpy_rates_e[i]
            int_input = self.integral_rpy_e[i]

            inputs = np.array([rot_input, rate_input, int_input])
            
            # ANFIS推理
            control_output, norm_strengths, rule_outs, weighted_outs = self._anfis_inference(inputs, self.att_anfis[axis])
            target_torques[i] = control_output

            # 保存中间结果用于学习
            self.att_anfis[axis]['last_inference'] = {
                "inputs": inputs,
                "norm_strengths": norm_strengths,
                "rule_outputs": rule_outs,
                "weighted_outputs": weighted_outs,
                "final_output": control_output
            }
        
        # Apply torque limits
        target_torques = np.clip(target_torques, -2000, 2000)
        
        # Convert to PWM and then RPM
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        
        return target_torques, pwm

    def computeControlFromState(self,
                            control_timestep,
                            state,
                            target_pos,
                            target_rpy=np.zeros(3),
                            target_vel=np.zeros(3),
                            target_rpy_rates=np.zeros(3)):
        """Compute control action from drone state."""
        cur_pos = state[0:3]
        cur_quat = state[3:7]
        cur_vel = state[10:13]
        cur_ang_vel = state[13:16]
        
        return self.computeControl(
            control_timestep=control_timestep,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=target_vel,
            target_rpy_rates=target_rpy_rates
        )
    
    def learn(self, ideal_thrust_vector, ideal_torques, anfis_outputs):

        # 根据PID的输出更新ANFIS参数。

        if self.learning_rate == 0:
            return

        # 1. 学习位置控制 (更新 pos_anfis)
        ideal_thrust_correction = ideal_thrust_vector - np.array([0, 0, self.GRAVITY])
    
        computed_thrust_correction = anfis_outputs['computed_thrust_vector'] - np.array([0, 0, self.GRAVITY])
    
        pos_error_vector = ideal_thrust_correction - computed_thrust_correction
        
        for i, axis in enumerate(['x', 'y', 'z']):
            axis_error = pos_error_vector[i]
            inference_data = anfis_outputs['pos'][axis]
            self._update_anfis_module(self.pos_anfis[axis], axis_error, inference_data)

        # 2. 学习姿态控制 (更新 att_anfis)
        computed_torques = anfis_outputs['computed_torques']
        att_error_vector = ideal_torques - computed_torques

        for i, axis in enumerate(['roll', 'pitch', 'yaw']):
            axis_error = att_error_vector[i]
            inference_data = anfis_outputs['att'][axis]
            self._update_anfis_module(self.att_anfis[axis], axis_error, inference_data)

    def _update_anfis_module(self, anfis_module, error, inference_data):
        """
        使用梯度下降更新单个ANFIS模块的参数。
        这里我们先实现后件参数的更新（LSE部分）。
        """
        inputs = inference_data['inputs']
        norm_strengths = inference_data['norm_strengths']
        
        # 更新后件参数 (Layer 4 - Consequent Parameters)
        # 梯度 = -误差 * 归一化强度 * (输入向量, 1)
        # 这是您提案中提到的LSE和梯度下降的混合学习的简化实现
        grad_consequent = -error * norm_strengths.reshape(-1, 1) * np.append(inputs, 1).reshape(1, -1)
        
        # 应用更新
        anfis_module['consequent'] -= self.learning_rate * grad_consequent

        # TODO: 实现前件参数的更新 (更复杂，涉及反向传播)
        # 完整的实现需要计算误差对每一层输出的偏导，链式法则一直传导回第一层
        # 对于当前项目，仅更新后件参数已经能体现出学习效果
    
