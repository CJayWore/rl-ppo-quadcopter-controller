import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 randomize_init: bool = True
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """

        self.RANDOMIZE_INIT = randomize_init
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 15

        self.target_visual_id = None  # Visual ID for the target point in the simulation
        self.connection_line_id = None  # Connection line ID for the target point

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value with improved shaping."""
        state = self._getDroneStateVector(0)
        
        # 如果达到目标，给予大奖励
        if self._computeTerminated():
            return 1000.0
        
        # 计算到目标的距离
        distance_to_target = np.linalg.norm(self.TARGET_POS - state[0:3])
        
        # 1. 距离奖励 - 使用更激进的奖励塑形
        # 近距离时奖励急剧增加
        if distance_to_target < 0.1:
            distance_reward = 100.0 * np.exp(-10.0 * distance_to_target)
        elif distance_to_target < 0.5:
            distance_reward = 50.0 * np.exp(-5.0 * distance_to_target)
        else:
            distance_reward = 20.0 * np.exp(-2.0 * distance_to_target)
        
        # 2. 方向奖励 - 鼓励朝目标方向移动
        target_direction = self.TARGET_POS - state[0:3]
        target_direction_norm = target_direction / (np.linalg.norm(target_direction) + 1e-8)
        velocity = state[10:13]
        velocity_norm = np.linalg.norm(velocity)
        
        if velocity_norm > 0.01:  # 如果有速度
            velocity_direction = velocity / velocity_norm
            direction_reward = 20.0 * np.dot(target_direction_norm, velocity_direction)
            direction_reward = max(direction_reward, 0)  # 只奖励正确方向
        else:
            direction_reward = 0
        
        # 3. 速度适应性奖励
        desired_speed = min(distance_to_target * 2.0, 0.5)  # 远距离快速，近距离慢速
        speed_error = abs(velocity_norm - desired_speed)
        speed_reward = 10.0 * np.exp(-speed_error * 5.0)
        
        # 4. 姿态稳定性奖励
        proximity_scaler = 1.0 / (distance_to_target + 0.05)
        attitude_penalty = np.sum(np.square(state[7:9])) * 2.0  # 基础惩罚 roll, pitch
        angular_velocity_penalty = np.sum(np.square(state[13:16])) * 1.0 # 基础惩罚

        # 应用缩放
        scaled_attitude_penalty = attitude_penalty * proximity_scaler
        scaled_angular_velocity_penalty = angular_velocity_penalty * proximity_scaler
        
        # 5. 边界惩罚 - 强烈惩罚离开合理范围
        boundary_penalty = 0
        if abs(state[0]) > 1.2 or abs(state[1]) > 1.2 or state[2] > 1.8 or state[2] < 0.1:
            boundary_penalty = 100.0
        
        # 6. 时间惩罚 - 鼓励快速到达
        time_penalty = self.step_counter * 0.01
        
        # 组合奖励
        total_reward = (
            distance_reward             # 主要驱动力
            + direction_reward 
            + speed_reward 
            - scaled_attitude_penalty   # 使用缩放后的惩罚
            - scaled_angular_velocity_penalty # 使用缩放后的惩罚
            - boundary_penalty 
            - time_penalty
        )
        
        return total_reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .02:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
    
    def _computeObs(self):
        """使用父类的观察计算，但添加目标信息."""
        # 获取原始的72维观察
        original_obs = super()._computeObs()
        
        # 可选：在观察中添加目标位置信息
        # 但这会改变观察维度，所以暂时不用
        return original_obs

    def _addObstacles(self):
        """Add obstacles and target visualization to the environment."""
        # 调用父类方法
        super()._addObstacles()
        
        # 添加目标点可视化
        self._visualizeTarget()

    def _visualizeTarget(self):
        """Create visual representation of the target position."""
        if self.GUI:  # 只在GUI模式下可视化
            # 创建目标点球体（红色半透明）
            target_visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.05,
                rgbaColor=[1, 0, 0, 0.7],  # 红色半透明
                physicsClientId=self.CLIENT
            )
            
            self.target_visual_id = p.createMultiBody(
                baseMass=0,  # 无质量（仅视觉）
                baseVisualShapeIndex=target_visual,
                basePosition=self.TARGET_POS,
                physicsClientId=self.CLIENT
            )
            
            # 添加目标点周围的圆环（显示目标区域）
            ring_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=0.02,  # 成功距离阈值
                length=0.001,
                rgbaColor=[1, 0, 0, 0.3],  # 红色更透明
                physicsClientId=self.CLIENT
            )
            
            self.target_ring_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=ring_visual,
                basePosition=self.TARGET_POS,
                physicsClientId=self.CLIENT
            )

    def _updateTargetVisualization(self):
        """Update target visualization when target position changes."""
        if self.GUI and self.target_visual_id is not None:
            # 更新目标点位置
            p.resetBasePositionAndOrientation(
                self.target_visual_id,
                self.TARGET_POS,
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT
            )
            
            # 更新目标环位置
            if hasattr(self, 'target_ring_id') and self.target_ring_id is not None:
                p.resetBasePositionAndOrientation(
                    self.target_ring_id,
                    self.TARGET_POS,
                    [0, 0, 0, 1],
                    physicsClientId=self.CLIENT
                )

    def _drawConnectionLine(self):
        """Draw line connecting drone to target."""
        if self.GUI:
            # 获取无人机当前位置
            drone_state = self._getDroneStateVector(0)
            drone_pos = drone_state[0:3]
            
            # 移除之前的连接线
            if self.connection_line_id is not None:
                p.removeUserDebugItem(self.connection_line_id, physicsClientId=self.CLIENT)
            
            # 绘制新的连接线
            self.connection_line_id = p.addUserDebugLine(
                lineFromXYZ=drone_pos,
                lineToXYZ=self.TARGET_POS,
                lineColorRGB=[0, 1, 0],  # 绿色线
                lineWidth=2,
                lifeTime=0.1,  # 短暂显示
                physicsClientId=self.CLIENT
            )

    def reset(self, seed: int = None, options: dict = None):
        """Resets the environment with proper domain randomization."""
        #### 关键修正：重构 reset 逻辑 ####

        # 1. 调用父类的内部重置方法，它只重置计数器等，不移动无人机
        super().reset(seed=seed, options=options)
        
        # 2. 如果开启了随机化，则完全由子类控制随机过程
        if self.RANDOMIZE_INIT:
            # 随机化目标点
            target_xyz = np.random.uniform(low=-0.8, high=0.8, size=(3,))
            target_xyz[2] = np.random.uniform(low=0.5, high=1.8)
            self.TARGET_POS = target_xyz
            
            # 随机化无人机的初始位置和姿态
            initial_xyz = np.random.uniform(low=-0.5, high=0.5, size=(1, 3))
            initial_xyz[0, 2] = np.random.uniform(low=0.1, high=1.5)
            initial_rpy = np.random.uniform(low=-np.pi/12, high=np.pi/12, size=(1, 3))

            # 直接在物理引擎中重置无人机到新的随机状态
            p.resetBasePositionAndOrientation(self.DRONE_IDS[0],
                                              initial_xyz[0],
                                              p.getQuaternionFromEuler(initial_rpy[0]),
                                              physicsClientId=self.CLIENT)
        else:
            # 如果不随机化，则使用默认的初始位置和目标
            self.TARGET_POS = np.array([0,0,1])
            p.resetBasePositionAndOrientation(self.DRONE_IDS[0],
                                              self.initial_xyzs[0],
                                              self.initial_rpys[0],
                                              physicsClientId=self.CLIENT)
            
        self._updateTargetVisualization()

        # 3. 返回符合Gymnasium API的 (observation, info) 元组
        return self._computeObs(), self._computeInfo()

    def step(self, action):
        """Environment step with target visualization updates."""
        # 调用父类的step方法
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 更新连接线可视化（每隔几步更新一次以提高性能）
        if self.step_counter % 5 == 0:  # 每5步更新一次
            self._drawConnectionLine()
        
        return obs, reward, terminated, truncated, info