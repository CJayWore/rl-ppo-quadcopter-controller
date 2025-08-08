# Modular RL Framework for Drone Control (Unified Task Only)

## 📁 文件结构

```
gym_pybullet_drones/scripts/
drl.py                   # 主要脚本
rl_framework/                    # 模块化框架目录
├── __init__.py                 # 包初始化和导出
├── config.py                   # 配置类 (TrainingConfig, EnvironmentConfig, ModelConfig)
├── notifications.py            # 系统通知管理
├── environment.py              # 环境工厂和配置 (只支持UnifiedAviary)
├── model_manager.py            # 模型加载、保存和管理
├── trainer.py                  # 主要训练器类
└── utils.py                    # 工具函数

gym_pybullet_drones/envs/DRLAviary.py # 强化学习环境核心定义
```

---

## 🔍 主要模块与功能说明

### 1. `drl.py`
- 命令行入口脚本，负责参数解析、配置生成、训练/评估流程启动。
- 支持训练、评估、模型管理、断点续训等多种模式。

### 2. `rl_framework/config.py`
- `TrainingConfig`：训练参数（如任务类型、学习率、训练轮数等）
- `EnvironmentConfig`：环境参数（如观测类型、动作类型、GUI等）
- `ModelConfig`：模型参数（如模型保存路径、是否加载已有模型等）

### 3. `rl_framework/environment.py`
- `EnvironmentFactory`：根据配置创建强化学习环境（目前只支持UnifiedAviary/DRLAviary）
- 环境参数自动注入，支持障碍物、目标点等配置

### 4. `rl_framework/model_manager.py`
- `ModelManager`：负责模型的保存、加载、管理、权重迁移、模型列表等
- 支持Stable-Baselines3 PPO模型的兼容性检查

### 5. `rl_framework/trainer.py`
- `DroneRLTrainer`：训练主循环、评估主循环、日志与可视化、性能统计
- 支持断点续训、定期评估、自动保存最佳模型

### 6. `rl_framework/utils.py`
- `create_argument_parser`：命令行参数解析
- 其他辅助函数

### 7. `gym_pybullet_drones/envs/DRLAviary.py`
- 强化学习环境核心，定义了 observation space、action space、奖励函数、终止条件等
- 支持障碍物、目标导航、悬停等多任务融合

---

## 🤖 DRL算法与环境说明

### 算法框架
- 使用 [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) 的 PPO 算法
- 支持多线程采样、断点续训、定期评估、自动保存最佳模型
- 可自定义网络结构、学习率、奖励函数等

### 训练流程
1. 解析命令行参数，生成配置对象
2. 创建环境（DRLAviary），自动注入障碍物、目标点等
3. 初始化PPO智能体，加载或新建模型
4. 进入训练主循环，周期性评估与保存
5. 训练完成后可直接评估或继续训练

---

## 🧠 Observation Space (观测空间)

环境观测空间为一维向量，包含以下所有信息，每一项都为智能体学习特定能力服务：

| 维度区间      | 含义                        | 作用/目的说明 |
|---------------|-----------------------------|----------------|
| 0-2           | 位置 (Position, x/y/z)      | 无人机绝对位置，辅助全局定位与导航 |
| 3-5           | 姿态 (Roll, Pitch, Yaw)     | 姿态感知，便于姿态稳定与控制 |
| 6-8           | 线速度 (Velocity, x/y/z)    | 速度感知，便于速度控制与减速悬停 |
| 9-11          | 角速度 (Angular Velocity)   | 姿态变化速率，便于角速度约束与抑制旋转 |
| 12-71         | 电机历史转速 (4电机x15步)   | 运动平滑性、动力学记忆，辅助学习动力学约束与动作平滑 |
| 72-83         | 激光雷达 (12方向)           | 障碍物感知，避障能力的关键输入 |
| 84-86         | 世界坐标系相对目标位置      | 全局导航，辅助无人机朝向目标点移动 |
| 87-89         | 机体坐标系相对目标位置      | 局部导航，便于学习机体朝向下的精细调整 |
| 90-92         | 机体坐标系速度              | 局部速度控制，便于多方向精细调整与减速 |
| 93            | 到目标的欧几里得距离         | 距离奖励、任务完成判据 |
| 94            | 目标角度 (相对yaw)          | 朝向调整，便于学习转向目标点 |
| 95-96         | 角度的sin/cos表示           | 角度平滑、消除角度不连续性，便于神经网络处理 |

> **总维度** = 97（如有扩展请同步更新）

#### 典型观测向量示例
```
[位置x, 位置y, 位置z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, 电机1历史..., 电机4历史..., 激光1, ..., 激光12, 世界目标x, 世界目标y, 世界目标z, 机体目标前后, 机体目标左右, 机体目标上下, 机体速度前后, 机体速度左右, 机体速度上下, 距离, 角度, sin(角度), cos(角度)]
```

---

## 🎮 Action Space (动作空间)

- **类型**：连续动作空间（Box）
- **维度**：4
- **含义**：四个电机的转速（RPM），范围通常为 [0, MAX_RPM]
- **接口**：`env.action_space = spaces.Box(low=0, high=MAX_RPM, shape=(4,), dtype=np.float32)`
- **智能体输出**：每步输出一个长度为4的向量，分别对应四个电机的控制信号

---

## 🏆 Reward Design (奖励设计)

- **导航奖励**：鼓励无人机靠近目标点，奖励与距离成反比
- **多方向速度奖励**：鼓励无人机在机体坐标系下朝正确方向移动（前后/左右/上下）
- **yaw抑制奖励**：接近目标时抑制不必要的yaw旋转
- **姿态/角速度奖励**：分别对roll/pitch/yaw角速度进行约束，悬停时yaw要求更严格
- **悬停奖励**：在目标点附近且速度/角速度足够小时，奖励持续悬停
- **避障奖励**：远离障碍物有正奖励，靠近/碰撞有惩罚
- **完成奖励**：在目标点稳定悬停一定时间后给予一次性大额奖励

---

## 🗺️ 典型训练命令

```bash
# 训练
python rl_modular.py --task unified --train_mode True --episodes 1000

# 评估
python rl_modular.py --task unified --train_mode False --gui True

# 继续训练
python rl_modular.py --task unified --train_mode True --load_model results/unified/best_model.zip

# 列出可用模型
python rl_modular.py --list_models
```

---

## 📝 注意事项

1. **观测空间和动作空间如有修改，需重新训练模型**
2. **建议使用GPU加速训练**
3. **奖励函数可根据实际任务需求自定义调整**
4. **如需扩展新任务/环境，建议参考`DRLAviary.py`的实现方式**

---

## 📚 参考
- [Stable-Baselines3官方文档](https://stable-baselines3.readthedocs.io/)
- [PyBullet官方文档](https://pybullet.org/)
- [gymnasium官方文档](https://gymnasium.farama.org/)

---
## English

# Modular RL Framework for Drone Control (Unified Task Only)

## 📁 Directory Structure

```
gym_pybullet_drones/scripts/
drl.py                   # Main script
rl_framework/            # Modular framework directory
├── __init__.py          # Package initialization and exports
├── config.py            # Config classes (TrainingConfig, EnvironmentConfig, ModelConfig)
├── notifications.py     # System notification management
├── environment.py       # Environment factory and config (UnifiedAviary only)
├── model_manager.py     # Model loading, saving, management
├── trainer.py           # Main trainer class
└── utils.py             # Utility functions

gym_pybullet_drones/envs/DRLAviary.py # Core RL environment definition
```

---

## 🔍 Main Modules & Functionality

### 1. `drl.py`
- Command-line entry script for argument parsing, config generation, and launching training/evaluation.
- Supports training, evaluation, model management, and resuming from checkpoints.

### 2. `rl_framework/config.py`
- `TrainingConfig`: Training parameters (task type, learning rate, episodes, etc.)
- `EnvironmentConfig`: Environment parameters (observation/action type, GUI, etc.)
- `ModelConfig`: Model parameters (save path, load existing model, etc.)

### 3. `rl_framework/environment.py`
- `EnvironmentFactory`: Creates RL environment based on config (currently UnifiedAviary/DRLAviary only)
- Auto-injects environment parameters, supports obstacles, target points, etc.

### 4. `rl_framework/model_manager.py`
- `ModelManager`: Handles model saving, loading, management, weight transfer, model listing
- Supports Stable-Baselines3 PPO model compatibility

### 5. `rl_framework/trainer.py`
- `DroneRLTrainer`: Main training loop, evaluation loop, logging/visualization, performance stats
- Supports checkpointing, periodic evaluation, auto-saving best model

### 6. `rl_framework/utils.py`
- `create_argument_parser`: Command-line argument parsing
- Other helper functions

### 7. `gym_pybullet_drones/envs/DRLAviary.py`
- Core RL environment: defines observation space, action space, reward function, termination conditions, etc.
- Supports multi-task fusion: obstacles, target navigation, hovering

---

## 🤖 DRL Algorithm & Environment

### Algorithm Framework
- Uses [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) PPO algorithm
- Supports multi-threaded sampling, checkpointing, periodic evaluation, auto-saving best model
- Customizable network architecture, learning rate, reward function, etc.

### Training Flow
1. Parse command-line arguments, generate config objects
2. Create environment (DRLAviary), auto-inject obstacles, target points, etc.
3. Initialize PPO agent, load or create model
4. Enter main training loop, with periodic evaluation and saving
5. After training, evaluate or continue training as needed

---

## 🧠 Observation Space

The environment observation is a 1D vector containing all information needed for the agent to learn specific skills. Each item serves a distinct purpose:

| Index Range   | Meaning                        | Purpose/Explanation |
|---------------|-------------------------------|---------------------|
| 0-2           | Position (x/y/z)               | Absolute drone position for global localization and navigation |
| 3-5           | Attitude (Roll, Pitch, Yaw)    | Attitude awareness for stability and control |
| 6-8           | Linear Velocity (x/y/z)        | Velocity awareness for speed control and hovering |
| 9-11          | Angular Velocity               | Attitude change rate for constraining and suppressing rotation |
| 12-71         | Motor RPM History (4 motors x 15 steps) | Motion smoothness, dynamics memory, helps learn dynamic constraints and smooth actions |
| 72-83         | Lidar (12 directions)          | Obstacle perception, key for obstacle avoidance |
| 84-86         | Target Position (world frame)  | Global navigation, helps move toward the target |
| 87-89         | Target Position (body frame)   | Local navigation, enables fine adjustment in body frame |
| 90-92         | Body Frame Velocity            | Local velocity control for fine multi-directional adjustment and deceleration |
| 93            | Euclidean Distance to Target   | Distance reward, task completion criterion |
| 94            | Target Angle (relative yaw)    | Heading adjustment, helps learn to turn toward the target |
| 95-96         | Angle sin/cos representation   | Smooth angle representation, removes discontinuity for neural networks |

> **Total Dimensions** = 97 (update if extended)

#### Example Observation Vector
```
[pos_x, pos_y, pos_z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, motor1_hist..., motor4_hist..., lidar1, ..., lidar12, target_x_world, target_y_world, target_z_world, target_x_body, target_y_body, target_z_body, body_vx, body_vy, body_vz, distance, angle, sin(angle), cos(angle)]
```

---

## 🎮 Action Space

- **Type**: Continuous (Box)
- **Dimensions**: 4
- **Meaning**: RPMs for four motors, typically in [0, MAX_RPM]
- **Interface**: `env.action_space = spaces.Box(low=0, high=MAX_RPM, shape=(4,), dtype=np.float32)`
- **Agent Output**: Each step outputs a 4D vector, each for one motor

---

## 🏆 Reward Design

- **Navigation Reward**: Encourages approaching the target, inversely proportional to distance
- **Multi-directional Velocity Reward**: Encourages moving in the correct direction in body frame (forward/backward, left/right, up/down)
- **Yaw Suppression Reward**: Suppresses unnecessary yaw rotation near the target
- **Attitude/Angular Velocity Reward**: Constrains roll/pitch/yaw angular velocity, stricter yaw requirement when hovering
- **Hovering Reward**: Rewards stable hovering near the target with low velocity/angular velocity
- **Obstacle Avoidance Reward**: Positive reward for staying away from obstacles, penalty for approaching/colliding
- **Completion Reward**: Large one-time reward for stable hovering at the target for a set duration

---

## 🗺️ Typical Training Commands

```bash
# Training
python rl_modular.py --task unified --train_mode True --episodes 1000

# Evaluation
python rl_modular.py --task unified --train_mode False --gui True

# Continue Training
python rl_modular.py --task unified --train_mode True --load_model results/unified/best_model.zip

# List Available Models
python rl_modular.py --list_models
```

---

## 📝 Notes

1. **If the observation or action space changes, retrain the model**
2. **GPU acceleration is recommended for training**
3. **Reward function can be customized for specific tasks**
4. **To extend new tasks/environments, refer to `DRLAviary.py` implementation**

---

## 📚 References
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PyBullet Documentation](https://pybullet.org/)
- [gymnasium Documentation](https://gymnasium.farama.org/)