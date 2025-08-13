# Deep Reinforcement Learning for Drone Control

This document outlines the structure and functionality of the deep reinforcement learning framework designed for drone control tasks.

## Acknowledgement
This project is based on [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
## 📁 File Structure

```
gym_pybullet_drones/scripts/
├── drl.py                   # Main script for training and evaluation
└── rl_framework/            # Modular framework directory
    ├── __init__.py          # Package initialization and exports
    ├── config.py            # Configuration classes (Training, Environment, Model)
    ├── environment.py       # Environment factory and configuration
    ├── gaussian_noise.py    # Gaussian noise management for simulation
    ├── model_manager.py     # Model loading, saving, and management
    ├── notifications.py     # System notification handling
    ├── performance_evaluation.py # Performance evaluation and logging
    ├── trainer.py           # Main trainer class for RL models
    └── utils.py             # Utility functions and argument parser

gym_pybullet_drones/envs/
└── DRLAviary.py             # Core reinforcement learning environment definition
```

---

## Core Modules and Functionality

### 1. `drl.py`
This is the main entry point for all operations. It handles command-line argument parsing, configuration object creation, and initiates the training or evaluation processes. It supports various modes including training, evaluation, model management, and performance analysis.

### 2. `rl_framework/config.py`
- **TrainingConfig**: Defines training-related parameters such as task type, learning rate, and number of episodes.
- **EnvironmentConfig**: Manages environment settings like observation type, action type, and whether to run with a GUI.
- **ModelConfig**: Handles model-specific parameters, including model save paths and whether to load a pre-existing model.

### 3. `rl_framework/environment.py`
- **EnvironmentFactory**: A factory class responsible for creating the reinforcement learning environment based on the specified configuration. It supports dynamic injection of parameters like obstacle presence and target locations.

### 4. `rl_framework/model_manager.py`
- **ModelManager**: Manages the lifecycle of RL models, including saving, loading, listing available models, and checking for compatibility with Stable-Baselines3 PPO.

### 5. `rl_framework/trainer.py`
- **DroneRLTrainer**: Contains the main training and evaluation loops. It manages logging, visualization, performance statistics, and handles features like resuming from checkpoints and periodic model evaluation.

### 6. `rl_framework/performance_evaluation.py`
- **DronePerformanceLogger**: A comprehensive tool for evaluating model performance. It collects detailed metrics, generates reports (Markdown, LaTeX), and creates visualizations of flight trajectories and key performance indicators.

### 7. `rl_framework/gaussian_noise.py`
- **GaussianNoiseManager**: Implements functionality to add and manage Gaussian noise to the simulation, affecting sensors, actions, and observations to improve model robustness.

### 8. `gym_pybullet_drones/envs/DRLAviary.py`
This is the core of the reinforcement learning environment. It defines the observation space, action space, reward function, and termination conditions. It integrates multiple sub-tasks such as obstacle avoidance, target navigation, and hovering.

---

## RL Algorithm and Environment

### Algorithm Framework
- The framework utilizes the **Proximal Policy Optimization (PPO)** algorithm from the [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) library.
- It supports multi-threaded data sampling, resuming from checkpoints, periodic evaluation, and automatic saving of the best-performing model.
- The network architecture, learning rate, and reward function can be customized.

---

## Observation Space

The observation space is a one-dimensional vector designed to provide the agent with all necessary information for the unified task. Each component serves a specific purpose:

| Dimension Range | Meaning                   | Purpose and Description                                       |
|-----------------|---------------------------|---------------------------------------------------------------|
| 0-2             | Position (x, y, z)        | Absolute position of the drone for global localization and navigation. |
| 3-5             | Attitude (Roll, Pitch, Yaw) | Orientation awareness for attitude stabilization and control. |
| 6-8             | Linear Velocity (x, y, z) | Velocity perception for speed control and deceleration for hovering. |
| 9-11            | Angular Velocity          | Rate of attitude change for angular velocity constraints and stability. |
| 12-71           | Motor Speed History       | Provides a memory of dynamics for learning motion smoothness and constraints. |
| 72-83           | LiDAR (12 directions)     | Key input for obstacle perception and avoidance.              |
| 84-86           | World-Frame Target Vector | Global navigation cue, directing the drone towards the target. |
| 87-89           | Body-Frame Target Vector  | Local navigation cue for fine-tuned adjustments relative to the drone's orientation. |
| 90-92           | Body-Frame Velocity       | Local velocity control for precise multi-directional adjustments and deceleration. |
| 93              | Euclidean Distance to Target | Serves as a basis for distance-based rewards and task completion criteria. |
| 94              | Angle to Target (relative to yaw) | Orientation adjustment for learning to turn towards the target. |
| 95-96           | sin/cos of Angle          | Provides a smooth, continuous representation of the angle for the neural network. |

**Total Dimensions**: 97

---

## Action Space

The action space defines the possible outputs of the agent. For this framework, it is configured as follows:

- **Type**: `RPM` (Revolutions Per Minute)
- **Shape**: (1, 4) Numbers of drones and RPM of 4 motors
- **Description**: A continuous 4-dimensional vector where each element corresponds to the RPM value for one of the four drone motors. This allows for direct and fine-grained control over the drone's thrust and movement.

---

## Usage

### Preparation
```bash
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`
```

### Command Line Interface

The main script `drl.py` provides a command-line interface for all major functions. To see all available arguments and their descriptions:

```bash
python drl.py --help
```

### Basic Usage Examples

#### Training
Start a new training session:
```bash
python drl.py --task unified --train_mode True --episodes 10000
```

Continue training from a saved model:
```bash
python drl.py --task unified --train_mode True --load_model results/unified/best_model.zip --episodes 5000
```

#### Evaluation
Evaluate a trained model with GUI visualization:
```bash
python drl.py --task unified --train_mode False --gui True --load_model results/unified/best_model.zip
```

#### Performance Evaluation
Run comprehensive performance evaluation:
```bash
python drl.py --performance_eval --evaluate_model results/unified/best_model.zip --eval_episodes 50
```

#### Utility Commands
List all available trained models:
```bash
python drl.py --list_models
```

---

## Notes

1. **If the observation or action space changes, retrain the model**
2. **GPU acceleration is recommended for training**
3. **Reward function can be customized for specific tasks**
4. **To extend new tasks/environments, refer to `DRLAviary.py` implementation**

---

## References
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PyBullet Documentation](https://pybullet.org/)
- [gymnasium Documentation](https://gymnasium.farama.org/)

---

# 无人机控制模块化强化学习框架

本文档概述了专为无人机控制任务设计的模块化强化学习框架的结构和功能。

## 致谢
本项目基于 [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)

## 文件结构

框架组织结构如下：

```
gym_pybullet_drones/scripts/
├── drl.py                   # 训练和评估的主脚本
└── rl_framework/            # 模块化框架目录
    ├── __init__.py          # 包初始化和导出
    ├── config.py            # 配置类（训练、环境、模型）
    ├── environment.py       # 环境工厂和配置
    ├── gaussian_noise.py    # 仿真高斯噪声管理
    ├── model_manager.py     # 模型加载、保存和管理
    ├── notifications.py     # 系统通知处理
    ├── performance_evaluation.py # 性能评估和日志记录
    ├── trainer.py           # RL模型主训练器类
    └── utils.py             # 工具函数和参数解析器

gym_pybullet_drones/envs/
└── DRLAviary.py             # 核心强化学习环境定义
```

---

## 核心模块和功能

### 1. `drl.py`
这是所有操作的主入口点。它处理命令行参数解析、配置对象创建，并启动训练或评估过程。支持各种模式，包括训练、评估、模型管理和性能分析。

### 2. `rl_framework/config.py`
- **TrainingConfig**: 定义训练相关参数，如任务类型、学习率和训练轮数
- **EnvironmentConfig**: 管理环境设置，如观测类型、动作类型和是否运行GUI
- **ModelConfig**: 处理模型特定参数，包括模型保存路径和是否加载预存在的模型

### 3. `rl_framework/environment.py`
- **EnvironmentFactory**: 负责根据指定配置创建强化学习环境的工厂类。支持动态注入参数，如障碍物存在和目标位置

### 4. `rl_framework/model_manager.py`
- **ModelManager**: 管理RL模型的生命周期，包括保存、加载、列出可用模型，以及检查与Stable-Baselines3 PPO的兼容性

### 5. `rl_framework/trainer.py`
- **DroneRLTrainer**: 包含主要的训练和评估循环。管理日志记录、可视化、性能统计，并处理从检查点恢复和定期模型评估等功能

### 6. `rl_framework/performance_evaluation.py`
- **DronePerformanceLogger**: 用于评估模型性能的综合工具。收集详细指标，生成报告（Markdown、LaTeX），并创建飞行轨迹和关键性能指标的可视化

### 7. `rl_framework/gaussian_noise.py`
- **GaussianNoiseManager**: 实现向仿真中添加和管理高斯噪声的功能，影响传感器、动作和观测，以提高模型鲁棒性

### 8. `gym_pybullet_drones/envs/DRLAviary.py`
这是强化学习环境的核心。它定义了观测空间、动作空间、奖励函数和终止条件。整合多个子任务，如避障、目标导航和悬停。

---

## 强化学习算法和环境

### 算法框架
- 框架利用来自[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)库的**近端策略优化（PPO）**算法
- 支持多线程数据采样、从检查点恢复、定期评估和自动保存最佳性能模型
- 网络架构、学习率和奖励函数可以自定义

### 训练流程
1. 解析命令行参数以生成配置对象
2. 创建带有动态注入参数（如障碍物、目标位置）的`DRLAviary`环境
3. 初始化PPO智能体，通过创建新模型或加载现有模型
4. 进入主训练循环，包括定期评估和模型保存
5. 训练后，可以评估模型或用于进一步训练

---

## 观测空间

观测空间是一个一维向量，旨在为智能体提供统一任务所需的所有必要信息。每个组件都有特定用途：

| 维度范围 | 含义 | 用途和描述 |
|---------|------|----------|
| 0-2 | 位置 (x, y, z) | 无人机绝对位置，用于全局定位和导航 |
| 3-5 | 姿态 (Roll, Pitch, Yaw) | 方向感知，用于姿态稳定和控制 |
| 6-8 | 线速度 (x, y, z) | 速度感知，用于速度控制和悬停减速 |
| 9-11 | 角速度 | 姿态变化率，用于角速度约束和稳定性 |
| 12-71 | 电机转速历史 | 提供动力学记忆，用于学习运动平滑性和约束 |
| 72-83 | 激光雷达（12个方向） | 障碍物感知和避障的关键输入 |
| 84-86 | 世界坐标系目标向量 | 全局导航线索，引导无人机朝向目标 |
| 87-89 | 机体坐标系目标向量 | 局部导航线索，用于相对于无人机方向的精细调整 |
| 90-92 | 机体坐标系速度 | 局部速度控制，用于精确的多方向调整和减速 |
| 93 | 到目标的欧几里得距离 | 作为基于距离的奖励和任务完成标准的基础 |
| 94 | 目标角度（相对于偏航） | 方向调整，用于学习朝向目标转向 |
| 95-96 | 角度的sin/cos表示 | 为神经网络提供平滑、连续的角度表示 |

**总维度**: 97

### 示例观测向量
```
[pos_x, pos_y, pos_z, roll, pitch, yaw, vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z, motor_history..., lidar_readings..., world_target_vec..., body_target_vec..., body_vel_vec..., distance, angle, sin(angle), cos(angle)]
```

---

## 动作空间

动作空间定义了智能体的可能输出。对于此框架，配置如下：

- **类型**: `RPM`（每分钟转数）
- **形状**: (1, 4) 无人机数量和4个电机的RPM
- **描述**: 一个连续的4维向量，其中每个元素对应四个无人机电机之一的RPM值。这允许对无人机的推力和运动进行直接和精细的控制

---

## 使用方法

### 准备工作
```bash
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # 如果需要，使用`sudo apt install build-essential`安装`gcc`并构建`pybullet`
```

### 命令行界面

主脚本`drl.py`为所有主要功能提供命令行界面。要查看所有可用参数及其描述：

```bash
python drl.py --help
```

### 基本使用示例

#### 训练
开始新的训练会话：
```bash
python drl.py --task unified --train_mode True --episodes 10000
```

从保存的模型继续训练：
```bash
python drl.py --task unified --train_mode True --load_model results/unified/best_model.zip --episodes 5000
```

#### 评估
使用GUI可视化评估训练好的模型：
```bash
python drl.py --task unified --train_mode False --gui True --load_model results/unified/best_model.zip
```

#### 性能评估
运行综合性能评估：
```bash
python drl.py --performance_eval --evaluate_model results/unified/best_model.zip --eval_episodes 50
```

#### 实用命令
列出所有可用的训练模型：
```bash
python drl.py --list_models
```

---

## 注意事项

1. **如果观测或动作空间发生变化，需要重新训练模型**
2. **建议使用GPU加速训练**
3. **奖励函数可以根据具体任务自定义**
4. **要扩展新任务/环境，请参考`DRLAviary.py`的实现**

---

## 参考文献
- [Stable-Baselines3文档](https://stable-baselines3.readthedocs.io/)
- [PyBullet文档](https://pybullet.org/)
- [gymnasium文档](https://gymnasium.farama.org/)