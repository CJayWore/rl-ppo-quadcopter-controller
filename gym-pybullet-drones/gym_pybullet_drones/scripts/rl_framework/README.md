# Modular RL Framework for Drone Control (Unified Task Only)

## 📁 文件结构

```
rl_framework/                    # 模块化框架目录
├── __init__.py                 # 包初始化和导出
├── config.py                   # 配置类 (TrainingConfig, EnvironmentConfig, ModelConfig)
├── notifications.py            # 系统通知管理
├── environment.py              # 环境工厂和配置 (只支持UnifiedAviary)
├── model_manager.py           # 模型加载、保存和管理
├── trainer.py                 # 主要训练器类
└── utils.py                   # 工具函数

rl_modular.py                  # 新的主要脚本（使用模块化结构）
rl_refactored.py              # 重构但单文件版本
rl.py                         # 原始版本
```

## 🎯 重要说明

**此版本只支持 `unified` 任务！**
- ✅ 专注于 UnifiedAviary 环境
- ✅ 支持障碍物开关 (`--enable_obstacles`)
- ✅ 更简洁的代码结构
- ❌ 不再支持 hover、trajectory、obstacle 单独任务

## 🔧 模块说明

### 1. **config.py** - 配置管理
- `TrainingConfig`: 训练参数（默认任务为unified）
- `EnvironmentConfig`: 环境参数
- `ModelConfig`: 模型参数

### 2. **environment.py** - 环境管理（简化版）
- `EnvironmentFactory`: 只创建 UnifiedAviary 环境
- 统一的网络架构配置（复杂网络）
- 固定目标奖励：500000.0

### 4. **model_manager.py** - 模型管理
- `ModelManager`: PPO 模型管理
- 模型加载、保存、兼容性检查
- 学习率更新和权重迁移

### 5. **trainer.py** - 训练器
- `DroneRLTrainer`: 主要训练和评估逻辑
- 训练流程管理和结果可视化
- 评估循环和性能统计

### 6. **utils.py** - 工具函数
- `create_argument_parser`: 命令行参数解析
- 其他辅助工具函数

## 🚀 使用方法

### 基本使用
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

### 高级使用
```bash
# 无障碍物训练
python rl_modular.py --task unified --enable_obstacles False --episodes 1000 --gui False

# 有障碍物评估
python rl_modular.py --task unified --enable_obstacles True --train_mode False --gui True --duration_sec 60

# 轨迹跟踪任务
python rl_modular.py --task trajectory --trajectory_type figure8 --episodes 500
```

## ✅ 优势

### 1. **模块化设计**
- 每个类都有单一职责
- 易于测试和维护
- 代码重用性高

### 2. **配置管理**
- 使用 dataclass 管理配置
- 类型安全和自动验证
- 易于扩展新参数

### 3. **错误处理**
- 更好的异常处理
- 用户友好的错误信息
- 优雅的失败恢复

### 4. **可扩展性**
- 易于添加新任务类型
- 插件式架构
- 松耦合设计

## 🔄 从原版本迁移

### 兼容性
- 所有命令行参数保持不变
- 行为和结果完全一致
- 可以与原版本并行使用

### 推荐迁移步骤
1. **测试新版本**: 使用相同参数运行确保功能正常
2. **逐步切换**: 可以同时保留两个版本
3. **自定义扩展**: 基于模块化结构添加新功能

## 🛠️ 开发和扩展

### 添加新任务类型
1. 在 `environment.py` 中添加环境配置
2. 在 `EnvironmentFactory` 中注册新环境
3. 更新配置和参数解析

### 添加新功能
1. 确定功能属于哪个模块
2. 扩展相应的类
3. 更新 `__init__.py` 导出

### 自定义配置
```python
from rl_framework import TrainingConfig, DroneRLTrainer

# 自定义配置
config = TrainingConfig(
    task="unified",
    episodes=2000,
    learning_rate=1e-4,
    enable_obstacles=True
)

# 使用自定义配置
trainer = DroneRLTrainer(config, env_config, model_config)
```

## 📝 注意事项

1. **导入路径**: 确保 `rl_framework` 在 Python 路径中
2. **依赖项**: 所有原始依赖项仍然需要
3. **文件权限**: 脚本具有可执行权限

## 🐛 故障排除

### 模块导入问题
```bash
# 确保在正确目录下运行
cd /path/to/gym-pybullet-drones/gym_pybullet_drones/examples/
python rl_modular.py --help
```

### 依赖问题
```bash
# 检查所有依赖
pip list | grep -E "(stable-baselines3|gym|pybullet)"
```
