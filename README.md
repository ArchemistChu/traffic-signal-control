
##  项目概述

本项目构建了一个完整的智能交通信号控制仿真平台，旨在通过对比研究不同信号控制策略，为智能交通系统的实际部署提供科学依据。

###  核心特性
- **真实仿真环境** - 基于SUMO的标准四车道交叉口建模
- **三种控制策略** - 固定周期、自适应控制、DQN强化学习完整实现
- **车辆检测模拟** - 模拟YOLO检测系统，包含噪声和延迟
- **全面性能评估** - 交通效率和环境影响多维度指标
- **简洁可视化界面** - 一键启动，实时观察，结果对比



### 核心架构层级
```
应用层 (User Interface)
├── app.py - Streamlit Web界面
└── start.py - 一键启动脚本

控制决策层 (Control Layer)  
├── SignalController - 信号控制策略管理器
│   ├── FixedTimeController - 固定周期控制
│   ├── AdaptiveController - 自适应控制  
│   └── DQNController - DQN强化学习控制
└── RLAgent - 深度强化学习代理 (PyTorch)

感知层 (Perception Layer)
├── VehicleDetector - 车辆检测模拟 (YOLO仿真)
├── InductionLoops - 感应线圈检测器
└── EmissionSensors - 排放监测传感器

仿真环境层 (Simulation Layer)
├── TrafficSimulator - 交通仿真引擎核心
├── SUMO Backend - 微观交通仿真平台
└── TraCI Interface - Python-SUMO实时通信接口
```

##  项目文件结构详解

```
SUMO/
├──  核心配置文件
│   ├── 需求.txt                    # 项目需求文档和架构设计  
│   ├── requirements.txt            # Python依赖包清单
│   ├── README.md                   # 项目说明文档 (本文件)
│   └── setup_guide.md              # SUMO环境安装配置指南
│
├──  启动脚本
│   ├── app.py                      # Streamlit主应用界面 (一键启动)
│   └── start.py                    # 应用启动脚本
│
├──  核心源代码 (src/)
│   ├── traffic_simulator.py       # 交通仿真引擎核心模块
│   │                              # - SUMO连接管理和TraCI通信
│   │                              # - 实时数据收集 (车辆、信号、排放)
│   │                              # - 性能指标计算和导出
│   │                              # - GUI/非GUI模式仿真控制
│   │
│   ├── signal_controller.py       # 信号控制策略实现模块  
│   │                              # - BaseController抽象基类
│   │                              # - FixedTimeController (固定周期)
│   │                              # - AdaptiveController (自适应控制) 
│   │                              # - DQNController (强化学习控制)
│   │                              # - 四相位信号灯逻辑管理
│   │
│   ├── rl_agent.py                # DQN强化学习代理模块
│   │                              # - PyTorch深度神经网络 (256-128-64)
│   │                              # - 经验回放和目标网络机制
│   │                              # - 37维状态空间，8个离散动作
│   │                              # - 综合奖励函数 (等待时间+效率+平衡)
│   │
│   ├── vehicle_detector.py        # 车辆检测模拟模块
│   │                              # - YOLO检测算法仿真
│   │                              # - 检测噪声、延迟、误检模拟  
│   │                              # - 非极大值抑制 (NMS) 算法
│   │                              # - 边界框和置信度计算
│   │
│   └── test_simulation.py         # 系统环境测试脚本
│                                  # - SUMO安装验证
│                                  # - 网络文件格式检查
│                                  # - TraCI连接测试
│                                  # - 各模块功能验证
│
├──  SUMO仿真配置文件  
│   ├── intersection.nod.xml       # 路网节点定义 (十字路口+端点)
│   ├── intersection.edg.xml       # 道路边定义 (车道数、限速、长度)
│   ├── intersection.con.xml       # 车道连接关系 (直行、左转、右转)
│   ├── intersection.net.xml       # 生成的完整路网文件 (netconvert输出)
│   ├── routes.rou.xml              # 车辆路径和交通流定义
│   │                              # - 3种车型: 乘用车、卡车、公交车
│   │                              # - 12条路径: 4方向×3转向
│   │                              # - 泊松分布车流 (可调密度)
│   ├── traffic_lights.add.xml     # 交通附加设施配置
│   │                              # - 8个感应线圈检测器
│   │                              # - 信号灯相位逻辑定义
│   └── simulation.sumocfg          # SUMO主仿真配置文件
│                                  # - 输入/输出文件路径
│                                  # - 仿真时间和步长设置
│                                  # - GUI和日志配置
│
├──  输出数据目录 (output/) 
│   ├── vehicle_data_*.csv         # 车辆轨迹数据 (位置、速度、排放)
│   ├── traffic_light_data_*.csv   # 信号灯状态数据 (相位、时间)
│   ├── performance_metrics_*.json # 性能指标报告 (JSON格式)
│   └── simulation_summary_*.txt   # 仿真总结报告 (可读文本)
│
└── 运行时生成文件
    ├── detectors.xml              # 检测器数据输出
    ├── emissions.xml              # 车辆排放数据  
    ├── fcd_output.xml             # 浮动车数据 (轨迹)
    ├── queue.xml                  # 队列长度统计
    └── summary.xml                # SUMO仿真统计摘要
```

##  技术栈详解

###  核心仿真技术
- **SUMO v1.24.0** - 微观交通仿真平台
  - 车辆动力学建模 (跟车、换道、加减速)
  - 排放模型 (CO₂, NOx, PMx, 燃油消耗)
  - 信号灯控制和检测器仿真
- **TraCI API** - Python与SUMO的实时通信接口
  - 实时数据获取 (车辆位置、速度、等待时间)
  - 动态信号灯控制 (相位切换、时长调整)
  - 仿真步进控制和状态管理

###  人工智能算法
- **PyTorch 2.0+** - 深度学习框架
  - DQN (Deep Q-Network) 强化学习算法
  - 经验回放机制 (Experience Replay)
  - 目标网络 (Target Network) 稳定训练
- **深度神经网络架构**
  - 输入层: 37维状态空间
  - 隐藏层: [256, 128, 64] 神经元
  - 输出层: 8个离散动作
  - 激活函数: ReLU, 优化器: Adam

###  数据处理与分析
- **NumPy 1.24+** - 数值计算和矩阵运算
- **Pandas 2.0+** - 数据清洗、分组聚合、时间序列分析
- **SciPy 1.11+** - 统计分析和数值优化
- **Scikit-learn 1.3+** - 数据预处理和评估指标

###  可视化与界面
- **Streamlit 1.28+** - Web应用框架
  - 响应式布局和实时更新
  - 交互式控件 (选择框、按钮、滑块)  
- **Plotly 5.15+** - 交互式图表库
  - 实时数据可视化 (折线图、柱状图、散点图)
  - 响应式图表和缩放功能
- **Altair 5.0+** - 声明式统计可视化

##  核心算法实现

### 1. 固定周期控制 (FIXED_TIME)
```python
# 算法特点: 预设时序，稳定可靠
相位序列: [
    东西直行 → 30秒,
    东西左转 → 15秒, 
    南北直行 → 30秒,
    南北左转 → 15秒
]
总周期: 120秒 (含间隔时间)
```

###  2. 自适应控制 (ADAPTIVE) 
```python
# 算法逻辑: 基于实时流量动态调整
def adaptive_control_logic():
    1. 检测各车道车辆数量 (感应线圈)
    2. 计算车道权重 = 车辆数 × 等待时间
    3. 选择权重最大的方向组
    4. 动态调整绿灯时长:
       - 基础时间: 20-45秒
       - 延长条件: 队列长度 > 5辆 
       - 最大限制: 60秒 (保证公平性)
    5. 低流量相位跳跃优化
```

###  3. DQN强化学习控制 (DQN)
```python
# 状态空间 (37维)
state_space = [
    车道车辆数量[8],        # 各车道当前车辆数  
    车道队列长度[8],        # 等待车辆队列长度
    车道平均速度[8],        # 车道内车辆平均速度
    检测器占有率[8],        # 感应线圈占有率
    交通灯状态[4],          # 当前相位和剩余时间
    系统指标[1]             # 全局拥堵指数
]

# 动作空间 (8个离散动作)
action_space = [
    0: 保持当前相位,
    1-4: 切换到特定相位 (东西直行/左转, 南北直行/左转),
    5-7: 延长当前绿灯 (5秒/10秒/15秒)
]

# 奖励函数设计
def calculate_reward(state, action, next_state):
    reward = (
        -0.1 * 平均等待时间 +           # 减少等待时间  
        +0.05 * 平均速度 +             # 提高通行速度
        -0.02 * 速度标准差 +           # 平衡各方向流量
        +0.01 * 通行效率 +             # 提升系统吞吐量
        -0.005 * 检测器占有率方差      # 均衡道路使用
    )
    return reward
```

##  核心功能实现

###  1. 实时数据收集系统
- **车辆轨迹追踪**: 实时记录所有车辆的位置、速度、加速度
- **排放数据监测**: CO₂、NOx、PMx等污染物排放量统计  
- **交通流分析**: 车道占有率、流量统计、密度计算
- **信号灯状态**: 相位切换历史、绿灯利用率分析

###  2. 车辆检测模拟系统  
```python
# YOLO检测性能参数
detection_params = {
    检测成功率: 95%,        # 真实车辆检测概率
    误检率: 2%,            # 虚假检测概率  
    检测延迟: 50-200ms,    # 网络传输和处理延迟
    位置噪声: ±0.5m,       # GPS误差模拟
    尺寸噪声: ±10%,        # 边界框误差
}

# NMS非极大值抑制
def non_maximum_suppression(detections, iou_threshold=0.5):
    # 去除重复检测，提高检测质量
```

###  3. 性能评估体系
```python
# 交通效率指标  
traffic_metrics = {
    'avg_waiting_time': '平均等待时间 (秒)',
    'avg_speed': '平均行驶速度 (m/s)', 
    'throughput_per_hour': '系统吞吐量 (辆/小时)',
    'congestion_index': '拥堵指数 (0-1)',
    'queue_length_variance': '队列长度方差'
}

# 环境影响指标
environmental_metrics = {
    'total_co2': 'CO₂总排放量 (mg)',
    'total_nox': 'NOx总排放量 (mg)', 
    'total_pmx': 'PMx总排放量 (mg)',
    'total_fuel': '燃油总消耗 (ml)',
    'emission_per_km': '单位距离排放 (mg/km)'
}
```

### 4. 可视化分析界面
- **一键启动**: 策略选择 → 点击启动 → 观看仿真 → 查看结果
- **实时监控**: SUMO GUI窗口实时显示车辆运动和信号变化  
- **性能对比**: 多策略运行结果自动对比分析
- **历史数据管理**: 保存多次仿真结果，支持清空重置

