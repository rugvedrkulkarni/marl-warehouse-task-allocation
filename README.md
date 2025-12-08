# Multi-Agent Reinforcement Learning for Warehouse Task Allocation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Multi-Agent Reinforcement Learning (MARL) framework for intelligent task allocation in warehouse environments. This project implements a custom Gymnasium environment simulating a warehouse with a dispatcher agent and multiple workstation acceptors, demonstrating how RL can optimize throughput in logistics operations similar to Amazon fulfillment centers, Tesla assembly lines, and BMW manufacturing facilities.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Environment Architecture](#environment-architecture)
- [Algorithms Implemented](#algorithms-implemented)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)
- [Author](#author)

## 🎯 Project Overview

This project addresses the challenge of **real-time task allocation** in warehouse sortation systems where:
- Jobs arrive dynamically at a dispatcher
- Multiple workstations with limited queue capacity must process jobs
- Intelligent routing decisions maximize throughput while respecting constraints

### Key Features

- **Custom Multi-Agent Environment**: Gymnasium-compatible environment with dispatcher and acceptor agents
- **Action Masking**: Ensures agents only select valid actions (respecting queue capacities)
- **CTDE Architecture**: Centralized Training with Decentralized Execution for scalable multi-agent learning
- **Progressive Algorithm Development**: From tabular Q-learning to Multi-Agent PPO
- **Load Balancing**: Congestion penalties ensure balanced workstation utilization

## 🏗️ Environment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WAREHOUSE ENVIRONMENT                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    ┌──────────┐         ┌─────────────────────────────┐     │
│    │   Job    │         │        DISPATCHER           │     │
│    │ Arrival  │────────▶│   Actions: Route to S0,     │     │
│    │          │         │   Route to S1, or Reject    │     │
│    └──────────┘         └──────────┬──────────────────┘     │
│                                    │                         │
│                    ┌───────────────┼───────────────┐        │
│                    ▼               ▼               ▼        │
│            ┌──────────────┐ ┌──────────────┐              │
│            │  STATION 0   │ │  STATION 1   │   (Reject)   │
│            │  Acceptor    │ │  Acceptor    │              │
│            │              │ │              │              │
│            │ Queue: [■■■] │ │ Queue: [■■]  │              │
│            │ Proc:  [■]   │ │ Proc:  [■■]  │              │
│            └──────────────┘ └──────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Observation Space (7 dimensions)
| Index | Description |
|-------|-------------|
| 0 | Station 0 queue length (normalized) |
| 1 | Station 1 queue length (normalized) |
| 2 | Station 0 processing load (normalized) |
| 3 | Station 1 processing load (normalized) |
| 4 | Station 0 avg remaining time |
| 5 | Station 1 avg remaining time |
| 6 | Pending job flag (0 or 1) |

### Action Spaces
- **Dispatcher**: 3 actions (Route to Station 0, Route to Station 1, Reject)
- **Acceptors**: 2 actions each (Accept, Reject)

### Reward Structure
| Event | Reward |
|-------|--------|
| Successful enqueue | +1 |
| Job completion | +10 |
| Rejection (queue full) | -2 |
| Rejection (dispatcher) | -5 |
| Congestion penalty | -0.8 × excess load |

## 🧠 Algorithms Implemented

### 1. Tabular Q-Learning (Baseline)
- Joint action space: 3 × 2 × 2 = 12 actions
- Discretized continuous observations
- ε-greedy exploration with decay

### 2. Deep Q-Network (DQN)
- Neural network function approximation
- Experience replay buffer
- Target network for stability
- **Action masking** for constraint satisfaction

### 3. Proximal Policy Optimization (PPO)
- Actor-Critic architecture
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Action masking integrated

### 4. Multi-Agent PPO with CTDE
- **Centralized Training**: Shared critic network
- **Decentralized Execution**: Independent actor policies
- Sequential decision-making (dispatcher → acceptors)
- Cooperative reward structure

## 📊 Results

### Performance Comparison

| Algorithm | Jobs/Episode | Improvement over Baseline |
|-----------|-------------|---------------------------|
| Shortest-Queue Heuristic | 12-15 | - |
| Tabular Q-Learning | 20-21 | 1.5x |
| DQN + Action Masking | ~30 | 2.1x |
| PPO | ~33 | 2.3x |
| **Multi-Agent PPO (CTDE)** | **34-37** | **2.5x** |

### Training Curves

The Multi-Agent PPO achieves:
- Stable convergence after ~800 episodes
- Balanced load distribution across stations
- Consistent throughput during evaluation

### Key Findings

1. **Action Masking is Essential**: Prevents invalid actions and accelerates learning
2. **Congestion Penalty Tuning**: Increasing from 0.5 to 0.8 resolved station bias
3. **CTDE Benefits**: Reduced action space complexity from exponential to linear
4. **Sequential Decision-Making**: Eliminates agent conflicts in task acceptance

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/marl-warehouse-task-allocation.git
cd marl-warehouse-task-allocation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Training

```python
# Train Tabular Q-Learning
python src/train_tabular.py --episodes 500 --horizon 50

# Train DQN with Action Masking
python src/train_dqn.py --episodes 1000 --batch_size 64

# Train PPO
python src/train_ppo.py --episodes 2000 --lr 3e-4

# Train Multi-Agent PPO
python src/train_mappo.py --episodes 3000 --n_stations 2
```

### Evaluation

```python
# Evaluate trained model
python src/evaluate.py --model_path results/mappo_best.pt --episodes 100

# Generate visualizations
python src/visualize.py --results_dir results/
```

### Jupyter Notebooks

Interactive notebooks are provided in the `notebooks/` directory:
- `01_environment_exploration.ipynb`: Understand the warehouse environment
- `02_tabular_qlearning.ipynb`: Baseline implementation
- `03_dqn_training.ipynb`: DQN with action masking
- `04_ppo_training.ipynb`: PPO implementation
- `05_mappo_ctde.ipynb`: Multi-Agent PPO with CTDE

## 📁 Project Structure

```
marl-warehouse-task-allocation/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── warehouse_env.py          # Custom Gymnasium environment
│   │   └── utils.py                  # Environment utilities
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── tabular_agent.py          # Q-learning agent
│   │   ├── dqn_agent.py              # DQN with action masking
│   │   ├── ppo_agent.py              # PPO agent
│   │   └── mappo_agent.py            # Multi-Agent PPO
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── q_network.py              # Q-value network
│   │   ├── actor_critic.py           # Actor-Critic networks
│   │   └── shared_critic.py          # CTDE shared critic
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_tabular.py
│   │   ├── train_dqn.py
│   │   ├── train_ppo.py
│   │   └── train_mappo.py
│   └── utils/
│       ├── __init__.py
│       ├── replay_buffer.py
│       ├── action_masking.py
│       └── visualization.py
├── notebooks/
│   ├── 01_environment_exploration.ipynb
│   ├── 02_tabular_qlearning.ipynb
│   ├── 03_dqn_training.ipynb
│   ├── 04_ppo_training.ipynb
│   └── 05_mappo_ctde.ipynb
├── configs/
│   └── default_config.yaml
├── results/
│   ├── figures/
│   └── models/
└── docs/
    ├── checkpoint_report.pdf
    └── final_presentation.pdf
```

## 📚 References

### Academic Papers

1. Ali, A.M., Tirel, L., & Hashim, H.A. (2025). "Novel multi-agent action masked deep reinforcement learning for general industrial assembly lines balancing problems." *Journal of Automation and Intelligence*.

2. Shen, Y., McClosky, B., Durham, J.W., & Zavlanos, M.M. (2023). "Multi-Agent Reinforcement Learning for Resource Allocation in Large-Scale Robotic Warehouse Sortation Centers." *IEEE Conference on Robotics and Automation*.

3. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*.

4. Yu, C., et al. (2022). "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." *NeurIPS*.

### Frameworks & Libraries

- [Gymnasium](https://gymnasium.farama.org/) - RL environment interface
- [PettingZoo](https://pettingzoo.farama.org/) - Multi-agent RL environments
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms

## 🎓 Academic Context

This project was developed as part of:
- **Course**: CSE 546 - Reinforcement Learning (Fall 2025)
- **Institution**: University at Buffalo
- **Instructor**: Dr. Alina Vereshchaka
- **Purpose**: Final Course Project & MS Robotics Culminating Experience

## 👤 Author

**Rugved R. Kulkarni**
- MS Robotics, University at Buffalo
- Email: rk62@buffalo.edu
- LinkedIn: [Your LinkedIn]
- GitHub: [@your-github-username]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dr. Alina Vereshchaka for course guidance
- University at Buffalo CSE Department
- The Farama Foundation for Gymnasium and PettingZoo
- Amazon Robotics research papers for real-world context

---

⭐ If you find this project useful, please consider giving it a star!
